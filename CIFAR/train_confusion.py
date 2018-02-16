# -*- coding: utf-8 -*-

import argparse
import os
import time
import math
import json
import torch
from torch.autograd import Variable as V
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import wideresnet as wrn
import numpy as np
from load_corrupted import CIFAR10, CIFAR100
from PIL import Image

np.random.seed(1)

# note: nosgdr, schedule, and epochs are highly related settings

parser = argparse.ArgumentParser(description='Trains WideResNet on CIFAR',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Positional arguments
parser.add_argument('data_path', type=str, help='Root for the Cifar dataset.')
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'],
    help='Choose between CIFAR-10, CIFAR-100.')
# Optimization options
parser.add_argument('--nosgdr', default=False, action='store_true', help='Turn off SGDR.')
parser.add_argument('--epochs', '-e', type=int, default=75, help='Number of epochs to train.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--gold_fraction', '-gf', type=float, default=0.1, help='What fraction of the data should be trusted?')
parser.add_argument('--corruption_prob', '-cprob', type=float, default=0.3, help='The label corruption probability.')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif', help='Type of corruption ("unif" or "flip").')
parser.add_argument('--adjust', '-a', action='store_true', help='Adjust the C_hat estimate with base-rate information.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--test_bs', type=int, default=128)
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs. Use when SGDR is off.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--nonlinearity', type=str, default='relu', help='Nonlinearity (relu, elu, gelu).')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
# i/o
parser.add_argument('--log', type=str, default='./', help='Log folder.')
args = parser.parse_args()


import socket
print()
print("This is on machine:", socket.gethostname())
print()
print(args)
print()

# Init logger
if not os.path.isdir(args.log):
    os.makedirs(args.log)
log = open(os.path.join(args.log, args.dataset + '_log.txt'), 'w')
state = {k: v for k, v in args._get_kwargs()}
state['tt'] = 0      # SGDR variable
state['init_learning_rate'] = args.learning_rate
log.write(json.dumps(state) + '\n')

# Init dataset
if not os.path.isdir(args.data_path):
    os.makedirs(args.data_path)

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
     transforms.Normalize(mean, std)])
test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)])

if args.dataset == 'cifar10':
    train_data_gold = CIFAR10(
        args.data_path, True, True, args.gold_fraction, args.corruption_prob, args.corruption_type,
        transform=train_transform, download=True)
    train_data_silver = CIFAR10(
        args.data_path, True, False, args.gold_fraction, args.corruption_prob, args.corruption_type,
        transform=train_transform, download=True, shuffle_indices=train_data_gold.shuffle_indices)
    train_data_gold_deterministic = CIFAR10(
        args.data_path, True, True, args.gold_fraction, args.corruption_prob, args.corruption_type,
        transform=test_transform, download=True, shuffle_indices=train_data_gold.shuffle_indices)
    test_data = CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 10

elif args.dataset == 'cifar100':
    train_data_gold = CIFAR100(
        args.data_path, True, True, args.gold_fraction, args.corruption_prob, args.corruption_type,
        transform=train_transform, download=True)
    train_data_silver = CIFAR100(
        args.data_path, True, False, args.gold_fraction, args.corruption_prob, args.corruption_type,
        transform=train_transform, download=True, shuffle_indices=train_data_gold.shuffle_indices)
    train_data_gold_deterministic = CIFAR100(
        args.data_path, True, True, args.gold_fraction, args.corruption_prob, args.corruption_type,
        transform=test_transform, download=True, shuffle_indices=train_data_gold.shuffle_indices)
    test_data = CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 100


class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, data_tensor, target_tensor, transform):
        # assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data_tensor[index], self.target_tensor[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return 50000

train_silver_loader = torch.utils.data.DataLoader(
    train_data_silver, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)
train_gold_deterministic_loader = torch.utils.data.DataLoader(
    train_data_gold_deterministic, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)
train_all_loader = torch.utils.data.DataLoader(
    TensorDataset(np.vstack((train_data_gold.train_data, train_data_silver.train_data)),
                  torch.from_numpy(np.array(train_data_gold.train_labels + train_data_silver.train_labels)),
                  train_transform),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)

# Init checkpoints
if not os.path.isdir(args.save):
    os.makedirs(args.save)

# Init model, criterion, and optimizer
net = wrn.WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
print(net)

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()

torch.manual_seed(1)
if args.ngpu > 0:
    torch.cuda.manual_seed(1)


optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                            weight_decay=state['decay'], nesterov=True)

# saving so we can start again from these same weights when applying the correction
torch.save(net.state_dict(), os.path.join(
    args.save, args.dataset+'_'+str(args.gold_fraction) + str(args.corruption_prob) + args.corruption_type + '_init.pytorch'))

# Restore model
start_epoch = 0
# if args.load != '':
#     for i in range(args.epochs-1,-1,-1):
#         model_name = os.path.join(args.load, args.dataset + '_model_epoch' + str(i) + '.pytorch')
#         if os.path.isfile(model_name):
#             net.load_state_dict(torch.load(model_name))
#             start_epoch = i+1
#             print('Model restored! Epoch:', i)
#             break
#     if start_epoch == 0:
#         assert False, "could not resume"


cudnn.benchmark = True  # fire on all cylinders


def train_phase1():
    net.train()     # enter train mode
    loss_avg = 0.0
    for batch_idx, (data, target) in enumerate(train_silver_loader):
        data, target = V(data.cuda()), V(target.cuda())

        # forward
        output = net(data)

        # backward
        optimizer.zero_grad()
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + loss.data[0] * 0.2

        if args.nosgdr is False:    # Use a cyclic learning rate
            dt = math.pi/float(args.epochs)
            state['tt'] += float(dt)/(len(train_silver_loader.dataset)/float(args.batch_size))
            if state['tt'] >= math.pi - 0.05:
                state['tt'] = math.pi - 0.05
            curT = math.pi/2.0 + state['tt']
            new_lr = args.learning_rate * (1.0 + math.sin(curT))/2.0    # lr_min = 0, lr_max = lr
            state['learning_rate'] = new_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['learning_rate']

    state['train_loss'] = loss_avg


# test function (forward only)
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = V(data.cuda(), volatile=True),\
                       V(target.cuda(), volatile=True)

        # forward
        output = net(data)
        loss = F.cross_entropy(output, target)

        # accuracy
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).sum()

        # test loss average
        loss_avg += loss.data[0]

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)


# Main loop
for epoch in range(start_epoch, args.epochs):
    # if epoch < 150:
    #     state['learning_rate'] = state['init_learning_rate']
    # elif epoch >= 150 and epoch < 225:
    #     state['learning_rate'] = state['init_learning_rate'] * args.gamma
    # elif epoch >= 225:
    #     state['learning_rate'] = state['init_learning_rate'] * (args.gamma ** 2)
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = state['learning_rate']
    state['epoch'] = epoch

    begin_epoch = time.time()
    train_phase1()
    print('Epoch', epoch, '| Time Spent:', round(time.time() - begin_epoch, 2))

    test()

    # torch.save(net.state_dict(), os.path.join(args.save, args.dataset + '_model_epoch' + str(epoch) + '.pytorch'))
    # Let us not waste space and delete the previous model
    # We do not overwrite the model because we need the epoch number
    # try: os.remove(os.path.join(args.save, args.dataset + '_model_epoch' + str(epoch-1) + '.pytorch'))
    # except: True    # prodigious programming form

    log.write('%s\n' % json.dumps(state))
    log.flush()
    print(state)


print('\nNow retraining with correction\n')


def get_C_hat_transpose():
    probs = []
    net.eval()
    for batch_idx, (data, target) in enumerate(train_gold_deterministic_loader):
        # we subtract 10 because we added 10 to gold so we could identify which example is gold in train_phase2
        data, target = torch.autograd.Variable(data.cuda(), volatile=True),\
                       torch.autograd.Variable((target - num_classes).cuda(), volatile=True)

        # forward
        output = net(data)
        pred = F.softmax(output)
        probs.extend(list(pred.data.cpu().numpy()))

    probs = np.array(probs, dtype=np.float32)
    preds = np.argmax(probs, axis=1)
    C_hat = np.zeros([num_classes, num_classes])
    for i in range(len(train_data_gold.train_labels)):
        C_hat[int(np.rint(train_data_gold.train_labels[i] - num_classes)), preds[i]] += 1

    C_hat /= (np.sum(C_hat, axis=1, keepdims=True) + 1e-7)
    C_hat = C_hat * 0.99 + np.full_like(C_hat, 1/num_classes) * 0.01  # smoothing

    return C_hat.T.astype(np.float32)

C_hat_transpose = torch.from_numpy(get_C_hat_transpose())
C_hat_transpose = V(C_hat_transpose.cuda(), requires_grad=False)


# /////// Resetting the network ////////
state = {k: v for k, v in args._get_kwargs()}
state['tt'] = 0      # SGDR variable
state['init_learning_rate'] = args.learning_rate
state['learning_rate'] = state['init_learning_rate']
for param_group in optimizer.param_groups:
    param_group['lr'] = state['learning_rate']

model_name = os.path.join(
    args.save,
    args.dataset+'_'+str(args.gold_fraction) + str(args.corruption_prob) + args.corruption_type + '_init.pytorch')
net.load_state_dict(torch.load(model_name))


def train_phase2(C_hat_transpose):
    net.train()     # enter train mode
    loss_avg = 0.0
    for batch_idx, (data, target) in enumerate(train_all_loader):
        # we subtract num_classes because we added num_classes to allow us to identify gold examples
        data, target = data.numpy(), target.numpy()

        gold_indices = target > (num_classes - 1)
        gold_len = np.sum(gold_indices)
        if gold_len > 0:
            data_g, target_g = data[gold_indices], target[gold_indices] - num_classes
            data_g, target_g = V(torch.FloatTensor(data_g).cuda()),\
                               V(torch.from_numpy(target_g).long().cuda())

        silver_indices = target < num_classes
        silver_len = np.sum(silver_indices)
        if silver_len > 0:
            data_s, target_s = data[silver_indices], target[silver_indices]

            data_s, target_s = V(torch.FloatTensor(data_s).cuda()),\
                               V(torch.from_numpy(target_s).long().cuda())

        optimizer.zero_grad()
        # forward

        loss_s = 0
        if silver_len > 0:
            output_s = net(data_s)
            pre1 = C_hat_transpose[torch.cuda.LongTensor(target_s.data)]
            pre2 = torch.mul(F.softmax(output_s), pre1)
            loss_s = -(torch.log(pre2.sum(1))).sum(0)
        loss_g = 0
        if gold_len > 0:
            output_g = net(data_g)
            loss_g = F.cross_entropy(output_g, target_g, size_average=False)

        # backward
        loss = (loss_g + loss_s)/args.batch_size
        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.2 + float(loss.cpu().data.numpy()[0]) * 0.8

        if args.nosgdr is False:    # Use a cyclic learning rate
            dt = math.pi/float(args.epochs)
            state['tt'] += float(dt)/(len(train_all_loader.dataset)/float(args.batch_size))
            if state['tt'] >= math.pi - 0.05:
                state['tt'] = math.pi - 0.05
            curT = math.pi/2.0 + state['tt']
            new_lr = args.learning_rate * (1.0 + math.sin(curT))/2.0    # lr_min = 0, lr_max = lr
            state['learning_rate'] = new_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['learning_rate']

    state['train_loss'] = loss_avg


# Main loop
for epoch in range(0, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()
    train_phase2(C_hat_transpose)
    print('Epoch', epoch, '| Time Spent:', round(time.time() - begin_epoch, 2))

    test()

    # torch.save(net.state_dict(), os.path.join(args.save, args.dataset + '_model_epoch' + str(epoch) + '.pytorch'))
    # Let us not waste space and delete the previous model
    # We do not overwrite the model because we need the epoch number
    # try: os.remove(os.path.join(args.save, args.dataset + '_model_epoch' + str(epoch-1) + '.pytorch'))
    # except: True    # prodigious programming form

    log.write('%s\n' % json.dumps(state))
    log.flush()
    print(state)

log.close()

try: os.remove(os.path.join(
    args.save,
    args.dataset+'_'+str(args.gold_fraction) + str(args.corruption_prob) + args.corruption_type + '_init.pytorch'))
except: True
