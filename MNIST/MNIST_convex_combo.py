import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
import pickle
from tensorflow.examples.tutorials.mnist import input_data
import argparse
mnist = input_data.read_data_sets(train_dir='mnist', one_hot=False)

parser = argparse.ArgumentParser(description='MNIST label corruption experiments')
parser.add_argument('--method', default='combo', type=str, choices=['ours', 'forward', 'forward_gold', 'ideal', 'confusion', 'combo'])
parser.add_argument('--corruption_type', default='flip_labels', type=str, choices=['uniform_mix', 'flip_labels'])
parser.add_argument('--lambda_choice', default='theirs', choices=['theirs', '1_minus_theirs', '0.5'])
args = parser.parse_args()
print(args)

print('CUDA available:', torch.cuda.is_available())

def prepare_data(corruption_matrix, gold_fraction=0.5, merge_valset=True):
    np.random.seed(1)

    mnist_images = np.copy(mnist.train.images)
    mnist_labels = np.copy(mnist.train.labels)
    if merge_valset:
        mnist_images = np.concatenate([mnist_images, np.copy(mnist.validation.images)], axis=0)
        mnist_labels = np.concatenate([mnist_labels, np.copy(mnist.validation.labels)])

    indices = np.arange(len(mnist_labels))
    np.random.shuffle(indices)

    mnist_images = mnist_images[indices]
    mnist_labels = mnist_labels[indices].astype(np.long)

    num_gold = int(len(mnist_labels)*gold_fraction)
    num_silver = len(mnist_labels) - num_gold

    for i in range(num_silver):
        mnist_labels[i] = np.random.choice(num_classes, p=corruption_matrix[mnist_labels[i]])

    dataset = {'x': mnist_images, 'y': mnist_labels}
    gold = {'x': dataset['x'][num_silver:], 'y': dataset['y'][num_silver:]}
    silver = {'x': dataset['x'][:num_silver], 'y': dataset['y'][:num_silver]}

    # for convex combo net
    iter = 0
    indices = np.arange(num_gold)
    while True:
        if len(np.unique(gold['y'][indices][:num_gold // 4])) == num_classes:
            gold_train = {'x': gold['x'][indices][num_gold // 4:], 'y': gold['y'][indices][num_gold // 4:]}
            gold_val = {'x': gold['x'][indices][:num_gold // 4], 'y': gold['y'][indices][:num_gold // 4]}
            print('Successfully split gold into a train and val set with all classes in the val set')
            break
        else:
            np.random.shuffle(indices)

        iter += 1
        if iter == 100:
            assert False, 'Failed to split gold data'

    return dataset, gold, num_gold, num_silver, gold_train, gold_val, silver


def uniform_mix_C(mixing_ratio):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)

def flip_labels_C(corruption_prob):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(1)

    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C

# //////////////////////// defining model ////////////////////////
learning_rate = 0.001
reg_str = 1e-6
num_epochs = 10
batch_size = 32
num_classes = 10


class ThreeLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
            )

        self.init_weights()

    def init_weights(self):
        self.main[0].weight.data.normal_(0, 1/np.sqrt(784))
        self.main[0].bias.data.zero_()
        self.main[2].weight.data.normal_(0, 1/np.sqrt(128))
        self.main[2].bias.data.zero_()
        self.main[4].weight.data.normal_(0, 1/np.sqrt(128))
        self.main[4].bias.data.zero_()


    def forward(self, x):
        return self.main(x)


def train_and_test(method='ours', corruption_level=0, gold_fraction=0.5, get_C=uniform_mix_C):
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    net = ThreeLayerNet().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=reg_str)

    C = get_C(corruption_level)

    dataset, gold, num_gold, num_silver, gold_train, gold_val, silver = prepare_data(C, gold_fraction)


    # //////////////////////// train net on clean ////////////////////////
    clean_net = ThreeLayerNet().cuda()
    optimizer_cn = torch.optim.Adam(clean_net.parameters(), lr=learning_rate, weight_decay=reg_str)

    num_examples = len(gold_train['y'])
    num_batches = num_examples//batch_size

    indices = np.arange(num_examples)
    for epoch in range(max(num_epochs, 1 + 2000//num_batches)):  # at least 2000 updates
        # shuffle data every epoch    
        np.random.shuffle(indices)

        for i in range(num_batches):
            offset = i * batch_size

            x_batch = gold_train['x'][indices[offset:offset + batch_size]]
            y_batch = gold_train['y'][indices[offset:offset + batch_size]]
            data, target = V(torch.from_numpy(x_batch).cuda()), V(torch.from_numpy(y_batch).cuda())

            # forward
            output = clean_net(data)

            # backward
            loss = F.cross_entropy(output, target)
            optimizer_cn.zero_grad()
            loss.backward()
            optimizer_cn.step()

    clean_net.eval()
    data, target = V(torch.from_numpy(mnist.test.images).cuda(), volatile=True),\
                   V(torch.from_numpy(mnist.test.labels.astype(np.long)).cuda(), volatile=True)

    output = clean_net(data)
    pred = output.data.max(1)[1]
    correct = pred.eq(target.data).sum()

    gold_only_acc = correct / len(mnist.test.labels)
    print('Gold only:', gold_only_acc)

    # //////////////////////// train for estimation ////////////////////////

    num_examples = num_silver
    num_batches = num_examples//batch_size

    indices = np.arange(num_examples)
    for epoch in range(num_epochs):
        # shuffle data every epoch    
        np.random.shuffle(indices)

        for i in range(num_batches):
            offset = i * batch_size

            x_batch = dataset['x'][indices[offset:offset + batch_size]]
            y_batch = dataset['y'][indices[offset:offset + batch_size]]
            data, target = V(torch.from_numpy(x_batch).cuda()), V(torch.from_numpy(y_batch).cuda())

            # forward
            output = net(data)

            # backward
            loss = F.cross_entropy(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    net.eval()
    data, target = V(torch.from_numpy(mnist.test.images).cuda(), volatile=True),\
                   V(torch.from_numpy(mnist.test.labels.astype(np.long)).cuda(), volatile=True)

    output = net(data)
    pred = output.data.max(1)[1]
    correct = pred.eq(target.data).sum()

    baseline_acc = correct / len(mnist.test.labels)


    # //////////////////////// estimate C ////////////////////////
    if method == 'ours' or method == 'combo':
        probs = F.softmax(net(V(torch.from_numpy(gold['x']).cuda(), volatile=True))).data.cpu().numpy()

        C_hat = np.zeros((num_classes,num_classes))
        for label in range(num_classes):
            indices = np.arange(len(gold['y']))[gold['y'] == label]
            C_hat[label] = np.mean(probs[indices], axis=0, keepdims=True)

    elif method == 'forward' or method == 'forward_gold':
        probs = F.softmax(net(V(torch.from_numpy(dataset['x']).cuda(), volatile=True))).data.cpu().numpy()

        C_hat = np.zeros((num_classes,num_classes))
        for label in range(num_classes):
            class_probs = probs[:,label]
            thresh = np.percentile(class_probs, 97, interpolation='higher')
            class_probs[class_probs >= thresh] = 0

            C_hat[label] = probs[np.argsort(class_probs)][-1]

    elif method == 'ideal': C_hat = C

    elif method == 'confusion':
        # directly estimate confusion matrix on gold
        probs = F.softmax(net(V(torch.from_numpy(gold['x']).cuda(), volatile=True))).data.cpu().numpy()
        preds = np.argmax(probs, axis=1)

        C_hat = np.zeros([num_classes, num_classes])

        for i in range(len(gold['y'])):
            C_hat[gold['y'][i], preds[i]] += 1

        C_hat /= (np.sum(C_hat, axis=1, keepdims=True) + 1e-7)


    print('True C:', np.round(C, decimals=3))
    print('C_hat:', np.round(C_hat, decimals=3))


    # //////////////////////// estimate lambda ////////////////////////
    
    # /////////// getting average precision of clean_net ///////////
    tp_count = np.zeros(num_classes)
    tp_fp_count = np.zeros(num_classes)

    for i in range(len(gold_val['y'])):
        data, target = gold_val['x'][i], np.array([gold_val['y'][i]])
        data, target = V(torch.FloatTensor(data).cuda(), volatile=True), \
                       V(torch.LongTensor(target).cuda(), volatile=True)

        # forward
        output = clean_net(data)

        # average precision
        pred = output.data.max(0)[1]
        batch_correct = pred.eq(target.data)
        for i in range(len(batch_correct)):
            tp_count[pred[i]] += batch_correct[i]
            tp_fp_count[pred[i]] += 1

    precisions = tp_count / (tp_fp_count + 1e-8)
    clean_ap = np.mean(precisions)

    # /////////// getting average precision of noisy labeling ///////////
    y_br = np.zeros(num_classes)
    y_tilde_br = np.zeros(num_classes)

    for i in range(len(gold['y'])):
        y_br[gold['y'][i]] += 1

    for i in range(len(silver['y'])):
        y_tilde_br[silver['y'][i]] += 1

    y_br /= np.sum(y_br)
    y_tilde_br += 1e-12
    y_tilde_br /= np.sum(y_tilde_br)

    # print(y_br)
    # print(y_tilde_br)

    # print(np.unique(silver['y']))

    C2_hat = (C_hat.T * y_br.reshape(1, num_classes)) / y_tilde_br.reshape(num_classes, 1)
    C2_hat += 1e-12
    C2_hat /= np.sum(C2_hat, axis=1)

    # print(C2_hat.sum(1))
    # print(C2_hat)

    # print(C_hat.sum(1))
    # print(C_hat)

    noisy_ap = np.mean(np.diag(C2_hat))

    print('Clean AP: {}, Noisy AP: {}'.format(clean_ap, noisy_ap))

    if args.lambda_choice == 'theirs':
        combo_lambda = float(clean_ap / (noisy_ap + clean_ap))
    elif args.lambda_choice == '1_minus_theirs':
        combo_lambda = 1 - float(clean_ap / (noisy_ap + clean_ap))
    elif args.lambda_choice == '0.5':
        combo_lambda = 0.5
    
    print('Combo Lambda: {}'.format(combo_lambda))

    C_hat = V(torch.eye(num_classes)).cuda()


    # //////////////////////// retrain with correction ////////////////////////
    net.train()
    net.init_weights()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=reg_str)

    if method == 'ours' or method == 'ideal' or method == 'confusion' or method == 'forward_gold':
        num_examples = dataset['y'].shape[0]
        num_batches = num_examples//batch_size

        indices = np.arange(num_examples)
        for epoch in range(num_epochs):
            np.random.shuffle(indices)

            for i in range(num_batches):
                offset = i * batch_size
                current_indices = indices[offset:offset + batch_size]

                data = dataset['x'][current_indices]
                target = dataset['y'][current_indices]

                gold_indices = current_indices >= num_silver
                silver_indices = current_indices < num_silver

                gold_len = np.sum(gold_indices)
                if gold_len > 0:
                    data_g, target_g = data[gold_indices], target[gold_indices]
                    data_g, target_g = V(torch.FloatTensor(data_g).cuda()),\
                                       V(torch.LongTensor(target_g).cuda())

                silver_len = np.sum(silver_indices)
                if silver_len > 0:
                    data_s, target_s = data[silver_indices], target[silver_indices]
                    data_s, target_s = V(torch.FloatTensor(data_s).cuda()),\
                                       V(torch.LongTensor(target_s).cuda())

                # forward
                loss_s = 0
                if silver_len > 0:
                    output_s = net(data_s)
                    pre1 = C_hat.t()[target_s]
                    pre2 = torch.mul(F.softmax(output_s), pre1)
                    loss_s = -(torch.log(pre2.sum(1))).sum(0)
                loss_g = 0
                if gold_len > 0:
                    output_g = net(data_g)
                    loss_g = F.cross_entropy(output_g, target_g, size_average=False)

                # backward
                loss = (loss_g + loss_s)/batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    elif method == 'combo':
        num_examples = dataset['y'].shape[0]
        num_batches = num_examples//batch_size

        for epoch in range(num_epochs):
            indices = np.arange(num_examples)
            np.random.shuffle(indices)

            for i in range(num_batches):
                offset = i * batch_size

                x_batch = dataset['x'][indices[offset:offset + batch_size]]
                y_batch = dataset['y'][indices[offset:offset + batch_size]]

                target_one_hot = np.zeros((len(y_batch), num_classes))
                target_one_hot[np.arange(len(y_batch)), y_batch] = 1
                target_one_hot = target_one_hot.astype(np.float32)

                data, target_one_hot = V(torch.from_numpy(x_batch).cuda()), V(torch.from_numpy(target_one_hot).cuda())

                target_soft = F.softmax(clean_net(data))
                target = (combo_lambda * target_one_hot) + ((1 - combo_lambda) * target_soft)

                # forward
                output = torch.mm(F.softmax(net(data)), C_hat)
                loss = -(target * torch.log(output)).sum(1).mean(0)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


    # //////////////////////// evaluate method ////////////////////////
    net.eval()
    data, target = V(torch.from_numpy(mnist.test.images).cuda(), volatile=True),\
                   V(torch.from_numpy(mnist.test.labels.astype(np.long)).cuda(), volatile=True)

    output = net(data)
    pred = output.data.max(1)[1]
    correct = pred.eq(target.data).sum()

    test_acc = correct / len(mnist.test.labels)

    # nudge garbage collector
    del dataset; del gold

    return test_acc, baseline_acc


# //////////////////////// run experiments ////////////////////////

corruption_fnctn = uniform_mix_C if args.corruption_type == 'uniform_mix' else flip_labels_C
filename = './' + args.method + '_' + args.corruption_type
results = {}
for gold_fraction in [0.001, 0.01, 0.05]:
    results[gold_fraction] = {}
    for corruption_level in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        test_acc, baseline_acc = train_and_test(args.method, corruption_level, gold_fraction, corruption_fnctn)
        results[gold_fraction][corruption_level] = {}
        results[gold_fraction][corruption_level]['method'] = test_acc
        results[gold_fraction][corruption_level]['baseline'] = baseline_acc
        print('Gold fraction:', gold_fraction, '| Corruption level:', corruption_level,
              '| Method acc:', results[gold_fraction][corruption_level]['method'],
              '| Baseline acc:', results[gold_fraction][corruption_level]['baseline'])
    print()
with open(filename, 'wb') as file:
    pickle.dump(results, file)
    print("Dumped results_ours in file: " + filename)
