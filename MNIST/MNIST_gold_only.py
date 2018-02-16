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
parser.add_argument('--method', default='gold_only', type=str, choices=['gold_only'])
parser.add_argument('--corruption_type', default='flip_labels', type=str, choices=['uniform_mix', 'flip_labels'])
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

    return dataset, gold, num_gold, num_silver

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
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=reg_str)

    C = get_C(corruption_level)

    dataset, gold, num_gold, num_silver = prepare_data(C, gold_fraction)


    # //////////////////////// train for estimation ////////////////////////

    num_examples = num_gold
    num_batches = num_examples//batch_size

    indices = np.arange(num_examples)
    for epoch in range(max(num_epochs, 1 + 2000//num_batches)):  # at least 2000 updates
        # shuffle data every epoch    
        np.random.shuffle(indices)

        for i in range(num_batches):
            offset = i * batch_size

            x_batch = gold['x'][indices[offset:offset + batch_size]]
            y_batch = gold['y'][indices[offset:offset + batch_size]]
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

    test_acc = correct / len(mnist.test.labels)
    baseline_acc = 0  # placeholder

    # nudge garbage collector
    del dataset; del gold

    return test_acc, baseline_acc


# //////////////////////// run experiments ////////////////////////

corruption_fnctn = uniform_mix_C if args.corruption_type == 'uniform_mix' else flip_labels_C
filename = './' + args.method
results = {}
for gold_fraction in [0.001, 0.01, 0.05]:
    results[gold_fraction] = {}
    for corruption_level in [0]:
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
