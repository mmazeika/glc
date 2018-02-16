import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
import pickle
import argparse
from helper_functions_twitter import *

parser = argparse.ArgumentParser(description='Twitter label corruption experiments')
parser.add_argument('--method', default='confusion', type=str, choices=['ours', 'forward', 'forward_gold', 'ideal', 'confusion'])
parser.add_argument('--corruption_type', default='flip_labels', type=str, choices=['uniform_mix', 'flip_labels'])
args = parser.parse_args()
print(args)

window_size = 1

# note that we encode the tags with numbers for later convenience
tag_to_number = {
    u'N': 0, u'O': 1, u'S': 2, u'^': 3, u'Z': 4, u'L': 5, u'M': 6,
    u'V': 7, u'A': 8, u'R': 9, u'!': 10, u'D': 11, u'P': 12, u'&': 13, u'T': 14,
    u'X': 15, u'Y': 16, u'#': 17, u'@': 18, u'~': 19, u'U': 20, u'E': 21, u'$': 22,
    u',': 23, u'G': 24
}

embeddings = embeddings_to_dict('./data/Tweets/embeddings-twitter.txt')
vocab = embeddings.keys()

# we replace <s> with </s> since it has no embedding, and </s> is a better embedding than UNK
X_train, Y_train = data_to_mat('./data/Tweets/tweets-train.txt', vocab, tag_to_number, window_size=window_size,
                     start_symbol=u'</s>')
X_dev, Y_dev = data_to_mat('./data/Tweets/tweets-dev.txt', vocab, tag_to_number, window_size=window_size,
                         start_symbol=u'</s>')
X_test, Y_test = data_to_mat('./data/Tweets/tweets-devtest.txt', vocab, tag_to_number, window_size=window_size,
                             start_symbol=u'</s>')


def prepare_data(corruption_matrix, gold_fraction=0.5, merge_valset=True):
    np.random.seed(1)

    twitter_tweets = np.copy(X_train)
    twitter_labels = np.copy(Y_train)
    if merge_valset:
        twitter_tweets = np.concatenate([twitter_tweets, np.copy(X_dev)], axis=0)
        twitter_labels = np.concatenate([twitter_labels, np.copy(Y_dev)])

    indices = np.arange(len(twitter_labels))
    np.random.shuffle(indices)

    twitter_tweets = twitter_tweets[indices]
    twitter_labels = twitter_labels[indices].astype(np.long)

    num_gold = int(len(twitter_labels)*gold_fraction)
    num_silver = len(twitter_labels) - num_gold

    for i in range(num_silver):
        twitter_labels[i] = np.random.choice(num_classes, p=corruption_matrix[twitter_labels[i]])

    dataset = {'x': twitter_tweets, 'y': twitter_labels}
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

reg_str = 5e-5
num_epochs = 15
num_classes = 25
hidden_size = 256
batch_size = 64
embedding_dimension = 50
example_size = (2*window_size + 1)*embedding_dimension
init_lr = 0.001
num_examples = Y_train.shape[0]
num_batches = num_examples//batch_size


# //////////////////////// defining graph ////////////////////////
class ThreeLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(example_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
            )

        self.init_weights()

    def init_weights(self):
        self.main[0].weight.data.normal_(0, 1/np.sqrt(example_size))
        self.main[0].bias.data.zero_()
        self.main[2].weight.data.normal_(0, 1/np.sqrt(256))
        self.main[2].bias.data.zero_()
        self.main[4].weight.data.normal_(0, 1/np.sqrt(256))
        self.main[4].bias.data.zero_()


    def forward(self, x):
        return self.main(x)


to_embeds = lambda x: word_list_to_embedding(x, embeddings, embedding_dimension)

def train_and_test(method='ours', corruption_level=0, gold_fraction=0.5, get_C=uniform_mix_C):
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    net = ThreeLayerNet().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=reg_str)

    C = get_C(corruption_level)

    dataset, gold, num_gold, num_silver = prepare_data(C, gold_fraction)


    # //////////////////////// train for estimation ////////////////////////

    if method == 'ours' or method == 'confusion' or method == 'forward_gold':
        num_examples = num_silver
    elif method == 'forward':
        num_examples = dataset['y'].shape[0]

    num_batches = num_examples//batch_size

    indices = np.arange(num_examples)
    for epoch in range(num_epochs):
        # shuffle data indices every epoch
        np.random.shuffle(indices)

        for i in range(num_batches):
            offset = i * batch_size

            x_batch = to_embeds(dataset['x'][indices[offset:offset + batch_size]])
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
    data, target = V(torch.from_numpy(to_embeds(X_test)).cuda(), volatile=True),\
                   V(torch.from_numpy(Y_test.astype(np.long)).cuda(), volatile=True)

    output = net(data)
    pred = output.data.max(1)[1]
    correct = pred.eq(target.data).sum()

    baseline_acc = correct / len(Y_test)


    # //////////////////////// estimate C ////////////////////////
    if method == 'ours':
        probs = F.softmax(net(V(torch.from_numpy(to_embeds(gold['x'])).cuda(), volatile=True))).data.cpu().numpy()

        C_hat = np.zeros((num_classes,num_classes))
        for label in range(num_classes):
            indices = np.arange(len(gold['y']))[gold['y'] == label]
            if indices.size == 0:
                C_hat[label] = np.ones(num_classes) / num_classes  # TODO: try a diagonal prior instead
            else:
                C_hat[label] = np.mean(probs[indices], axis=0, keepdims=True)

    elif method == 'forward' or method == 'forward_gold':
        probs = F.softmax(net(V(torch.from_numpy(to_embeds(dataset['x'])).cuda(), volatile=True))).data.cpu().numpy()

        C_hat = np.zeros((num_classes,num_classes))
        for label in range(num_classes):
            class_probs = probs[:,label]
            thresh = np.percentile(class_probs, 97, interpolation='higher')
            class_probs[class_probs >= thresh] = 0

            C_hat[label] = probs[np.argsort(class_probs)][-1]

    elif method == 'ideal': C_hat = C

    elif method == 'confusion':
        # directly estimate confusion matrix on gold
        probs = F.softmax(net(V(torch.from_numpy(to_embeds(gold['x'])).cuda(), volatile=True))).data.cpu().numpy()
        preds = np.argmax(probs, axis=1)

        C_hat = np.zeros([num_classes, num_classes])

        for i in range(len(gold['y'])):
            C_hat[gold['y'][i], preds[i]] += 1

        C_hat /= (np.sum(C_hat, axis=1, keepdims=True) + 1e-7)

        C_hat = C_hat * 0.99 + np.full_like(C_hat, 1/num_classes) * 0.01


    #print('True C:', np.round(C, decimals=3))
    #print('C_hat:', np.round(C_hat, decimals=3))

    C_hat = V(torch.from_numpy(C_hat.astype(np.float32))).cuda()


    # //////////////////////// retrain with correction ////////////////////////
    net.train()
    net.init_weights()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=reg_str)

    if method == 'ours' or method == 'ideal' or method == 'confusion' or method == 'forward_gold':
        num_examples = dataset['y'].shape[0]
        num_batches = num_examples//batch_size

        indices = np.arange(num_examples)
        for epoch in range(num_epochs):
            np.random.shuffle(indices)

            for i in range(num_batches):
                offset = i * batch_size
                current_indices = indices[offset:offset + batch_size]

                data = to_embeds(dataset['x'][current_indices])
                target = dataset['y'][current_indices]

                gold_indices = current_indices >= num_silver
                silver_indices = current_indices < num_silver

                gold_len = np.sum(gold_indices)
                if gold_len > 0:
                    data_g, target_g = data[gold_indices], target[gold_indices]
                    data_g, target_g = V(torch.FloatTensor(data_g).cuda()),\
                                       V(torch.from_numpy(target_g).long().cuda())

                silver_len = np.sum(silver_indices)
                if silver_len > 0:
                    data_s, target_s = data[silver_indices], target[silver_indices]
                    data_s, target_s = V(torch.FloatTensor(data_s).cuda()),\
                                       V(torch.from_numpy(target_s).long().cuda())

                # forward
                loss_s = 0
                if silver_len > 0:
                    output_s = net(data_s)
                    pre1 = C_hat.t()[torch.cuda.LongTensor(target_s.data)]
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

    elif method == 'forward':
        num_examples = dataset['y'].shape[0]
        num_batches = num_examples//batch_size

        for epoch in range(num_epochs):
            indices = np.arange(num_examples)
            np.random.shuffle(indices)

            for i in range(num_batches):
                offset = i * batch_size

                x_batch = to_embeds(dataset['x'][indices[offset:offset + batch_size]])
                y_batch = dataset['y'][indices[offset:offset + batch_size]]
                data, target = V(torch.from_numpy(x_batch).cuda()), V(torch.from_numpy(y_batch).cuda())

                # forward
                output = net(data)
                pre1 = C_hat.t()[torch.cuda.LongTensor(target.data)]
                pre2 = torch.mul(F.softmax(output), pre1)
                loss = -(torch.log(pre2.sum(1))).mean(0)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


    # //////////////////////// evaluate method ////////////////////////
    net.eval()
    data, target = V(torch.from_numpy(to_embeds(X_test)).cuda(), volatile=True), \
                   V(torch.from_numpy(Y_test.astype(np.long)).cuda(), volatile=True)

    output = net(data)
    pred = output.data.max(1)[1]
    correct = pred.eq(target.data).sum()

    test_acc = correct / len(Y_test)

    # nudge garbage collector
    del dataset; del gold

    return test_acc, baseline_acc


# //////////////////////// run experiments ////////////////////////

corruption_fnctn = uniform_mix_C if args.corruption_type == 'uniform_mix' else flip_labels_C
filename = './' + args.method + '_' + args.corruption_type
results = {}
for gold_fraction in [0.01, 0.05, 0.25]:
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
