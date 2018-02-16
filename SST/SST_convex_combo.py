import numpy as np
import re
import collections
import pickle
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='sst label corruption experiments')
parser.add_argument('--method', default='combo', type=str, choices=['ours', 'forward', 'ideal', 'confusion', 'forward_gold', 'combo'])
parser.add_argument('--corruption_type', default='flip_labels', type=str, choices=['uniform_mix', 'flip_labels'])
parser.add_argument('--lambda_choice', default='theirs', choices=['theirs', '1_minus_theirs', '0.5'])
args = parser.parse_args()
print(args)

print('CUDA available:', torch.cuda.is_available())

def load_data(filename='./data/SST/senti.train.onlyroot'):
    '''
    :param filename: the system location of the data to load
    :return: the text (x) and its label (y)
             the text is a list of words and is not processed
    '''

    # stop words taken from nltk
    stop_words = ['i','me','my','myself','we','our','ours','ourselves','you','your','yours',
                  'yourself','yourselves','he','him','his','himself','she','her','hers','herself',
                  'it','its','itself','they','them','their','theirs','themselves','what','which',
                  'who','whom','this','that','these','those','am','is','are','was','were','be',
                  'been','being','have','has','had','having','do','does','did','doing','a','an',
                  'the','and','but','if','or','because','as','until','while','of','at','by','for',
                  'with','about','against','between','into','through','during','before','after',
                  'above','below','to','from','up','down','in','out','on','off','over','under',
                  'again','further','then','once','here','there','when','where','why','how','all',
                  'any','both','each','few','more','most','other','some','such','no','nor','not',
                  'only','own','same','so','than','too','very','s','t','can','will','just','don',
                  'should','now','d','ll','m','o','re','ve','y','ain','aren','couldn','didn',
                  'doesn','hadn','hasn','haven','isn','ma','mightn','mustn','needn','shan',
                  'shouldn','wasn','weren','won','wouldn']

    x, y = [], []
    with open(filename, "r") as f:
        for line in f:
            line = re.sub(r'\W+', ' ', line).strip().lower()  # perhaps don't make words lowercase?
            x.append(line[:-1])
            x[-1] = ' '.join(word for word in x[-1].split() if word not in stop_words)
            y.append(line[-1])
    return x, np.array(y, dtype=int)

def get_vocab(dataset):
    '''
    :param dataset: the text from load_data

    :return: a _ordered_ dictionary from words to counts
    '''
    vocab = {}

    # create a counter for each word
    for example in dataset:
        example_as_list = example.split()
        for word in example_as_list:
            vocab[word] = 0

    for example in dataset:
        example_as_list = example.split()
        for word in example_as_list:
            vocab[word] += 1

    # sort from greatest to least by count
    return collections.OrderedDict(sorted(vocab.items(), key=lambda x: x[1], reverse=True))

def text_to_rank(dataset, _vocab, desired_vocab_size=5000):
    '''
    :param dataset: the text from load_data
    :vocab: a _ordered_ dictionary of vocab words and counts from get_vocab
    :param desired_vocab_size: the desired vocabulary size
    words no longer in vocab become UUUNNNKKK
    :return: the text corpus with words mapped to their vocab rank,
    with all sufficiently infrequent words mapped to UUUNNNKKK; UUUNNNKKK has rank desired_vocab_size
    (the infrequent word cutoff is determined by desired_vocab size)
    '''
    _dataset = dataset[:]     # aliasing safeguard
    vocab_ordered = list(_vocab)
    count_cutoff = _vocab[vocab_ordered[desired_vocab_size-1]] # get word by its rank and map to its count

    word_to_rank = {}
    for i in range(len(vocab_ordered)):
        # we add one to make room for any future padding symbol with value 0
        word_to_rank[vocab_ordered[i]] = i + 1

    # we need to ensure that other words below the word on the edge of our desired_vocab size
    # are not also on the count cutoff, so we subtract a bit
    # this is likely quicker than adding another preventative if case
    for i in range(len(vocab_ordered[desired_vocab_size:])):
        _vocab[vocab_ordered[desired_vocab_size+i]] -= 0.1

    for i in range(len(_dataset)):
        example = _dataset[i]
        example_as_list = example.split()
        for j in range(len(example_as_list)):
            try:
                if _vocab[example_as_list[j]] >= count_cutoff:
                    example_as_list[j] = word_to_rank[example_as_list[j]]
                else:
                    example_as_list[j] = desired_vocab_size  # UUUNNNKKK
            except:
                example_as_list[j] = desired_vocab_size  # UUUNNNKKK
        _dataset[i] = example_as_list

    return _dataset

# taken from keras
def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    '''Pads each sequence to the same length:
    the length of the longest sequence.
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

# //////////////////////// loading data ////////////////////////

max_example_len = 30
batch_size = 50
embedding_dims = 100
vocab_size = 10000
init_lr = 5e-4
reg_str = 1e-5
num_epochs = 5

print('Loading Data')
X_train, Y_train = load_data('./data/SST/senti.binary.train')
X_dev, Y_dev = load_data('./data/SST/senti.binary.dev')
X_test, Y_test = load_data('./data/SST/senti.binary.test')
num_classes = 2

vocab = get_vocab(X_train)
X_train = text_to_rank(X_train, vocab, vocab_size)
X_dev = text_to_rank(X_dev, vocab, vocab_size)
X_test = text_to_rank(X_test, vocab, vocab_size)

X_train = np.array(pad_sequences(X_train, maxlen=max_example_len), dtype=np.long)
X_dev = np.array(pad_sequences(X_dev, maxlen=max_example_len), dtype=np.long)
X_test = np.array(pad_sequences(X_test, maxlen=max_example_len), dtype=np.long)

Y_train = np.array(Y_train, dtype=np.long)
Y_dev = np.array(Y_dev, dtype=np.long)
Y_test = np.array(Y_test, dtype=np.long)
print('Data loaded')


def prepare_data(corruption_matrix, gold_fraction=0.5, merge_valset=True):
    np.random.seed(1)

    examples = np.copy(X_train)
    labels = np.copy(Y_train)
    if merge_valset:
        examples = np.concatenate([examples, np.copy(X_dev)], axis=0)
        labels = np.concatenate([labels, np.copy(Y_dev)])

    indices = np.arange(len(labels))
    np.random.shuffle(indices)

    examples = examples[indices]
    labels = labels[indices]

    num_gold = int(len(labels)*gold_fraction)
    num_silver = len(labels) - num_gold

    for i in range(num_silver):
        labels[i] = np.random.choice(num_classes, p=corruption_matrix[labels[i]])

    dataset = {'x': examples, 'y': labels}
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

# //////////////////////// defining graph ////////////////////////

class WordAveragingLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size+1, embedding_dims, padding_idx=0)
        self.out = nn.Linear(embedding_dims, num_classes)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-np.sqrt(6. / (vocab_size+1 + embedding_dims)),
                                       np.sqrt(6. / (vocab_size+1 + embedding_dims)))
        self.out.weight.data.normal_(0, 1 / np.sqrt(embedding_dims))
        self.out.bias.data.zero_()

    def forward(self, x):
        return self.out(self.embedding(x.view(-1, max_example_len)).mean(1))


def train_and_test(method='ours', corruption_level=0, gold_fraction=0.5, get_C=uniform_mix_C):
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    net = WordAveragingLinear().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=init_lr, weight_decay=0)

    C = get_C(corruption_level)

    dataset, gold, num_gold, num_silver, gold_train, gold_val, silver = prepare_data(C, gold_fraction)


    # //////////////////////// train net on clean ////////////////////////
    clean_net = WordAveragingLinear().cuda()
    optimizer_cn = torch.optim.Adam(clean_net.parameters(), lr=init_lr, weight_decay=0)

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
            l2_loss = (clean_net.out.weight**2).sum() / 2
            loss = F.cross_entropy(output, target) + (reg_str * l2_loss)
            optimizer_cn.zero_grad()
            loss.backward()
            optimizer_cn.step()

    clean_net.eval()
    data, target = V(torch.from_numpy(X_test).cuda(), volatile=True),\
                   V(torch.from_numpy(Y_test.astype(np.long)).cuda(), volatile=True)

    output = clean_net(data)
    pred = output.data.max(1)[1]
    correct = pred.eq(target.data).sum()

    gold_only_acc = correct / len(Y_test)
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
            l2_loss = (net.out.weight**2).sum() / 2
            loss = F.cross_entropy(output, target) + (reg_str * l2_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    net.eval()
    data, target = V(torch.from_numpy(X_test).cuda(), volatile=True),\
                   V(torch.from_numpy(Y_test.astype(np.long)).cuda(), volatile=True)

    output = net(data)
    pred = output.data.max(1)[1]
    correct = pred.eq(target.data).sum()

    baseline_acc = correct / len(Y_test)


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
        data, target = V(torch.from_numpy(data).cuda(), volatile=True), \
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
    optimizer = torch.optim.Adam(net.parameters(), lr=init_lr, weight_decay=0)

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
                    data_g, target_g = V(torch.LongTensor(data_g).cuda()),\
                                       V(torch.from_numpy(target_g).long().cuda())

                silver_len = np.sum(silver_indices)
                if silver_len > 0:
                    data_s, target_s = data[silver_indices], target[silver_indices]
                    data_s, target_s = V(torch.LongTensor(data_s).cuda()),\
                                       V(torch.from_numpy(target_s).long().cuda())

                # forward
                loss_s = 0
                if silver_len > 0:
                    output_s = net(data_s)
                    output_s -= torch.max(output_s, 1, keepdim=True)[0]
                    output_s = torch.log(torch.mm(F.softmax(output_s), C_hat))
                    loss_s = F.cross_entropy(output_s, target_s, size_average=False)
                    # pre1 = C_hat.t()[torch.cuda.LongTensor(target_s.data)]
                    # pre2 = torch.mul(F.softmax(output_s), pre1)
                    # loss_s = -(torch.log(pre2.sum(1))).sum(0)
                loss_g = 0
                if gold_len > 0:
                    output_g = net(data_g)
                    loss_g = F.cross_entropy(output_g, target_g, size_average=False)

                # backward
                l2_loss = (net.out.weight**2).sum() / 2
                loss = (loss_g + loss_s)/batch_size + (reg_str * l2_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    elif method == 'combo':
        num_examples = dataset['y'].shape[0]
        num_batches = num_examples//batch_size

        indices = np.arange(num_examples)
        for epoch in range(num_epochs):
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
                l2_loss = (net.out.weight**2).sum() / 2
                loss = -(target * torch.log(output)).sum(1).mean(0) + (reg_str * l2_loss)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


    # //////////////////////// evaluate method ////////////////////////
    net.eval()
    data, target = V(torch.from_numpy(X_test).cuda(), volatile=True),\
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
