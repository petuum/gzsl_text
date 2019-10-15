import numpy as np
import os

from torch.utils import data
from preprocess_data import read_raw_label_description
from constant import *


class Dataset(data.Dataset):
    def __init__(self, mode):
        self.data_dir = f"{PROCESSED_DIR}/{mode}_files"
        names = [name.replace('.npy', '') for name in os.listdir(self.data_dir)]

        names = sorted(names)
        self.label_paths, self.input_paths = [], []
        self.names = names

        for name in names:
            self.label_paths.append(os.path.join(LABEL_DIR, name))
            self.input_paths.append(os.path.join(self.data_dir, f'{name}.npy'))

        assert len(self.label_paths) == len(self.input_paths), \
            'len of labels files is {} and is not equal to len of inputs files which is {}'.format(
                len(self.label_paths), len(self.input_paths))

        self.mode = mode
        self.labels = get_labels(self.label_paths)

    def get_data(self):
        return np.asarray(self.input_paths), np.asarray(self.labels)

    def get_all_labels(self):
        return set([c for l in self.labels for c in l])

    def __len__(self):
        " Deinputs the total number of samples "
        return len(self.label_paths)

    def __getitem__(self, index):
        " Generate one sample of data "
        # Load data
        X = np.load(self.input_paths[index])
        # get label
        y = self.labels[index]
        return X, y


def get_labels(paths):
    labels = []
    for p in paths:
        with open(p) as f:
            labels.append(f.readlines())
    return labels


def preload_data(train_inputs, train_labels, codes_to_targets, max_input_len=2000, idx_offset=1):
    x = []
    y = []
    mask = []
    row = 0
    for input_path, labels in zip(train_inputs, train_labels):
        code_idx = [codes_to_targets[code] for code in labels]
        y.append(code_idx)
        m = np.ones(max_input_len)
        xx = np.zeros(max_input_len)
        input = np.load(input_path) + idx_offset  # shift 1 for padding

        if len(input) < max_input_len:
            m[len(input):] = 0
            xx[:len(input)] = input
        else:
            xx = input[:max_input_len]

        x.append(xx)
        mask.append(m)
        row += 1

    x = np.vstack(x).astype(int)
    y = np.asarray(y)
    mask = np.vstack(mask).astype(np.float32)
    return x, y, mask


def load_label_adj_graph(labels_to_targets):
    num_labels = len(labels_to_targets)
    adj_matrix = np.zeros((num_labels, num_labels), dtype=np.float32)

    with open(LABEL_ADJ_PATH) as f:
        for line in f:
            node, neighbors = line.split('\t')
            neighbors = neighbors.split()
            for nei in neighbors:
                adj_matrix[labels_to_targets[node], labels_to_targets[nei]] = 1.0

    return adj_matrix


def load_label_description(labels_to_targets, max_len=30):
    raw_label_desc = read_raw_label_description()
    vocab = load_vocab()
    num_labels = len(labels_to_targets)
    label_idx_matrix = np.zeros((num_labels, max_len), dtype=np.int64)
    label_idx_mask = np.zeros((num_labels, max_len), dtype=np.float32)

    for label in labels_to_targets:
        desc_idx = [vocab.get(word, 1) for word in raw_label_desc[label]]
        if len(desc_idx) > max_len:
            desc_idx = max_len

        row = labels_to_targets[label]
        label_idx_matrix[row][:len(desc_idx)] = desc_idx
        label_idx_mask[row][:len(desc_idx)] = 1.0

    return label_idx_matrix, label_idx_mask


def load_word_embedding(emb_path=None, emb_dim=300):
    if emb_path is None:
        # load random embedding
        vocab = load_vocab()
        return np.random.uniform(-0.01, 0.01, (len(vocab), emb_dim)).astype(np.float32)

    with np.load(emb_path) as f:
        return f['arr_0']


def load_vocab():
    vocab = dict()
    with open(VOCAB_PATH, 'r') as f:
        for line in f:
            word, idx = line.replace('\n', '').split('\t')
            vocab[word] = int(idx)
    return vocab
