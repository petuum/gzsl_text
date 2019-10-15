from constant import *
from collections import Counter
from sklearn.model_selection import train_test_split
import os
import numpy as np


def read_raw_inputs(filenames):
    def read_one_file(filename):
        words = []
        with open(os.path.join(INPUTS_DIR, filename)) as f:
            for line in f:
                words += line.strip().split()
        return words

    return [read_one_file(filename) for filename in filenames]


def read_raw_label_description():
    label_desc = dict()
    with open(LABEL_DESCRIPTION_PATH) as f:
        for line in f:
            label, desc = line.strip().split('\t')
            label_desc[label] = desc.split()
    return label_desc


def build_vocabulary(all_words, min_count=5):
    word_count = Counter(all_words)

    vocab = dict()
    with open(VOCAB_PATH, 'w') as f:
        i = 2   # 0 for padding, 1 for unknown
        for word, count in word_count.most_common():
            if count < min_count:
                break
            f.write(f'{word}\t{i}')
            vocab[word] = i
            i += 1
    return vocab


def texts_to_indices(texts, filenames, vocab, save_dir):
    for text, filename in zip(texts, filenames):
        indices = [vocab.get(word, 1) for word in text]
        indices = np.asarray(indices, dtype=int)
        write_file = os.path.join(save_dir, filename + '.npy')
        np.save(write_file, indices)


def preprocess_data(min_count=5):
    filenames = sorted(os.listdir(INPUTS_DIR))

    input_texts = read_raw_inputs(filenames)
    train_texts, dev_texts, train_files, dev_files = train_test_split(input_texts, filenames,
                                                                      train_size=0.8, random_state=12345)

    label_desc = read_raw_label_description()
    all_words = [word for text in train_texts for word in text]
    all_words += [word for desc in label_desc.values() for word in desc]

    vocab = build_vocabulary(all_words, min_count)

    texts_to_indices(train_texts, train_files, vocab, TRAIN_DIR)
    texts_to_indices(train_texts, train_files, vocab, DEV_DIR)


if __name__ == '__main__':
    preprocess_data()
