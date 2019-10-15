import copy
import random
import sys
from collections import Counter, defaultdict, OrderedDict
from typing import Iterator

import numpy as np
import torch

from metrics import all_metrics


def seed(s=8888):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_feature_extractor(model_path, device=None):
    log(f'Loading pretrained model from {model_path}')
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    pretrained_dict = OrderedDict()

    for k in state_dict:
        if k.split('.')[0] in {'emb', 'conv_modules', 'proj', 'output_proj', 'graph_label_encoder'}:
            pretrained_dict[k] = state_dict[k].to(device)

    return pretrained_dict


class InfListIterator(Iterator):
    def __init__(self, l, step_size=1, shuffle=True):
        self.l = l
        self.step_size = step_size
        self.shuffle = shuffle
        self.idx = 0
        if self.shuffle:
            random.shuffle(self.l)

    def __len__(self):
        return len(self.l)

    def __next__(self):
        if self.idx == len(self):
            if self.shuffle:
                random.shuffle(self.l)
            self.idx = 0

        end = min(self.idx + self.step_size, len(self))
        items = self.l[self.idx: end]
        self.idx = end
        return items


inf_list_iterator = InfListIterator


def iterate_minibatch(train_notes, train_labels, train_masks, eval_label_size, batch_size=16, shuffle=True,
                      rtn_indices=False):
    assert len(train_notes) == len(train_labels)
    n = len(train_notes)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)

    num_batches = n // batch_size + (n % batch_size != 0)
    for i in range(num_batches):
        batch_indices = indices[i * batch_size: (i + 1) * batch_size]
        y = np.zeros((len(batch_indices), eval_label_size))
        y_indices = np.zeros((len(batch_indices), eval_label_size), dtype=int) - 1

        for row, b_idx in enumerate(batch_indices):
            c_idx = train_labels[b_idx]
            y[row,  c_idx] = 1
            y_indices[row, :len(c_idx)] = c_idx

        y = np.asarray(y, dtype=np.float32)
        y_indices = np.asarray(y_indices, dtype=int)
        x = train_notes[batch_indices]
        mask = train_masks[batch_indices]
        if rtn_indices:
            yield torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask), batch_indices
        else:
            yield torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask), torch.from_numpy(y_indices)


def simple_iterate_minibatch(train_feats, train_labels, batch_size=16, shuffle=True):
    assert len(train_feats) == len(train_labels)
    n = len(train_feats)
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    num_batches = n // batch_size + (n % batch_size != 0)
    for i in range(num_batches):
        batch_indices = indices[i * batch_size: (i + 1) * batch_size]
        yield batch_indices


def get_label_feat_list(x, y, step_size=8):
    label_data_list = defaultdict(list)

    data_id = 0
    for xx, yy in zip(x, y):
        label_data_list[yy].append(data_id)
        data_id += 1

    label_data_iterator = dict()
    for code in label_data_list:
        label_data_iterator[code] = inf_list_iterator(label_data_list[code], step_size=step_size)

    return label_data_iterator


def log(s, f=sys.stderr):
    if s[-1] != '\n':
        s += '\n'
    f.write(s)
    f.flush()


def labels_to_index_labels(all_labels):
    return dict(zip(sorted(all_labels), np.arange(len(all_labels))))


def log_eval_metrics(epoch, y_score, y_true, frequent_indices, few_shot_indices, zero_shot_indices,
                     train_losses, dev_losses, fz_only=False):
    if train_losses is not None and dev_losses is not None:
        log(f"Epoch {epoch}, train loss={np.mean(train_losses):.4f}, dev loss={np.mean(dev_losses):.4f}\n")

    if not fz_only:
        metrics = all_metrics(y_score, y_true)
        log(f"\tA: {metric_string(metrics)}\n")

        freq_metircs = all_metrics(y_score[:, frequent_indices], y_true[:, frequent_indices])
        log(f"\tS: {metric_string(freq_metircs)}\n")

    few_metircs = all_metrics(y_score[:, few_shot_indices], y_true[:, few_shot_indices])
    log(f"\tF: {metric_string(few_metircs)}\n")

    zero_metircs = all_metrics(y_score[:, zero_shot_indices], y_true[:, zero_shot_indices])
    log(f"\tZ: {metric_string(zero_metircs)}\n")

    if not fz_only:
        return metrics['f1_micro']
    else:
        return few_metircs['f1_micro'], zero_metircs['f1_micro']


def metric_string(metrics):
    mic_pre, mic_rec, mic_f1, mic_auc = metrics['pre_micro'], metrics['rec_micro'],\
                                        metrics['f1_micro'], metrics['auc_micro']
    mac_pre, mac_rec, mac_f1, mac_auc = metrics['pre_macro'], metrics['rec_macro'],\
                                        metrics['f1_macro'], metrics['auc_macro']
    auc_pr = metrics['auc_pr']

    string = f"mic pre={mic_pre * 100:.2f}, mic rec={mic_rec * 100:.2f}, " \
        f"mic f1={mic_f1 * 100:.2f}, mic auc={mic_auc * 100:.2f}, " \
        f"mac pre={mac_pre * 100:.2f}, mac rec={mac_rec * 100:.2f}, " \
        f"mac f1={mac_f1 * 100:.2f}, mac auc={mac_auc * 100:.2f}, " \
        f"auc-pr={auc_pr * 100:.2f}"

    return string


def filter_out_rare_labels(dev_y, label_indices):
    counter = Counter()
    for y in dev_y:
        for c in y:
            counter[c] += 1
    return [code for code in label_indices if counter[code] > 1]


def label_label_counter(labels):
    label_counter = Counter()

    for l in labels:
        for code in l:
            label_counter[code] += 1
    return label_counter


def split_labels_by_count(train_labels, dev_labels, train_label_set, dev_label_set, thresh=5):
    frequent_labels = set()
    few_shot_labels = set()
    zero_shot_labels = set()
    label_counter = label_label_counter(train_labels)
    dev_label_counter = label_label_counter(dev_labels)

    for code in dev_label_set:
        if dev_label_counter[code] < thresh:
            continue
        if code not in train_label_set:
            zero_shot_labels.add(code)
        elif label_counter[code] <= 5:
            few_shot_labels.add(code)
        else:
            frequent_labels.add(code)

    return frequent_labels, few_shot_labels, zero_shot_labels, label_counter


def targets_to_count(labels_to_targets, label_counter):
    target_count = np.zeros(len(labels_to_targets), dtype=np.float32)
    for code in labels_to_targets:
        t = labels_to_targets[code]
        if code in label_counter:
            target_count[t] = label_counter[code]

    log(f'Imbalance ratio={target_count.max()}')
    return target_count


def label_to_indices(labels, labels_to_targets):
    return list(set([labels_to_targets[code] for code in labels]))
