import pickle

import numpy as np
import torch
import tqdm

from data_loader import Dataset, load_label_description
from helper import log, labels_to_index_labels


def load_pretrained_word_emb(model_path):
    log(f'Loading pretrained model from {model_path}')
    state_dict = torch.load('./models/' + model_path, map_location=lambda storage, loc: storage)
    return state_dict['emb.embed.weight']


def cosine_similarity(a, b):
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.t())


def remove_duplicate(arr):
    _, idx = np.unique(arr, return_index=True)
    arr = arr[np.sort(idx)]
    return arr


def get_keywords_by_word_emb(model_path, gpu='cuda:1'):
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")

    word_emb = load_pretrained_word_emb(model_path)
    word_emb = word_emb.to(device)

    train_data = Dataset('train')
    dev_data = Dataset('dev')
    test_data = Dataset('test')

    train_notes, train_labels = train_data.get_data()
    train_label_set = train_data.get_all_labels()
    dev_label_set = dev_data.get_all_labels()
    test_label_set = test_data.get_all_labels()

    all_labels = train_label_set.union(dev_label_set).union(test_label_set)
    all_labels = sorted(all_labels)

    labels_to_targets = labels_to_index_labels(all_labels)
    label_sizes = len(all_labels)

    label_idx_matrix, label_idx_mask = load_label_description()
    label_idx_matrix = label_idx_matrix[:label_sizes].to(device)
    label_idx_mask = label_idx_mask[:label_sizes].to(device)

    label_emb = torch.embedding(word_emb, label_idx_matrix).transpose(1, 2).matmul(label_idx_mask.unsqueeze(2))
    label_emb = torch.div(label_emb.squeeze(2), torch.sum(label_idx_mask, dim=-1).unsqueeze(1))

    note_keywords = dict()
    n = len(train_notes)

    for i in tqdm.tqdm(range(n)):
        note_path, labels = train_notes[i], train_labels[i]
        note = np.load(note_path) + 1  # shift 1 for padding
        note = torch.from_numpy(note.astype(int)).to(device)
        note_word_emb = torch.embedding(word_emb, note)

        label_indices = []
        for label in labels:
            label_indices.append(labels_to_targets[label])

        label_emb = label_emb[torch.LongTensor(label_indices).to(device)]
        similarity = cosine_similarity(label_emb, note_word_emb)
        label_keywords = dict()

        for label, sim, emb in zip(label_indices, similarity, label_emb):
            sorted_position = torch.argsort(sim, descending=True)
            largest_indices = note[sorted_position].data.cpu().numpy()
            _, idx = np.unique(largest_indices, return_index=True)
            idx = np.sort(idx)
            word_indices = largest_indices[idx]
            scores = sim[sorted_position[idx]].data.cpu().numpy()

            pos_score_mask = ~np.isnan(scores) & (scores > 0.1)
            word_indices = word_indices[pos_score_mask]
            scores = scores[pos_score_mask]

            if len(scores) == 0:
                log(f'No pos words: {label}, {note_path}')

            label_keywords[label] = (word_indices, scores)

        note_keywords[note_path] = label_keywords

    with open('./data/note_attn_keywords.pkl', 'wb') as f:
        pickle.dump(note_keywords, f, -1)

    return note_keywords
