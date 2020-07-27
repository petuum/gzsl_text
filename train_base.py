# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pprint
import sys

import numpy as np
import torch

from config import get_base_config
from constant import CACHE_PATH, MODEL_DIR
from modules.icd_modules import ConvLabelAttnModel
from modules.common_module import get_optimizer, get_loss_fn, get_scheduler
from loaders.data_loader import Dataset, preload_data, load_adj_matrix
from loaders.emb_loader import init_emb_and_code_input
from loaders.model_loader import load_first_stage_model
from utils.helper import iterate_minibatch, log, codes_to_index_labels, targets_to_count, code_to_indices,\
    split_codes_by_count, log_eval_metrics


def train(lr=1e-3, batch_size=8, eval_batch_size=16, num_epochs=30, max_note_len=2000, loss='bce', gpu='cuda:1',
          save_model=True, graph_encoder='conv', class_margin=False, C=0.):
    pprint.pprint(locals(), stream=sys.stderr)
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")

    train_data = Dataset('train')
    dev_data = Dataset('dev')
    test_data = Dataset('test')

    train_notes, train_labels = train_data.get_data()
    dev_notes, dev_labels = dev_data.get_data()

    log(f'Loaded {len(train_notes)} train data, {len(dev_notes)} dev data...')
    n_train_data = len(train_notes)

    train_codes = train_data.get_all_codes()
    dev_codes = dev_data.get_all_codes()
    test_codes = test_data.get_all_codes()

    all_codes = train_codes.union(dev_codes).union(test_codes)
    all_codes = sorted(all_codes)

    codes_to_targets = codes_to_index_labels(all_codes, False)
    dev_eval_code_size = len(codes_to_targets)

    frequent_codes, few_shot_codes, zero_shot_codes, codes_counter = split_codes_by_count(train_labels, dev_labels,
                                                                                          train_codes, dev_codes)

    frequent_indices = code_to_indices(frequent_codes, codes_to_targets)
    few_shot_indices = code_to_indices(few_shot_codes, codes_to_targets)
    zero_shot_indices = code_to_indices(zero_shot_codes, codes_to_targets)

    extended_codes_to_targets, adj_matrix, codes_to_parents = load_adj_matrix(codes_to_targets)

    target_count = targets_to_count(codes_to_targets, codes_counter) if class_margin else None
    eval_code_size = len(codes_to_targets)

    log('Preloading data in memory...')
    train_x, train_y, train_masks = preload_data(train_notes, train_labels, codes_to_targets, max_note_len,
                                                 save_path=CACHE_PATH)
    dev_x, dev_y, dev_masks = preload_data(dev_notes, dev_labels, codes_to_targets, max_note_len)

    log(f'Building model on {device}...')
    word_emb, code_idx_matrix, code_idx_mask = init_emb_and_code_input(extended_codes_to_targets)

    num_neighbors = torch.from_numpy(adj_matrix.sum(axis=1).astype(np.float32)).to(device)
    adj_matrix = torch.from_numpy(adj_matrix.astype(np.float32)).to_sparse().to(device)  # L x L

    loss_fn = get_loss_fn(loss, reduction='sum')
    model = ConvLabelAttnModel(word_emb, code_idx_matrix, code_idx_mask, adj_matrix, num_neighbors, loss_fn,
                               graph_encoder=graph_encoder, eval_code_size=eval_code_size, target_count=target_count, C=C)
    model.to(device)

    log(f'Evaluating on {len(frequent_indices)} frequent codes, '
        f'{len(few_shot_indices)} few shot codes and {len(zero_shot_indices)} zero shot codes...')

    optimizer = get_optimizer(lr, model, weight_decay=1e-5)
    scheduler = get_scheduler(optimizer, num_epochs, ratios=(0.6, 0.85))

    log(f'Start training with {eval_code_size} codes')
    best_dev_f1 = 0.
    for epoch in range(num_epochs):
        train_losses = []

        # train one epoch
        with torch.set_grad_enabled(True):
            model.train()
            if gpu:
                torch.cuda.empty_cache()

            for batch in iterate_minibatch(train_x, train_y, train_masks, eval_code_size, batch_size, shuffle=True):
                x, y, mask, y_indices = batch
                x, y, mask, y_indices = x.to(device), y.to(device), mask.to(device), y_indices.to(device)
                optimizer.zero_grad()

                # forward pass
                logits, loss = model.forward(x, y, mask)

                # backward pass
                loss.mean().backward()
                optimizer.step()
                # train stats
                train_losses.append(loss.data.cpu().numpy())

        dev_losses = []
        y_true = []
        y_score = []
        with torch.set_grad_enabled(False):
            model.eval()
            if gpu:
                torch.cuda.empty_cache()

            for batch in iterate_minibatch(dev_x, dev_y, dev_masks, eval_code_size,
                                           batch_size=eval_batch_size, shuffle=False):
                x, y, mask, _ = batch
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                y_true.append(y.cpu().numpy()[:, :dev_eval_code_size])

                # forward pass
                logits, loss = model.forward(x, y, mask)
                probs = torch.sigmoid(logits[:, :dev_eval_code_size])

                # eval stats
                y_score.append(probs.cpu().numpy())
                dev_losses.append(loss.mean().data.cpu().numpy())

        y_score = np.vstack(y_score)
        y_true = np.vstack(y_true)
        dev_f1 = log_eval_metrics(epoch, y_score, y_true, frequent_indices, few_shot_indices, zero_shot_indices,
                                  train_losses, dev_losses)

        if epoch > int(num_epochs * 0.75) and dev_f1 > best_dev_f1 and save_model:
            best_dev_f1 = dev_f1
            torch.save(model.state_dict_to_save(), f"{MODEL_DIR}/{model.name}")

        # update lr
        scheduler.step(epoch)

    if save_model:
        torch.save(model.state_dict_to_save(), f"{MODEL_DIR}/final_{model.name}")


def eval_trained(eval_batch_size=16, max_note_len=2000, loss='bce', gpu='cuda:1', save_model=True,
                 graph_encoder='conv', class_margin=False, C=0.):
    pprint.pprint(locals(), stream=sys.stderr)
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")

    train_data = Dataset('train')
    dev_data = Dataset('dev')
    test_data = Dataset('test')

    train_notes, train_labels = train_data.get_data()
    dev_notes, dev_labels = dev_data.get_data()
    test_notes, test_labels = test_data.get_data()

    log(f'Loaded {len(train_notes)} train data, {len(dev_notes)} dev data...')
    n_train_data = len(train_notes)

    train_codes = train_data.get_all_codes()
    dev_codes = dev_data.get_all_codes()
    test_codes = test_data.get_all_codes()

    all_codes = train_codes.union(dev_codes).union(test_codes)
    all_codes = sorted(all_codes)

    codes_to_targets = codes_to_index_labels(all_codes, False)
    dev_eval_code_size = len(codes_to_targets)

    frequent_codes, few_shot_codes, zero_shot_codes, codes_counter = split_codes_by_count(train_labels, dev_labels,
                                                                                          train_codes, dev_codes)
    eval_code_size = len(codes_to_targets)
    dev_freq_codes, dev_few_shot_codes, dev_zero_shot_codes, codes_counter = \
        split_codes_by_count(train_labels, dev_labels, train_codes, dev_codes, 5)
    test_freq_codes, test_few_shot_codes, test_zero_shot_codes, _ = \
        split_codes_by_count(train_labels, test_labels, train_codes, test_codes, 5)

    dev_eval_indices = code_to_indices(dev_codes, codes_to_targets)
    dev_freq_indices = code_to_indices(dev_freq_codes, codes_to_targets)
    dev_few_shot_indices = code_to_indices(dev_few_shot_codes, codes_to_targets)
    dev_zero_shot_indices = code_to_indices(dev_zero_shot_codes, codes_to_targets)

    test_eval_indices = code_to_indices(test_codes, codes_to_targets)
    test_freq_indices = code_to_indices(test_freq_codes, codes_to_targets)
    test_few_shot_indices = code_to_indices(test_few_shot_codes, codes_to_targets)
    test_zero_shot_indices = code_to_indices(test_zero_shot_codes, codes_to_targets)

    extended_codes_to_targets, adj_matrix, codes_to_parents = load_adj_matrix(codes_to_targets)
    target_count = targets_to_count(codes_to_targets, codes_counter)

    eval_code_size = len(codes_to_targets)
    log(f'Building model on {device}...')
    word_emb, code_idx_matrix, code_idx_mask = init_emb_and_code_input(extended_codes_to_targets)

    num_neighbors = torch.from_numpy(adj_matrix.sum(axis=1).astype(np.float32)).to(device)
    adj_matrix = torch.from_numpy(adj_matrix.astype(np.float32)).to(device)  # L x L

    loss_fn = get_loss_fn(loss, reduction='sum')
    model = ConvLabelAttnModel(word_emb, code_idx_matrix, code_idx_mask, adj_matrix, num_neighbors, loss_fn,
                               graph_encoder=graph_encoder, eval_code_size=eval_code_size,
                               target_count=target_count if class_margin else None, C=C)

    pretrained_model_path = f"{MODEL_DIR}/{model.name}"
    pretrained_dict = load_first_stage_model(pretrained_model_path, device)

    model_dict = model.state_dict()
    for k in pretrained_dict:
        if k in model_dict:
            model_dict[k] = pretrained_dict[k]
    model.load_state_dict(model_dict)

    # set to sparse here
    model.adj_matrix = model.adj_matrix.to_sparse()
    model.to(device)

    log('Preloading data in memory...')
    dev_x, dev_y, dev_masks = preload_data(dev_notes, dev_labels, codes_to_targets, max_note_len)
    test_x, test_y, test_masks = preload_data(test_notes, test_labels, codes_to_targets, max_note_len)

    def eval_wrapper(x, y, masks):
        y_true = []
        y_score = []
        with torch.set_grad_enabled(False):
            model.eval()
            for batch in iterate_minibatch(x, y, masks, eval_code_size, batch_size=eval_batch_size, shuffle=False):
                x, y, mask, _ = batch
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                y_true.append(y.cpu().numpy()[:, :dev_eval_code_size])

                # forward pass
                logits, _ = model.forward(x, y, mask)
                probs = torch.sigmoid(logits[:, :dev_eval_code_size])

                # eval stats
                y_score.append(probs.cpu().numpy())

        y_score = np.vstack(y_score)
        y_true = np.vstack(y_true).astype(int)
        return y_true, y_score

    log('Evaluating on dev set...')
    dev_true, dev_score = eval_wrapper(dev_x, dev_y, dev_masks)
    log_eval_metrics(0, dev_score, dev_true, dev_freq_indices, dev_few_shot_indices, dev_zero_shot_indices)

    log('Evaluating on test set...')
    test_true, test_score = eval_wrapper(test_x, test_y, test_masks)
    log_eval_metrics(0, test_score, test_true, test_freq_indices, test_few_shot_indices, test_zero_shot_indices)


if __name__ == '__main__':
    config = get_base_config()
    if config.evaluate:
        eval_trained(eval_batch_size=config.eval_batch_size,
                     max_note_len=config.max_note_len,
                     loss=config.loss,
                     gpu=config.gpu,
                     save_model=config.save_model,
                     graph_encoder=config.graph_encoder,
                     class_margin=config.class_margin,
                     C=config.C)
    else:
        train(lr=config.lr,
              batch_size=config.batch_size,
              eval_batch_size=config.eval_batch_size,
              num_epochs=config.num_epochs,
              max_note_len=config.max_note_len,
              loss=config.loss,
              gpu=config.gpu,
              save_model=config.save_model,
              graph_encoder=config.graph_encoder,
              class_margin=config.class_margin,
              C=config.C)
