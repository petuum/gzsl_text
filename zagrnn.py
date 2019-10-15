from absl import flags, app


import numpy as np
import torch
import torch.nn as nn

from data_loader import Dataset, preload_data, load_label_adj_graph, load_label_description, load_word_embedding
from common_module import get_optimizer, get_scheduler
from helper import iterate_minibatch, log, labels_to_index_labels, targets_to_count, \
    label_to_indices, split_labels_by_count, log_eval_metrics
from models import ZAGRNN

FLAGS = flags.FLAGS

flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_float('C', 2.0, 'Label distribution aware margin')
flags.DEFINE_integer('max_input_len', 2000, 'Max number of words in input document')
flags.DEFINE_integer('epochs', 50, 'Number of epoch for training')
flags.DEFINE_integer('batch_size', 8, 'Batch size for training')
flags.DEFINE_integer('eval_batch_size', 8, 'Batch size for evaluation')
flags.DEFINE_string('graph_encoder', 'gate', 'Graph neural network for encoding label hierarchy')
flags.DEFINE_string('gpu', 'cuda:0', 'Which gpu to use')
flags.DEFINE_boolean('save_model', True, 'Whether to save model')


def train(lr=1e-3, batch_size=8, eval_batch_size=16, num_epochs=30, max_input_len=2000, graph_encoder='conv', C=0.,
          gpu='cuda:1', save_model=True):

    device = torch.device(gpu if torch.cuda.is_available() else "cpu")

    train_data = Dataset('train')
    dev_data = Dataset('dev')

    train_inputs, train_labels = train_data.get_data()
    dev_inputs, dev_labels = dev_data.get_data()

    log(f'Loaded {len(train_inputs)} train data, {len(dev_inputs)} dev data...')

    train_label_set = train_data.get_all_labels()
    dev_label_set = dev_data.get_all_labels()

    all_labels = train_label_set.union(dev_label_set)
    all_labels = sorted(all_labels)

    labels_to_targets = labels_to_index_labels(all_labels)
    dev_eval_label_size = len(labels_to_targets)

    frequent_labels, few_shot_labels, zero_shot_labels, labels_counter = \
        split_labels_by_count(train_labels, dev_labels, train_label_set, dev_label_set)

    frequent_indices = label_to_indices(frequent_labels, labels_to_targets)
    few_shot_indices = label_to_indices(few_shot_labels, labels_to_targets)
    zero_shot_indices = label_to_indices(zero_shot_labels, labels_to_targets)

    target_count = targets_to_count(labels_to_targets, labels_counter) if C > 0. else None
    eval_label_size = len(labels_to_targets)

    log('Preloading data in memory...')
    train_x, train_y, train_masks = preload_data(train_inputs, train_labels, labels_to_targets, max_input_len)
    dev_x, dev_y, dev_masks = preload_data(dev_inputs, dev_labels, labels_to_targets, max_input_len)

    word_emb = load_word_embedding()
    adj_matrix = load_label_adj_graph(labels_to_targets)
    label_idx_matrix, label_idx_mask = load_label_description(labels_to_targets)

    num_neighbors = torch.from_numpy(adj_matrix.sum(axis=1).astype(np.float32)).to(device)
    adj_matrix = torch.from_numpy(adj_matrix.astype(np.float32)).to_sparse().to(device)  # L x L
    loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

    log(f'Building model on {device}...')
    model = ZAGRNN(word_emb, label_idx_matrix, label_idx_mask, adj_matrix, num_neighbors, loss_fn, C=C,
                   graph_encoder=graph_encoder, eval_label_size=eval_label_size, target_count=target_count)
    model.to(device)

    log(f'Evaluating on {len(frequent_indices)} frequent labels, '
        f'{len(few_shot_indices)} few shot labels and {len(zero_shot_indices)} zero shot labels...')

    optimizer = get_optimizer(lr, model, weight_decay=1e-5)
    scheduler = get_scheduler(optimizer, num_epochs, ratios=(0.6, 0.85))

    log(f'Start training with {eval_label_size} labels')
    best_dev_f1 = 0.
    for epoch in range(num_epochs):
        train_losses = []

        # train one epoch
        with torch.set_grad_enabled(True):
            model.train()
            if gpu:
                torch.cuda.empty_cache()

            for batch in iterate_minibatch(train_x, train_y, train_masks, eval_label_size, batch_size, shuffle=True):
                x, y, mask = batch
                x, y, mask = x.to(device), y.to(device), mask.to(device)
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

            for batch in iterate_minibatch(dev_x, dev_y, dev_masks, eval_label_size,
                                           batch_size=eval_batch_size, shuffle=False):
                x, y, mask = batch
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                y_true.append(y.cpu().numpy()[:, :dev_eval_label_size])

                # forward pass
                logits, loss = model.forward(x, y, mask)
                probs = torch.sigmoid(logits[:, :dev_eval_label_size])

                # eval stats
                y_score.append(probs.cpu().numpy())
                dev_losses.append(loss.mean().data.cpu().numpy())

        y_score = np.vstack(y_score)
        y_true = np.vstack(y_true)
        dev_f1 = log_eval_metrics(epoch, y_score, y_true, frequent_indices, few_shot_indices, zero_shot_indices,
                                  train_losses, dev_losses)

        if epoch > int(num_epochs * 0.75) and dev_f1 > best_dev_f1 and save_model:
            best_dev_f1 = dev_f1
            torch.save(model.state_dict_to_save(), f"./models/{model.name}")

        # update lr
        scheduler.step(epoch)


def main(unused_argv):
    train(FLAGS.lr,  FLAGS.batch_size, FLAGS.eval_batch_size, FLAGS.epochs, FLAGS.max_input_len, FLAGS.graph_encoder,
          FLAGS.C, FLAGS.gpu, FLAGS.save_model)


if __name__ == '__main__':
    app.run(main)
