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


import argparse


def get_base_config():
    parser = argparse.ArgumentParser(description='Train base model (feature extractor)')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=8, type=int, help='training batch size')
    parser.add_argument('--eval_batch_size', default=8, type=int, help='evaluating batch size')
    parser.add_argument('--num_epochs', default=30, type=int, help='epochs of training base model')
    parser.add_argument('--max_note_len', default=2000, type=int, help='max number of words in each patient note')
    parser.add_argument('--loss', default='bce', type=str, help='loss function, default to BCELoss')
    parser.add_argument('--graph_encoder', default='gate', type=str,
                        help='which GNN to use, conv for GCN or gate for GRNN, default to gate')
    parser.add_argument('--class_margin', action='store_true', default=True, help='use LDAM loss')
    parser.add_argument('--C', default=2.0, type=float, help='margin value for LDAM loss')
    parser.add_argument('--save_model', action='store_true', default=True, help='save trained model')
    parser.add_argument('--evaluate', action='store_true', default=False, help='evaluate the trained model')
    parser.add_argument('--gpu_id', default=0, type=int, help='which gpu to use')
    args = parser.parse_args()
    args.gpu = f'cuda:{args.gpu_id}'
    return args


def get_gan_config():
    parser = argparse.ArgumentParser(description='Train GAN model')
    parser.add_argument('--lr', default=1e-4, type=float, help='GAN learning rate')
    parser.add_argument('--batch_size', default=8, type=int, help='GAN training batch size')
    parser.add_argument('--num_epochs', default=80, type=int, help='epochs of training GAN model')
    parser.add_argument('--max_note_len', default=2000, type=int, help='max number of words in each patient note')
    parser.add_argument('--loss', default='bce', type=str, help='loss function, default to BCELoss')
    parser.add_argument('--graph_encoder', default='gate', type=str,
                        help='which GNN to use, conv for GCN or gate for GRNN, default to gate')
    parser.add_argument('--class_margin', action='store_true', default=True, help='whether to use LDAM loss')
    parser.add_argument('--C', default=2.0, type=float, help='margin value for LDAM loss')
    parser.add_argument('--gpu_id', default=0, type=int, help='which gpu to use')
    parser.add_argument('--save_every', default=5, type=int, help='frequency of saving GAN model')
    # GAN related
    parser.add_argument('--pool_mode', default='max', type=str, help='which pooling of RNN label encoder')
    parser.add_argument('--critic_iters', default=5, type=int, help='iterations of discriminator per generator')
    parser.add_argument('--ndh', default=800, type=int, help='hidden size of discriminator')
    parser.add_argument('--ngh', default=800, type=int, help='hidden size of generator')
    parser.add_argument('--reg_ratio', default=0.0, type=float, help='loss ratio for keywords regularizer')
    parser.add_argument('--decoder', default='linear', type=str, help='decoder for generating keywords')
    parser.add_argument('--top_k', default=20, type=int, help='number of keywords to generate per note')
    parser.add_argument('--add_zero', action='store_true', default=False, help='add zeroshot coding when training GAN')
    args = parser.parse_args()
    args.gpu = f'cuda:{args.gpu_id}'
    return args


def get_finetune_config():
    parser = argparse.ArgumentParser(description='Finetune model based on GAN generated features')
    parser.add_argument('--lr', default=1e-5, type=float, help='Finetuning learning rate')
    parser.add_argument('--batch_size', default=32, type=int, help='training batch size')
    parser.add_argument('--eval_batch_size', default=32, type=int, help='evaluating batch size')
    parser.add_argument('--finetune_epochs', default=50, type=int, help='epochs of finetuning')
    parser.add_argument('--max_note_len', default=2000, type=int, help='max number of words in each patient note')
    parser.add_argument('--loss', default='bce', type=str, help='loss function, default to BCELoss')
    parser.add_argument('--graph_encoder', default='gate', type=str,
                        help='which GNN to use, conv for GCN or gate for GRNN, default to gate')
    parser.add_argument('--class_margin', action='store_true', default=True, help='whether to use LDAM loss')
    parser.add_argument('--C', default=2.0, type=float, help='margin value for LDAM loss')
    parser.add_argument('--l2_ratio', default=5e-4, type=float, help='Weight decay ratio')
    parser.add_argument('--gpu_id', default=0, type=int, help='which gpu to use')
    parser.add_argument('--eval_zero', action='store_true', default=False, help='evaluate fewshot or zeroshot labels')
    # GAN related
    parser.add_argument('--neg_iters', default=3, type=int, help='iterations of real feature per synthetic feature')
    parser.add_argument('--syn_num', default=256, type=int, help='number of synthetic feature per ICD code')
    parser.add_argument('--gan_batch_size', default=128, type=int, help='GAN features training batch size')
    parser.add_argument('--gan_epoch', default=60, type=int, help='use GAN trained at this epoch')
    parser.add_argument('--gan_lr', default=1e-4, type=float, help='use GAN trained with this learning rate')
    parser.add_argument('--pool_mode', default='max', type=str, help='which pooling of RNN label encoder')
    parser.add_argument('--critic_iters', default=5, type=int, help='iterations of discriminator per generator')
    parser.add_argument('--ndh', default=800, type=int, help='hidden size of discriminator')
    parser.add_argument('--ngh', default=800, type=int, help='hidden size of generator')
    parser.add_argument('--reg_ratio', default=0.0, type=float, help='loss ratio for keywords regularizer')
    parser.add_argument('--decoder', default='linear', type=str, help='decoder for generating keywords')
    parser.add_argument('--top_k', default=20, type=int, help='number of keywords to generate per note')
    parser.add_argument('--add_zero', action='store_true', default=False, help='add zeroshot coding when training GAN')
    args = parser.parse_args()
    args.gpu = f'cuda:{args.gpu_id}'
    return args
