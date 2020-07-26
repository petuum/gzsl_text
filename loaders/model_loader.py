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


from collections import OrderedDict

import torch

from constant import MODEL_DIR
from modules.generative_module import ConditionalGenerator, ConditionalDiscriminator
from modules.lm_module import RNNLabelEncoder
from utils.helper import log

RNN_POOL = {'mean': 'ar', 'max': 'mx'}


def load_first_stage_model(model_path, device=None):
    log(f'Loading pretrained model from {model_path}')
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    pretrained_dict = OrderedDict()
    for k in state_dict:
        if k.split('.')[0] in {'emb', 'conv_modules', 'proj', 'output_proj', 'graph_label_encoder'}:
            pretrained_dict[k] = state_dict[k].to(device)
    return pretrained_dict


def get_gan_model_name(lr, ndh, ngh, critic_iters, reg_ratio, decoder, top_k, add_zero, pool_mode):
    gan_hyper = f'gan_lr{lr}_ndh{ndh}_ngh{ngh}_diter{critic_iters}'

    if reg_ratio > 0.:
        gan_hyper += f'_dec{decoder}{reg_ratio}_kw{top_k}'

    if add_zero:
        gan_hyper += '_z'

    if pool_mode in RNN_POOL:
        gan_hyper += f'_{RNN_POOL[pool_mode]}'

    log(f'{gan_hyper}')
    return gan_hyper


def load_gan_model(base_model_name, noise_size=100, feat_size=200, epoch=30, ndh=256, ngh=512, critic_iters=5,
                   lr=2e-5, reg_ratio=0., decoder='linear', top_k=30, add_zero=False, pool_mode='last'):
    gan_hyper = get_gan_model_name(lr, ndh, ngh, critic_iters, reg_ratio, decoder, top_k, add_zero, pool_mode)
    model_path = f'epoch{epoch}_{gan_hyper}_{base_model_name}'

    log(f'Loading GAN from {model_path}...')
    gan_states = torch.load(f'{MODEL_DIR}/{model_path}', map_location=lambda storage, loc: storage)

    generator = ConditionalGenerator(noise_size, noise_size, ngh, feat_size)
    discriminator = ConditionalDiscriminator(feat_size, noise_size, ndh, 1)

    word_emb = gan_states['label_rnn']['encoder.weight'].data.numpy()
    label_rnn = RNNLabelEncoder(word_emb)
    label_rnn.load_state_dict(gan_states['label_rnn'])

    generator.load_state_dict(gan_states['generator'])
    discriminator.load_state_dict(gan_states['discriminator'])

    for m in [generator, discriminator, label_rnn]:
        for p in m.parameters():
            p.requires_grad = False
    return generator, discriminator, label_rnn, model_path
