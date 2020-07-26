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


import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable


def calc_gradient_penalty(discriminator, real_feats, fake_feats, labels, lambda1=10.):
    b = real_feats.size(0)
    alpha = torch.rand(b, 1).to(real_feats.device)
    alpha = alpha.expand(real_feats.size())

    interpolates = alpha * real_feats + ((1 - alpha) * fake_feats)
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates, labels)

    ones = torch.ones(disc_interpolates.size()).to(real_feats.device)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda1
    return gradient_penalty


class ConditionalGenerator(nn.Module):
    def __init__(self, noise_size, label_size, hidden_size, output_size):
        super(ConditionalGenerator, self).__init__()
        self.fc1 = nn.Linear(noise_size + label_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, noise, label):
        x = torch.cat([noise, label], dim=-1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class ConditionalDiscriminator(nn.Module):
    def __init__(self, feat_size, label_size, hidden_size, output_size=1):
        super(ConditionalDiscriminator, self).__init__()
        self.fc1 = nn.Linear(feat_size + label_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, feat, label):
        x = torch.cat([feat, label], dim=-1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x.squeeze()
