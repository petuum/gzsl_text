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


def reshape_input(x, n):
    x = x.transpose(1, 2)  # B x F x N
    x = x.contiguous().view(-1, n)
    return x.t()    # N x (BF)


def reshape_output(x, b, n):
    x = x.t()  # (B x F) x N
    x = x.view(b, -1, n)
    return x.transpose(1, 2)


class ConvGraphEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_nodes, concat=True, sparse=True):
        super(ConvGraphEncoder, self).__init__()
        self.sparse = sparse
        self.N = n_nodes
        self.gc1 = ConvGraphLayer(input_size, hidden_size, sparse)
        self.gc2 = ConvGraphLayer(hidden_size, hidden_size, sparse)
        self.concat = concat
        self.output_size = hidden_size
        if concat:
            self.output_size += hidden_size

    def forward(self, x, adj_matrix, num_neighbors):
        f = self.gc1(x, adj_matrix, num_neighbors)
        f = self.gc2(f, adj_matrix, num_neighbors)
        if self.concat:
            return torch.cat([f, x], dim=-1)
        else:
            return f


class ConvGraphLayer(nn.Module):
    def __init__(self, input_size, hidden_size, sparse=True, activation=torch.relu):
        super(ConvGraphLayer, self).__init__()
        self.fc = nn.Linear(input_size * 2, hidden_size)
        self.sparse = sparse
        self.activation = activation

    def forward(self, x, adj_matrix, num_neighbors=None):
        dim = x.dim()   # dim=3 for input, 2 for label
        if dim == 3:
            if self.sparse:
                b, n = x.size(0), x.size(1)
                neighbors = torch.spmm(adj_matrix, reshape_input(x, n))
                neighbors = reshape_output(neighbors, b, n)
            else:
                neighbors = torch.matmul(x.transpose(1, 2), adj_matrix).transpose(1, 2)
        else:
            neighbors = adj_matrix.matmul(x)

        if num_neighbors is not None:
            neighbors = torch.div(neighbors, num_neighbors.unsqueeze(1) + 1e-7)

        self_and_neighbor = torch.cat([x, neighbors], dim=-1)

        output = self.fc(self_and_neighbor)
        if self.activation is not None:
            output = self.activation(output)

        return output


class GraphGatedEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_nodes, concat=True, steps=2):
        super(GraphGatedEncoder, self).__init__()
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.steps = steps
        self.concat = concat
        self.output_size = hidden_size
        if concat:
            self.output_size += hidden_size

    @staticmethod
    def reshape_for_gru(x):
        x = x.contiguous().view(-1, x.size(-1))
        return x

    @staticmethod
    def reshape_from_gru(x, b):
        x = x.contiguous().view(b, -1, x.size(-1))
        return x

    def forward(self, x, adj_matrix, num_neighbors=None):
        dim = x.dim()   # dim=3 for input, 2 for label
        h = x
        b = x.size(0)
        for i in range(self.steps):
            if dim == 3:
                b, n = h.size(0), h.size(1)
                u = torch.spmm(adj_matrix, reshape_input(h, n))
                u = reshape_output(u, b, n)
            else:
                u = adj_matrix.matmul(h)

            if num_neighbors is not None:
                u = torch.div(u, num_neighbors.unsqueeze(1) + 1e-7)

            if dim == 3:
                u, h = self.reshape_for_gru(u), self.reshape_for_gru(h)

            h = self.gru(u, h)
            if dim == 3:
                u, h = self.reshape_from_gru(u, b), self.reshape_from_gru(h, b)

        if self.concat:
            return torch.cat([x, h], dim=-1)
        else:
            return h
