# source: https://github.com/xk-huang/yet-another-vectornet

import numpy as np
from itertools import permutations

import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing, max_pool
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data


class GlobalGraph(nn.Module):
    """
    Global graph that compute the global information
    """
    def __init__(self, in_channels,
                 global_graph_width,
                 num_global_layers=1,
                 need_scale=False,
                 with_norm=False,
                 device=torch.device("cpu")):
        super(GlobalGraph, self).__init__()
        self.in_channels = in_channels
        self.global_graph_width = global_graph_width

        self.device = device

        self.layers = nn.Sequential()
        for i in range(num_global_layers):
            self.layers.add_module(
                f'glp_{i}', SelfAttentionLayer(self.in_channels,
                                               self.global_graph_width,
                                               need_scale,
                                               with_norm,
                                               self.device)
            )

    def forward(self, global_data):
        x, valid_len, time_step_len = global_data.x, global_data.valid_lens, global_data.time_step_len

        for name, layer in self.layers.named_modules():
            if isinstance(layer, SelfAttentionLayer):
                x = layer(x, valid_len, time_step_len)

        return x


class SelfAttentionLayer(MessagePassing):
    """
    Self-attention layer. no scale_factor d_k
    """

    def __init__(self,
                 in_channels,
                 global_graph_width,
                 need_scale=False,
                 with_norm=False,
                 device=torch.device("cpu")):
        super(SelfAttentionLayer, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.with_norm = with_norm
        self.device = device

        self.q_lin = nn.Linear(global_graph_width, global_graph_width)
        self.k_lin = nn.Linear(global_graph_width, global_graph_width)
        self.v_lin = nn.Linear(global_graph_width, global_graph_width)

        self.scale_factor_d = 1 + \
            int(np.sqrt(self.in_channels)) if need_scale else 1

    def forward(self, x, valid_len, time_step_len):
        # cosntruct the fully connected graph(s)
        edge_index = torch.Tensor([]).to(self.device)
        for graph_id in range(len(valid_len)):
            node_list = torch.Tensor([i for i in range(valid_len[graph_id])]).to(self.device) + graph_id*time_step_len
            edge_index = torch.cat((edge_index, torch.combinations(node_list, 2)), 0)
        edge_index = edge_index.transpose(1, 0).long()
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = x.view(-1, time_step_len, self.in_channels)
        # attention
        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)
        scores = torch.bmm(query, key.transpose(1, 2))
        attention_weights = self.masked_softmax(scores, valid_len)
        x = torch.bmm(attention_weights, value)

        x = x.view(-1, self.in_channels)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        return x_j

    def masked_softmax(self, X, valid_len):
        """
        masked softmax for attention scores
        args:
            X: 3-D tensor, valid_len: 1-D or 2-D tensor
        """
        if valid_len is None:
            return nn.functional.softmax(X, dim=-1)
        else:
            shape = X.shape
            if valid_len.dim() == 1:
                valid_len = torch.repeat_interleave(
                    valid_len, repeats=shape[1], dim=0)
            else:
                valid_len = valid_len.reshape(-1)
            # Fill masked elements with a large negative, whose exp is 0
            X = X.reshape(-1, shape[-1])
            for count, row in enumerate(X):
                row[int(valid_len[count]):] = -1e6
            return nn.functional.softmax(X.reshape(shape), dim=-1)


if __name__ == "__main__":
    data = Data(x=torch.tensor([[1.0], [7.0]]),
                edge_index=torch.tensor([[0, 1], [1, 0]]),
                valid_lens=torch.tensor([1]))
    print(data)

    layer = SelfAttentionLayer(1, 1)

    for k, v in layer.state_dict().items():
        if k.endswith('weight'):
            v[:] = torch.tensor([[1.0]])
        elif k.endswith('bias'):
            v[:] = torch.tensor([1.0])

    y = layer(data)
