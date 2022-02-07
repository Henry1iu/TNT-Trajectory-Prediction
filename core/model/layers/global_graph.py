# source: https://github.com/xk-huang/yet-another-vectornet

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, max_pool
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GATv2Conv, TransformerConv, SuperGATConv


class GlobalGraph(nn.Module):
    """
    Global graph that compute the global information
    """
    def __init__(self, in_channels,
                 global_graph_width,
                 num_global_layers=1,
                 need_scale=False,
                 with_norm=False):
        super(GlobalGraph, self).__init__()
        self.in_channels = in_channels
        self.global_graph_width = global_graph_width

        self.layers = nn.Sequential()

        in_channels = self.in_channels
        for i in range(num_global_layers):
            self.layers.add_module(
                f'glp_{i}', SelfAttentionFCLayer(in_channels,
                                                 self.global_graph_width,
                                                 need_scale)
            )

            # self.layers.add_module(
            #     # f'glp_{i}', SelfAttentionFCLayer(in_channels, self.global_graph_width)
            #     f'glp_{i}', GATv2Conv(in_channels, self.global_graph_width, add_self_loops=False)
            #     # f'glp_{i}', TransformerConv(in_channels=in_channels, out_channels=self.global_graph_width)
            # )
            in_channels = self.global_graph_width

    def forward(self, global_data, **kwargs):
        x, edge_index = global_data.x, global_data.edge_index
        valid_lens, time_step_len = global_data.valid_lens, int(global_data.time_step_len[0])

        # print("x size:", x.size())
        # x = x.view(-1, time_step_len, self.in_channels)
        for name, layer in self.layers.named_modules():
            if isinstance(layer, SelfAttentionLayer):
                x = layer(x, edge_index, valid_lens)

            elif isinstance(layer, GATv2Conv):
                x = layer(x, edge_index)

            elif isinstance(layer, TransformerConv):
                x = layer(x, edge_index)

            elif isinstance(layer, SelfAttentionFCLayer):
                x = layer(x, valid_lens, **kwargs)
                # x = layer(x, valid_lens)

        return x


class SelfAttentionLayer(MessagePassing):
    """
    Self-attention layer. no scale_factor d_k
    """

    def __init__(self,
                 in_channels,
                 global_graph_width,
                 need_scale=False,
                 with_norm=False):
        super(SelfAttentionLayer, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.with_norm = with_norm

        self.global_graph_width = global_graph_width

        self.q_lin = nn.Linear(in_channels, global_graph_width)
        self.k_lin = nn.Linear(in_channels, global_graph_width)
        self.v_lin = nn.Linear(in_channels, global_graph_width)

        self.scale_factor_d = 1 + \
            int(np.sqrt(self.in_channels)) if need_scale else 1

    def forward(self, x, edge_index, valid_len):
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # print("x size", x.size())

        # attention
        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)

        # print("key size", key.size())
        scores = torch.bmm(query, key.transpose(1, 2))
        attention_weights = self.masked_softmax(scores, valid_len)
        x = torch.bmm(attention_weights, value)

        x = x.view(-1, self.global_graph_width)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        return x_j

    @staticmethod
    def masked_softmax(X, valid_len):
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
                valid_len = torch.repeat_interleave(valid_len, repeats=shape[1], dim=0)
            else:
                valid_len = valid_len.reshape(-1)
            # Fill masked elements with a large negative, whose exp is 0
            mask = torch.ones_like(X, dtype=torch.bool)
            for batch_id, cnt in enumerate(valid_len):
                mask[batch_id, :cnt] = False
            X_masked = X.masked_fill(mask, -1e12)
            return nn.functional.softmax(X_masked, dim=-1)


class SelfAttentionFCLayer(nn.Module):
    """
    Self-attention layer. no scale_factor d_k
    """

    def __init__(self, in_channels, global_graph_width, need_scale=False):
        super(SelfAttentionFCLayer, self).__init__()
        self.in_channels = in_channels
        self.q_lin = nn.Linear(in_channels, global_graph_width)
        self.k_lin = nn.Linear(in_channels, global_graph_width)
        self.v_lin = nn.Linear(in_channels, global_graph_width)
        self.scale_factor_d = 1 + \
            int(np.sqrt(self.in_channels)) if need_scale else 1

    def forward(self, x, valid_len, batch_size):
        x = x.view(batch_size, -1, self.in_channels)

        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)
        # print("shape of key: {};".format(key.shape))
        scores = torch.bmm(query, key.transpose(1, 2))
        attention_weights = self.masked_softmax(scores, valid_len)
        x = torch.bmm(attention_weights, value).reshape(-1, self.in_channels)
        return x

    @staticmethod
    def masked_softmax(X, valid_len):
        """
        masked softmax for attention scores
        args:
            X: 3-D tensor, valid_len: 1-D or 2-D tensor
        """
        if valid_len is None:
            return nn.functional.softmax(X, dim=-1)
        else:
            shape = X.shape
            if valid_len.shape[0] != shape[0]:
                valid_len = torch.repeat_interleave(valid_len, repeats=shape[0], dim=0)
            else:
                valid_len = valid_len.reshape(-1)
            # Fill masked elements with a large negative, whose exp is 0
            mask = torch.zeros_like(X, dtype=torch.bool)
            for batch_id, cnt in enumerate(valid_len):
                mask[batch_id, :, cnt:] = True
                mask[batch_id, cnt:] = True
            X_masked = X.masked_fill(mask, -1e32)
            return nn.functional.softmax(X_masked, dim=-1) * (1 - mask.float())


if __name__ == "__main__":
    data = Data(x=torch.tensor([[[1.0], [7.0]]]),
                edge_index=torch.tensor([[0, 1], [1, 0]]),
                valid_lens=torch.tensor([1]))
    print(data.x.size())

    layer = SelfAttentionFCLayer(1, 1)

    for k, v in layer.state_dict().items():
        if k.endswith('weight'):
            v[:] = torch.tensor([[1.0]])
        elif k.endswith('bias'):
            v[:] = torch.tensor([1.0])

    y = layer(data.x, data.valid_lens, 1)
