# The imlementation of subgraph encoding of VectorNet
# Written by: Jianbang LIU @ RPAI, CUHK
# Created: 2021.10.02

import torch
import torch.nn as nn
import torch.nn.functional as F

# source: https://github.com/xk-huang/yet-another-vectornet
import numpy as np
import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, max_pool, avg_pool
from torch_geometric.utils import add_self_loops, remove_self_loops

from core.model.layers.basic_module import MLP


class SubGraph(nn.Module):
    """
    Subgraph that computes all vectors in a polyline, and get a polyline-level feature
    """

    def __init__(self, in_channels, num_subgraph_layres=3, hidden_unit=64):
        super(SubGraph, self).__init__()
        self.num_subgraph_layres = num_subgraph_layres
        self.hidden_unit = hidden_unit
        self.out_channels = hidden_unit

        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layres):
            self.layer_seq.add_module(
                f'glp_{i}', MLP(in_channels, hidden_unit, hidden_unit))
            in_channels = hidden_unit * 2

        self.linear = nn.Linear(hidden_unit * 2, hidden_unit)

    def forward(self, sub_data):
        """
        polyline vector set in torch_geometric.data.Data format
        args:
            sub_data (Data): [x, y, cluster, edge_index, valid_len]
        """
        x = sub_data.x
        sub_data.cluster = sub_data.cluster.long()
        sub_data.edge_index = sub_data.edge_index.long()

        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                x = layer(x)
                sub_data.x = x
                agg_data = max_pool(sub_data.cluster, sub_data)

                x = torch.cat([x, agg_data.x[sub_data.cluster]], dim=-1)

        x = self.linear(x)
        sub_data.x = x
        out_data = max_pool(sub_data.cluster, sub_data)
        x = out_data.x

        assert x.shape[0] % int(sub_data.time_step_len[0]) == 0

        return F.normalize(x, p=2.0, dim=1)      # L2 normalization
# %%


if __name__ == "__main__":
    # data = Data(x=torch.tensor([[1.0], [7.0]]), edge_index=torch.tensor([[0, 1], [1, 0]]))
    # print(data)
    # layer = GraphLayerProp(1, 1, True)
    # for k, v in layer.state_dict().items():
    #     if k.endswith('weight'):
    #         v[:] = torch.tensor([[1.0]])
    #     elif k.endswith('bias'):
    #         v[:] = torch.tensor([1.0])
    # y = layer(data.x, data.edge_index)
    pass
