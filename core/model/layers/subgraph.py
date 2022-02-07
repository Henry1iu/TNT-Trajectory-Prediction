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
        self.out_channels = hidden_unit * 2

        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layres):
            self.layer_seq.add_module(
                f'glp_{i}', GraphLayerProp(in_channels, hidden_unit))
            in_channels = hidden_unit * 2

        # self.linear = nn.Linear(hidden_unit * 2, hidden_unit)

    def forward(self, sub_data):
        """
        polyline vector set in torch_geometric.data.Data format
        args:
            sub_data (Data): [x, y, cluster, edge_index, valid_len]
        """
        x, edge_index, batch = sub_data.x, sub_data.edge_index, sub_data.batch

        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, GraphLayerProp):
                x = layer(x, edge_index)
        sub_data.x = x
        out_data = max_pool(sub_data.cluster, sub_data)

        # try:
        # ###################################### DEBUG ###################################### #
        # print("\nsize of sub_data.x: {};".format(sub_data.x.shape[0]))
        # print("\nsize of out_data.x: {};".format(out_data.x.shape[0]))
        # print("time_step_len: {};".format(sub_data.time_step_len[0]))
        # print("cluster: {};".format(sub_data.cluster))
        # print("size of cluster: {};".format(sub_data.cluster.shape))
        # print("num of cluster: {};".format(torch.unique(sub_data.cluster).shape))
        # ###################################### DEBUG ###################################### #

        assert out_data.x.shape[0] % int(sub_data.time_step_len[0]) == 0
        # out_data.x = torch.div(out_data.x.T, torch.norm(out_data.x, dim=-1)).T
        out_data.x = out_data.x / (out_data.x.norm(dim=0) + 1e-12)      # L2 normalization
        return out_data

        # node_feature, _ = torch.max(x, dim=0)
        # # l2 noramlize node_feature before feed it to global graph
        # node_feature = node_feature / node_feature.norm(dim=0)
        # return node_feature

# %%


class GraphLayerProp(MessagePassing):
    """
    Message Passing mechanism for infomation aggregation
    """

    def __init__(self, in_channels, hidden_unit=64, verbose=False):
        super(GraphLayerProp, self).__init__(
            aggr='max')  # MaxPooling aggragation
        self.verbose = verbose
        self.residual = True if in_channels == hidden_unit else False

        # self.mlp = nn.Sequential(
        #     nn.Linear(in_channels, hidden_unit),
        #     nn.LayerNorm(hidden_unit),
        #     nn.ReLU(),
        #     nn.Linear(hidden_unit, hidden_unit)
        # )
        self.mlp = MLP(in_channels, hidden_unit, hidden_unit)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if self.verbose:
            print(f'x before mlp: {x}')

        x = self.mlp(x)

        if self.verbose:
            print(f"x after mlp: {x}")
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x):
        if self.verbose:
            print(f"x after mlp: {x}")
            print(f"aggr_out: {aggr_out}")
        return torch.cat([x, aggr_out], dim=1)


if __name__ == "__main__":
    data = Data(x=torch.tensor([[1.0], [7.0]]), edge_index=torch.tensor([[0, 1], [1, 0]]))
    print(data)
    layer = GraphLayerProp(1, 1, True)
    for k, v in layer.state_dict().items():
        if k.endswith('weight'):
            v[:] = torch.tensor([[1.0]])
        elif k.endswith('bias'):
            v[:] = torch.tensor([1.0])
    y = layer(data.x, data.edge_index)
