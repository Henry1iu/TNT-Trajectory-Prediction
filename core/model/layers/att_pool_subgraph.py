# vectornet subgraph module implementation with attention pooling
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, max_pool, avg_pool
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.data import Data, DataLoader

from core.dataloader.argoverse_loader_v2 import ArgoverseInMem, GraphData


class SubGraph(nn.Module):
    """
    Subgraph that computes all vectors in a polyline, and get a polyline-level feature
    """

    def __init__(self, in_channels, num_subgraph_layers=3, hidden_unit=64):
        super(SubGraph, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.out_channels = hidden_unit * 2

        self.layer0 = GraphLayerProp(in_channels, hidden_unit)
        self.layers = nn.Sequential()

        for i in range(1, num_subgraph_layers):
            self.layers.add_module(f'glp_{i}', GraphLayerProp(2 * hidden_unit, hidden_unit))

        self.linear = nn.Sequential(
            nn.Linear(4 * hidden_unit, 2 * hidden_unit),
            nn.ReLU(),
        )

    def forward(self, sub_data):
        """
        polyline vector set in torch_geometric.data.Data format
        args:
            sub_data (Data): [x, y, cluster, edge_index, valid_len]
        """
        x, edge_index, batch = sub_data.x, sub_data.edge_index, sub_data.batch
        x = self.layer0(x, edge_index)
        sub_data.x = x
        mx = max_pool(sub_data.cluster, sub_data)
        mn = avg_pool(sub_data.cluster, sub_data)
        xs = torch.cat([mx.x, mn.x], dim=-1)

        for _, layer in self.layers.named_modules():
            if isinstance(layer, GraphLayerProp):
                x = layer(x, edge_index)
                sub_data.x = x
                mx = max_pool(sub_data.cluster, sub_data)
                mn = avg_pool(sub_data.cluster, sub_data)
                xs += torch.cat([mx.x, mn.x], dim=-1)

        sub_data.x = self.linear(xs)

        # try:
        # ###################################### DEBUG ###################################### #
        # print("\nsize of sub_data.x: {};".format(sub_data.x.shape[0]))
        # print("\nsize of out_data.x: {};".format(out_data.x.shape[0]))
        # print("time_step_len: {};".format(sub_data.time_step_len[0]))
        # print("cluster: {};".format(sub_data.cluster))
        # print("size of cluster: {};".format(sub_data.cluster.shape))
        # print("num of cluster: {};".format(torch.unique(sub_data.cluster).shape))
        # ###################################### DEBUG ###################################### #

        assert sub_data.x.shape[0] % int(sub_data.time_step_len[0]) == 0
        sub_data.x = sub_data.x / (sub_data.x.norm(dim=0) + 1e-8)
        return sub_data


class GraphLayerProp(MessagePassing):
    """
    Message Passing mechanism for infomation aggregation
    """

    def __init__(self, in_channels, hidden_unit=64):
        super(GraphLayerProp, self).__init__(aggr='max')  # MaxPooling aggragation
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_unit, hidden_unit),
            nn.LayerNorm(hidden_unit),
        )

        self.res_layer = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.mlp(x) + self.res_layer(x)
        x = self.relu(x)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x):
        return torch.cat([x, aggr_out], dim=1)


if __name__ == "__main__":
    INTERMEDIATE_DATA_DIR = "~/projects/Code/trajectory-prediction/TNT-Trajectory-Predition/dataset/interm_tnt_n_s_0804_small"

    for folder in ["train", "val"]:
        dataset_input_path = os.path.join(INTERMEDIATE_DATA_DIR, f"{folder}_intermediate")
        dataset = ArgoverseInMem(dataset_input_path).shuffle()

        layer = SubGraph(dataset.num_features, 1).cpu()
        batch_iter = DataLoader(dataset, batch_size=16, num_workers=16, shuffle=True, pin_memory=True)

        for i, data in enumerate(tqdm(batch_iter, total=len(batch_iter), bar_format="{l_bar}{r_bar}")):
            y = layer(data.cpu())
