import os
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Batch, Data

from core.model.layers.global_graph import GlobalGraph
from core.model.layers.att_pool_subgraph import SubGraph
from core.dataloader.argoverse_loader_v2 import ArgoverseInMem, GraphData


class VectorNetBackbone(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """

    def __init__(self,
                 in_channels=8,
                 num_subgraph_layres=3,
                 num_global_graph_layer=1,
                 subgraph_width=64,
                 global_graph_width=64,
                 aux_mlp_width=64,
                 with_aux: bool = False,
                 device=torch.device("cpu")):
        super(VectorNetBackbone, self).__init__()
        # some params
        self.num_subgraph_layres = num_subgraph_layres
        self.global_graph_width = global_graph_width

        self.device = device

        # subgraph feature extractor
        self.subgraph = SubGraph(in_channels, num_subgraph_layres, subgraph_width)

        # global graph
        self.global_graph = GlobalGraph(self.subgraph.out_channels + 2,
                                        self.global_graph_width,
                                        num_global_layers=num_global_graph_layer)

        # auxiliary recoverey mlp
        self.with_aux = with_aux
        if self.with_aux:
            self.aux_mlp = nn.Sequential(
                nn.Linear(self.global_graph_width, aux_mlp_width),
                nn.LayerNorm(aux_mlp_width),
                nn.ReLU(),
                nn.Linear(aux_mlp_width, self.subgraph.out_channels)
            )

    def forward(self, data):
        """
        args:
            data (Data): [x, y, cluster, edge_index, valid_len]
        """
        time_step_len = data.time_step_len[0].int()
        valid_lens = data.valid_len

        id_embedding = data.identifier

        sub_graph_out = self.subgraph(data)

        if self.training and self.with_aux:
            batch_size = data.num_graphs
            randoms = torch.rand((batch_size,), device=self.device) * (valid_lens - 1) + \
                      time_step_len * torch.arange(batch_size, device=self.device)
            # mask_polyline_indices = [torch.randint(1, valid_lens[i] - 1) + i * time_step_len for i in range(batch_size)]
            mask_polyline_indices = randoms.long()
            aux_gt = sub_graph_out.x[mask_polyline_indices]
            sub_graph_out.x[mask_polyline_indices] = 0.0

        # reconstruct the batch global interaction graph data
        sub_graph_out.x = torch.cat([sub_graph_out.x, id_embedding], dim=1)
        sub_graph_out.valid_lens = data.valid_len
        sub_graph_out.time_step_len = data.time_step_len

        sub_graph_out.edge_index = self._get_global_edge_index_fc(valid_lens, time_step_len)

        if self.training:
            # mask out the features for a random subset of polyline nodes
            # for one batch, we mask the same polyline features

            # global_graph_out = self.global_graph(sub_graph_out, batch_size=data.num_graphs)
            global_graph_out = self.global_graph(sub_graph_out)
            global_graph_out = global_graph_out.view(-1, time_step_len, self.global_graph_width)

            if self.with_aux:
                aux_in = global_graph_out.view(-1, self.global_graph_width)[mask_polyline_indices]
                aux_out = self.aux_mlp(aux_in)

                return global_graph_out, aux_out, aux_gt
            else:
                return global_graph_out, None, None

        else:
            # global_graph_out = self.global_graph(sub_graph_out, batch_size=data.num_graphs)
            global_graph_out = self.global_graph(sub_graph_out)
            global_graph_out = global_graph_out.view(-1, time_step_len, self.global_graph_width)

            return global_graph_out, None, None

    def _get_global_edge_index_fc(self, valid_lens, time_step_len):
        """generating the fully-connected undirected graph"""
        if len(valid_lens) > 1:                 # mini-batch case
            edge_index = torch.empty((2, 0), device=self.device, dtype=torch.long)
            for idx in range(len(valid_lens)):
                node_list = torch.arange(idx * time_step_len, idx * time_step_len + valid_lens[idx], device=self.device).long()
                xx, yy = torch.meshgrid(node_list, node_list)
                xy = torch.vstack([xx.reshape(-1), yy.reshape(-1)])
                edge_index = torch.hstack([edge_index, xy])

        elif len(valid_lens) == 1:                # single batch case
            node_list = torch.arange(0, valid_lens[0], device=self.device).long()
            xx, yy = torch.meshgrid(node_list, node_list)
            edge_index = torch.vstack([xx.reshape(-1), yy.reshape(-1)])
        else:
            raise NotImplementedError

        return edge_index

    def _get_global_edge_index_directed(self, valid_lens, traj_lens, time_step_len):
        """generating the a directed graph with each traj connecting with other trajs and lane polyline. Lane polylines
           are isolated with other lane polylines, only passing information out and will not update.
        """
        if len(valid_lens) > 1:                 # mini-batch case
            edge_index = torch.empty((2, 0), device=self.device, dtype=torch.long)
            for idx in range(len(valid_lens)):
                traj_node_list = torch.arange(idx * time_step_len,
                                              idx * time_step_len + traj_lens[idx], device=self.device).long()
                full_node_list = torch.arange(idx * time_step_len,
                                              idx * time_step_len + valid_lens[idx], device=self.device).long()
                xx, yy = torch.meshgrid(full_node_list, traj_node_list)
                edge_index = torch.hstack([edge_index, torch.vstack([xx.reshape(-1), yy.reshape(-1)])])

        elif len(valid_lens) == 1:                # single batch case
            traj_node_list = torch.arange(0, traj_lens[0], device=self.device).long()
            full_node_list = torch.arange(0, valid_lens[0], device=self.device).long()
            xx, yy = torch.meshgrid(full_node_list, traj_node_list)
            edge_index = torch.vstack([xx.reshape(-1), yy.reshape(-1)])
        else:
            raise NotImplementedError

        return edge_index


if __name__ == "__main__":
    device = torch.device('cuda:1')
    batch_size = 2
    decay_lr_factor = 0.9
    decay_lr_every = 10
    lr = 0.005
    pred_len = 30

    INTERMEDIATE_DATA_DIR = "~/projects/Code/trajectory-prediction/TNT-Trajectory-Predition/dataset/interm_tnt_n_s_0804_small"
    dataset_input_path = os.path.join(INTERMEDIATE_DATA_DIR, "train_intermediate")
    dataset = ArgoverseInMem(dataset_input_path)
    data_iter = DataLoader(dataset, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True)

    model = VectorNetBackbone(dataset.num_features, with_aux=True, device=device).to(device)

    model.train()
    for i, data in enumerate(tqdm(data_iter, total=len(data_iter), bar_format="{l_bar}{r_bar}")):
        out, aux_out, mask_feat_gt = model(data.to(device))
        print("Training Pass")

    model.eval()
    for i, data in enumerate(tqdm(data_iter, total=len(data_iter), bar_format="{l_bar}{r_bar}")):
        out, _, _ = model(data.to(device))
        print("Evaluation Pass")
