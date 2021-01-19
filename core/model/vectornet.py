# VectorNet Implementation
# Author: Jianbang LIU @ RPAI Lab, CUHK
# Email: henryliu@link.cuhk.edu.hk
# Cite: https://github.com/xk-huang/yet-another-vectornet
# Modification: Add auxiliary layer and loss

import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, DataListLoader, Batch, Data

from core.model.layers.global_graph import GlobalGraph, SelfAttentionFCLayer
from core.model.layers.subgraph import SubGraph
from core.dataloader.dataset import GraphDataset, GraphData


class VectorNet(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """

    def __init__(self,
                 in_channels=8,
                 pred_len=30,
                 num_subgraph_layres=3,
                 num_global_graph_layer=1,
                 subgraph_width=64,
                 global_graph_width=64,
                 traj_pred_mlp_width=64,
                 with_aux: bool = False,
                 device=torch.device("cpu")):
        super(VectorNet, self).__init__()
        # some params
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layres)
        self.out_channels = 2
        self.pred_len = pred_len
        self.subgraph_width = subgraph_width
        self.max_n_guesses = 1

        self.device = device

        # subgraph feature extractor
        self.subgraph = SubGraph(in_channels, num_subgraph_layres, subgraph_width)

        # global graph
        self.global_graph = GlobalGraph(self.polyline_vec_shape,
                                        global_graph_width,
                                        num_global_layers=num_global_graph_layer,
                                        device=self.device)

        # pred mlp
        self.traj_pred_mlp = nn.Sequential(
            nn.Linear(global_graph_width, traj_pred_mlp_width),
            nn.LayerNorm(traj_pred_mlp_width),
            nn.ReLU(),
            nn.Linear(traj_pred_mlp_width, self.pred_len * self.out_channels)
        )

        # auxiliary recoverey mlp
        self.with_aux = with_aux
        if self.with_aux:
            self.aux_mlp = nn.Sequential(
                nn.Linear(global_graph_width, traj_pred_mlp_width),
                nn.LayerNorm(traj_pred_mlp_width),
                nn.ReLU(),
                nn.Linear(traj_pred_mlp_width, subgraph_width)
            )

    def forward(self, data):
        """
        args:
            data (Data): [x, y, cluster, edge_index, valid_len]
        """
        time_step_len = int(data.time_step_len[0])
        valid_lens = data.valid_len

        # print("valid_lens type:", type(valid_lens).__name__)
        # print("data batch size:", data.num_batch)

        sub_graph_out = self.subgraph(data)
        x = sub_graph_out.x.view(-1, time_step_len, self.polyline_vec_shape)

        # TODO: compute the adjacency matrix???
        # reconstruct the batch global interaction graph data
        if isinstance(data, Batch):
            # mini-batch case
            global_g_data = Batch()
            batch_list = []
            for idx in range(data.num_graphs):
                node_list = torch.Tensor([i for i in range(valid_lens[idx])]).long()
                edge_index = torch.combinations(node_list, 2).transpose(1, 0)

                # print(x[idx, :, :].size())
                batch_list.append(Data(x=F.normalize(x[idx, :, :], dim=1).squeeze(0),
                                       edge_index=edge_index,
                                       valid_lens=valid_lens[idx],
                                       time_step_len=time_step_len))
            global_g_data = global_g_data.from_data_list(batch_list)
        elif isinstance(data, Data):
            # single batch case
            node_list = torch.Tensor([i for i in range(valid_lens[0])]).long()
            edge_index = torch.combinations(node_list, 2).transpose(1, 0)
            c = Data(x=F.normalize(x[0, :, :], dim=3).squeeze(0),
                                 edge_index=edge_index,
                                 valid_lens=valid_lens,
                                 time_step_len=time_step_len)
        else:
            raise NotImplementedError

        global_g_data.to(self.device)
        if self.training:
            # mask out the features for a random subset of polyline nodes
            # for one batch, we mask the same polyline features

            global_graph_out, mask_polyline_indices = self.global_graph(global_g_data)
            global_graph_out = global_graph_out.view(-1, time_step_len, self.polyline_vec_shape)

            pred = self.traj_pred_mlp(global_graph_out[:, [0]].squeeze(1))
            if self.with_aux:
                aux_in = torch.empty((global_graph_out.size()[0],
                                      self.polyline_vec_shape)
                                     ).to(self.device)
                aux_gt = torch.empty((global_graph_out.size()[0],
                                      self.polyline_vec_shape)).to(self.device)
                for i, idx in enumerate(mask_polyline_indices):
                    aux_in[i] = global_graph_out[i, idx].squeeze(0)
                    aux_gt[i] = x[i, idx].squeeze(0)
                aux_out = self.aux_mlp(aux_in)

                return pred, aux_out, aux_gt
            else:
                return pred, None, None

        else:
            # print("x size:", x.size())

            global_graph_out, _ = self.global_graph(global_g_data)
            global_graph_out = global_graph_out.view(-1, time_step_len, self.polyline_vec_shape)

            pred = self.traj_pred_mlp(global_graph_out[:, [0]].squeeze(1))

            return pred


class OriginalVectorNet(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """

    def __init__(self,
                 in_channels=8,
                 pred_len=30,
                 num_subgraph_layres=3,
                 num_global_graph_layer=1,
                 subgraph_width=64,
                 global_graph_width=64,
                 traj_pred_mlp_width=64,
                 with_aux: bool = False,
                 device=torch.device("cpu")):
        super(OriginalVectorNet, self).__init__()
        # some params
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layres)
        self.out_channels = 2
        self.pred_len = pred_len
        self.subgraph_width = subgraph_width
        self.max_n_guesses = 1

        self.device = device

        # subgraph feature extractor
        self.subgraph = SubGraph(in_channels, num_subgraph_layres, subgraph_width)

        self.global_graph = SelfAttentionFCLayer(
            subgraph_width, global_graph_width, need_scale=False
        )

        # pred mlp
        self.traj_pred_mlp = nn.Sequential(
            nn.Linear(global_graph_width, traj_pred_mlp_width),
            nn.LayerNorm(traj_pred_mlp_width),
            nn.ReLU(),
            nn.Linear(traj_pred_mlp_width, self.pred_len * self.out_channels)
        )

        # auxiliary recoverey mlp
        self.with_aux = with_aux
        if self.with_aux:
            self.aux_mlp = nn.Sequential(
                nn.Linear(global_graph_width, traj_pred_mlp_width),
                nn.LayerNorm(traj_pred_mlp_width),
                nn.ReLU(),
                nn.Linear(traj_pred_mlp_width, subgraph_width)
            )

    def forward(self, data):
        """
        args:
            data (Data): [x, y, cluster, edge_index, valid_len]
        """
        time_step_len = int(data.time_step_len[0])
        valid_lens = data.valid_len

        # print("valid_lens type:", type(valid_lens).__name__)
        # print("data batch size:", data.num_batch)

        sub_graph_out = self.subgraph(data)
        x = sub_graph_out.x.view(-1, time_step_len, self.subgraph_width)

        if self.training:
            # mask out the features for a random subset of polyline nodes
            # for one batch, we mask the same polyline features
            if self.with_aux:
                mask_polyline_indices = [random.randint(0, time_step_len-1) for _ in range(x.size()[0])]
                for i, idx in enumerate(mask_polyline_indices):
                    x[i, idx, :] = 0.0

            global_graph_out = self.global_graph(x, valid_lens)
            pred = self.traj_pred_mlp(global_graph_out[:, [0]].squeeze(1))

            if self.with_aux:
                aux_in = torch.empty((global_graph_out.size()[0],
                                      self.polyline_vec_shape)
                                     ).to(self.device)
                aux_gt = torch.empty((global_graph_out.size()[0],
                                      self.polyline_vec_shape))
                for i, idx in enumerate(mask_polyline_indices):
                    aux_in[i] = global_graph_out[i, idx].squeeze(0)
                    aux_gt[i] = x[i, idx].squeeze(0)
                aux_out = self.aux_mlp(aux_in)

                return pred, aux_out, aux_gt
            else:
                return pred, None, None

        else:
            # print("x size:", x.size())

            global_graph_out = self.global_graph(x, valid_lens)

            pred = self.traj_pred_mlp(global_graph_out[:, [0]].squeeze(1))

            return pred


# %%
if __name__ == "__main__":
    epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    decay_lr_factor = 0.9
    decay_lr_every = 10
    lr = 0.005
    in_channels, pred_len = 8, 30
    show_every = 10
    os.chdir('..')
    # get model
    # model = VectorNet(in_channels, pred_len, with_aux=True).to(device)
    model = OriginalVectorNet(in_channels, pred_len, with_aux=True).to(device)

    DATA_DIR = "/Users/jb/projects/trajectory_prediction_algorithms/yet-another-vectornet"
    TRAIN_DIR = os.path.join(DATA_DIR, 'data/interm_data', 'train_intermediate')

    dataset = GraphDataset(TRAIN_DIR)
    data_iter = DataLoader(dataset[:10], batch_size=batch_size)

    model.train()
    for data in data_iter:
        out, aux_out, mask_feat_gt = model(data)
        print("Training Pass")

    model.eval()
    for data in data_iter:
        out = model(data)
        print("Evaluation Pass")
