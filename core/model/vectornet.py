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
from torch_geometric.data import DataLoader, Data

from core.model.layers.global_graph import GlobalGraph
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
        # edge_index = data.edge_index

        sub_graph_out = self.subgraph(data)
        x = sub_graph_out.x.view(-1, time_step_len, self.polyline_vec_shape)

        # TODO: compute the adjacency matrix???
        global_graph_data = Data(x=F.normalize(x),
                                 valid_lens=valid_lens,
                                 time_step_len=time_step_len)
        if self.training:
            # mask out the features for a random subset of polyline nodes
            # for one batch, we mask the same polyline features
            mask_id = random.randint(0, time_step_len-1)
            mask_polyline_feat = x[:, mask_id, :]
            global_graph_data.x[:, mask_id, :] = 0.0

            global_graph_out = self.global_graph(global_graph_data)
            x = global_graph_out.view(-1, time_step_len, self.polyline_vec_shape)

            pred = self.traj_pred_mlp(x[:, [0]].squeeze(1))
            if self.with_aux:
                aux_out = self.aux_mlp(x[:, [0]].squeeze(1))

                return pred, aux_out, mask_polyline_feat
            else:
                return pred, None, None

        else:
            global_graph_out = self.global_graph(global_graph_data)
            x = global_graph_out.view(-1, time_step_len, self.polyline_vec_shape)

            pred = self.traj_pred_mlp(x[:, [0]].squeeze(1))

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
    model = VectorNet(in_channels, pred_len).to(device)
    model.train()

    DATA_DIR = "/Users/jb/projects/trajectory_prediction_algorithms/yet-another-vectornet"
    TRAIN_DIR = os.path.join(DATA_DIR, 'data/interm_data', 'train_intermediate')

    dataset = GraphDataset(TRAIN_DIR)
    data_iter = DataLoader(dataset, batch_size=batch_size)
    for data in data_iter:
        out, aux_out, mask_feat_gt = model(data)
        print("Training Pass")

    model.eval()
    for data in data_iter:
        out = model(data)
        print("Evaluation Pass")
