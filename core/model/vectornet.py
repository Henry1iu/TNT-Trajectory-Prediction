# VectorNet Implementation
# Author: Jianbang LIU @ RPAI Lab, CUHK
# Email: henryliu@link.cuhk.edu.hk
# Cite: https://github.com/xk-huang/yet-another-vectornet
# Modification: Add auxiliary layer and loss

import os

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
# from core.dataloader import GraphDataset

from core.model.layers.selfatten import SelfAttentionLayer
from core.model.layers.subgraph import SubGraph
from core.dataloader.dataset import GraphDataset, GraphData


class HGNN(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """

    def __init__(self, in_channels, out_channels, num_subgraph_layres=3, num_global_graph_layer=1, subgraph_width=64,
                 global_graph_width=64, traj_pred_mlp_width=64):
        super(HGNN, self).__init__()
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layres)

        # subgraph feature extractor
        self.subgraph = SubGraph(in_channels, num_subgraph_layres, subgraph_width)

        self.self_atten_layer = SelfAttentionLayer(self.polyline_vec_shape,
                                                   global_graph_width,
                                                   need_scale=False)
        # # Global GNN network
        # self_atten_layer_list = []
        # for i in range(num_global_graph_layer):
        #     self_atten_layer_list.append(SelfAttentionLayer(self.polyline_vec_shape,
        #                                                     global_graph_width,
        #                                                     need_scale=False))
        # self.self_atten_layer = nn.Sequential(*self_atten_layer_list)

        # pred mlp
        self.traj_pred_mlp = nn.Sequential(
            nn.Linear(global_graph_width, traj_pred_mlp_width),
            nn.LayerNorm(traj_pred_mlp_width),
            nn.ReLU(),
            nn.Linear(traj_pred_mlp_width, out_channels)
        )

        # auxiliary recoverey mlp
        self.aux_mlp = nn.Sequential(
            nn.Linear(global_graph_width, traj_pred_mlp_width),
            nn.LayerNorm(traj_pred_mlp_width),
            nn.ReLU(),
            nn.Linear(traj_pred_mlp_width, out_channels)
        )

    def forward(self, data):
        """
        args:
            data (Data): [x, y, cluster, edge_index, valid_len]
        """
        time_step_len = int(data.time_step_len[0])
        valid_lens = data.valid_len
        sub_graph_out = self.subgraph(data)
        x = sub_graph_out.x.view(-1, time_step_len, self.polyline_vec_shape)

        if self.training:
            # mask out the features for a random subset of polyline nodes

            # global GNN
            out = self.self_atten_layer(x, valid_lens)

            pred = self.traj_pred_mlp(out[:, [0]].squeeze(1))

            aux_out = self.aux_mlp(out[:, [0]].squeeze(1))
        else:
            out = self.self_atten_layer(x, valid_lens)
            pred = self.traj_pred_mlp(out[:, [0]].squeeze(1))
            return pred


# %%
if __name__ == "__main__":
    epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    decay_lr_factor = 0.9
    decay_lr_every = 10
    lr = 0.005
    in_channels, out_channels = 8, 60
    show_every = 10
    os.chdir('..')
    # get model
    model = HGNN(in_channels, out_channels).to(device)
    model.eval()

    DATA_DIR = "/Users/jb/projects/trajectory_prediction_algorithms/yet-another-vectornet"
    TRAIN_DIR = os.path.join(DATA_DIR, 'data/interm_data', 'train_intermediate')

    dataset = GraphDataset(TRAIN_DIR)
    data_iter = DataLoader(dataset, batch_size=batch_size)
    for data in data_iter:
        out = model(data)