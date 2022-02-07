# VectorNet Implementation
# Author: Jianbang LIU @ RPAI Lab, CUHK
# Email: henryliu@link.cuhk.edu.hk
# Cite: https://github.com/xk-huang/yet-another-vectornet
# Modification: Add auxiliary layer and loss

import os
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, DataListLoader, Batch, Data

# from core.model.layers.global_graph import GlobalGraph, SelfAttentionFCLayer
from core.model.layers.global_graph import GlobalGraph, SelfAttentionFCLayer
from core.model.layers.subgraph import SubGraph
from core.dataloader.dataset import GraphDataset, GraphData
# from core.model.backbone.vectornet import VectorNetBackbone
from core.model.layers.basic_module import MLP
from core.model.backbone.vectornet_v2 import VectorNetBackbone
from core.loss import VectorLoss
from core.dataloader.argoverse_loader import Argoverse

from core.dataloader.argoverse_loader_v2 import GraphData, ArgoverseInMem


class VectorNet(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """

    def __init__(self,
                 in_channels=8,
                 horizon=30,
                 num_subgraph_layers=3,
                 num_global_graph_layer=1,
                 subgraph_width=64,
                 global_graph_width=64,
                 traj_pred_mlp_width=64,
                 with_aux: bool = False,
                 device=torch.device("cpu")):
        super(VectorNet, self).__init__()
        # some params
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layers)
        self.out_channels = 2
        self.horizon = horizon
        self.subgraph_width = subgraph_width
        self.global_graph_width = global_graph_width
        self.k = 1

        self.device = device
        self.criterion = VectorLoss(with_aux)

        # subgraph feature extractor
        self.backbone = VectorNetBackbone(
            in_channels=in_channels,
            num_subgraph_layres=num_subgraph_layers,
            subgraph_width=subgraph_width,
            num_global_graph_layer=num_global_graph_layer,
            global_graph_width=global_graph_width,
            with_aux=with_aux,
            device=device
        )

        # pred mlp
        self.traj_pred_mlp = nn.Sequential(
            MLP(global_graph_width, traj_pred_mlp_width, traj_pred_mlp_width),
            nn.Linear(traj_pred_mlp_width, self.horizon * self.out_channels)
        )

    def forward(self, data):
        """
        args:
            data (Data): [x, y, cluster, edge_index, valid_len]
        """
        global_feat, _, _ = self.backbone(data)              # [batch_size, time_step_len, global_graph_width]
        target_feat = global_feat[:, 0]

        pred = self.traj_pred_mlp(target_feat)

        return pred

    def loss(self, data):
        global_feat, aux_out, aux_gt = self.backbone(data)
        target_feat = global_feat[:, 0]

        pred = self.traj_pred_mlp(target_feat)

        y = data.y.view(-1, self.out_channels * self.horizon)

        return self.criterion(pred, y, aux_out, aux_gt)

    def inference(self, data):
        return self.forward(data)

# class VectorNet(nn.Module):
#     """
#     hierarchical GNN with trajectory prediction MLP
#     """
#
#     def __init__(self,
#                  in_channels=8,
#                  horizon=30,
#                  num_subgraph_layres=3,
#                  num_global_graph_layer=1,
#                  subgraph_width=64,
#                  global_graph_width=64,
#                  traj_pred_mlp_width=64,
#                  with_aux: bool = False,
#                  device=torch.device("cpu")):
#         super(VectorNet, self).__init__()
#         # some params
#         self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layres)
#         self.out_channels = 2
#         self.horizon = horizon
#         self.subgraph_width = subgraph_width
#         self.global_graph_width = global_graph_width
#         self.k = 1
#
#         self.device = device
#
#         # subgraph feature extractor
#         self.subgraph = SubGraph(in_channels, num_subgraph_layres, subgraph_width)
#
#         # global graph
#         self.global_graph = GlobalGraph(self.polyline_vec_shape,
#                                         global_graph_width,
#                                         num_global_layers=num_global_graph_layer)
#
#         # pred mlp
#         self.traj_pred_mlp = nn.Sequential(
#             nn.Linear(global_graph_width, traj_pred_mlp_width),
#             nn.LayerNorm(traj_pred_mlp_width),
#             nn.ReLU(),
#             nn.Linear(traj_pred_mlp_width, self.horizon * self.out_channels)
#         )
#
#         # auxiliary recoverey mlp
#         self.with_aux = with_aux
#         if self.with_aux:
#             self.aux_mlp = nn.Sequential(
#                 nn.Linear(global_graph_width, traj_pred_mlp_width),
#                 nn.LayerNorm(traj_pred_mlp_width),
#                 nn.ReLU(),
#                 nn.Linear(traj_pred_mlp_width, subgraph_width)
#             )
#
#     def forward(self, data):
#         """
#         args:
#             data (Data): [x, y, cluster, edge_index, valid_len]
#         """
#         time_step_len = int(data.time_step_len[0])
#         valid_lens = data.valid_len
#
#         # print("valid_lens type:", type(valid_lens).__name__)
#         # print("data batch size:", data.num_batch)
#
#         sub_graph_out = self.subgraph(data)
#         x = sub_graph_out.x.view(-1, time_step_len, self.subgraph_width)
#
#         if self.training and self.with_aux:
#             batch_size = x.size()[0]
#             mask_polyline_indices = [random.randint(0, time_step_len - 1) + i*time_step_len for i in range(batch_size)]
#             x = x.view(-1, self.subgraph_width)
#             aux_gt = x[mask_polyline_indices]
#             x[mask_polyline_indices] = 0.0
#             x = x.view(-1, time_step_len, self.subgraph_width)
#
#         # TODO: compute the adjacency matrix???
#         # reconstruct the batch global interaction graph data
#         if isinstance(data, Batch):
#             # mini-batch case
#             global_g_data = Batch()
#             batch_list = []
#             for idx in range(data.num_graphs):
#                 node_list = torch.tensor([i for i in range(valid_lens[idx])]).long()
#                 edge_index = torch.combinations(node_list, 2).transpose(1, 0)
#
#                 batch_list.append(Data(x=F.normalize(x[idx, :, :], dim=1).squeeze(0),
#                                        edge_index=edge_index,
#                                        valid_lens=valid_lens[idx],
#                                        time_step_len=time_step_len))
#             global_g_data = global_g_data.from_data_list(batch_list)
#         elif isinstance(data, Data):
#             # single batch case
#             node_list = torch.tensor([i for i in range(valid_lens[0])]).long()
#             edge_index = torch.combinations(node_list, 2).transpose(1, 0)
#             global_g_data = Data(x=F.normalize(x[0, :, :], dim=3).squeeze(0),
#                                  edge_index=edge_index,
#                                  valid_lens=valid_lens,
#                                  time_step_len=time_step_len)
#         else:
#             raise NotImplementedError
#
#         global_g_data.to(self.device)
#         if self.training:
#             # mask out the features for a random subset of polyline nodes
#             # for one batch, we mask the same polyline features
#
#             global_graph_out = self.global_graph(global_g_data)
#             global_graph_out = global_graph_out.view(-1, time_step_len, self.polyline_vec_shape)
#
#             pred = self.traj_pred_mlp(global_graph_out[:, [0]].squeeze(1))
#             if self.with_aux:
#                 aux_in = global_graph_out.view(-1, self.global_graph_width)[mask_polyline_indices]
#                 aux_out = self.aux_mlp(aux_in)
#
#                 return pred, aux_out, aux_gt
#             else:
#                 return pred, None, None
#
#         else:
#             global_graph_out = self.global_graph(global_g_data)
#             global_graph_out = global_graph_out.view(-1, time_step_len, self.polyline_vec_shape)
#
#             pred = self.traj_pred_mlp(global_graph_out[:, [0]].squeeze(1))
#
#             return pred
#
#     def loss(self, data, reduction="sum"):
#         pass


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
        self.global_graph_width = global_graph_width
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
                mask_polyline_indices = [random.randint(0, time_step_len-1) + i * time_step_len for i in range(x.size()[0])]
                x = x.view(-1, self.subgraph_width)
                aux_gt = x[mask_polyline_indices]
                x[mask_polyline_indices] = 0.0

                x = x.view(-1, time_step_len, self.subgraph_width)
                global_graph_out = self.global_graph(x, valid_lens)
                pred = self.traj_pred_mlp(global_graph_out[:, [0]].squeeze(1))

                aux_in = global_graph_out.view(-1, self.global_graph_width)[mask_polyline_indices]
                aux_out = self.aux_mlp(aux_in)

                return pred, aux_out, aux_gt
            else:
                global_graph_out = self.global_graph(x, valid_lens)
                pred = self.traj_pred_mlp(global_graph_out[:, [0]].squeeze(1))

                return pred, None, None

        else:
            global_graph_out = self.global_graph(x, valid_lens)

            pred = self.traj_pred_mlp(global_graph_out[:, [0]].squeeze(1))

            return pred


# %%
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2

    in_channels, pred_len = 10, 30
    show_every = 10
    os.chdir('..')
    # get model
    model = VectorNet(in_channels, pred_len, with_aux=True).to(device)
    # model = OriginalVectorNet(in_channels, pred_len, with_aux=True).to(device)

    DATA_DIR = "../../dataset/interm_data/"
    TRAIN_DIR = os.path.join(DATA_DIR, 'train_intermediate')
    dataset = ArgoverseInMem(TRAIN_DIR)
    data_iter = DataLoader(dataset, batch_size=batch_size, num_workers=1, pin_memory=True)

    # train mode
    model.train()
    for i, data in enumerate(data_iter):
        # out, aux_out, mask_feat_gt = model(data)
        loss = model.loss(data.to(device))
        print("Training Pass! loss: {}".format(loss))

        if i == 2:
            break

    # eval mode
    model.eval()
    for i, data in enumerate(data_iter):
        out = model(data.to(device))
        print("Evaluation Pass! Shape of out: {}".format(out.shape))

        if i == 2:
            break
