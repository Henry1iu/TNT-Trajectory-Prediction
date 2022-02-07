import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Batch, Data

from core.model.layers.global_graph import GlobalGraph
from core.model.layers.subgraph import SubGraph
from core.dataloader.dataset import GraphDataset


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
        # self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layres)
        self.polyline_vec_shape = subgraph_width * 2
        self.subgraph_width = subgraph_width
        self.global_graph_width = global_graph_width
        self.max_n_guesses = 1

        self.device = device

        # subgraph feature extractor
        self.subgraph = SubGraph(in_channels, num_subgraph_layres, subgraph_width)

        # global graph
        self.global_graph = GlobalGraph(self.polyline_vec_shape,
                                        self.global_graph_width,
                                        num_global_layers=num_global_graph_layer)

        # auxiliary recoverey mlp
        self.with_aux = with_aux
        if self.with_aux:
            self.aux_mlp = nn.Sequential(
                nn.Linear(global_graph_width, aux_mlp_width),
                nn.LayerNorm(aux_mlp_width),
                nn.ReLU(),
                nn.Linear(aux_mlp_width, aux_mlp_width),
                nn.LayerNorm(aux_mlp_width),
                nn.ReLU(),
                nn.Linear(aux_mlp_width, self.polyline_vec_shape)
            )
            # self.aux_mlp.apply(self._init_weights)
            # self.aux_mlp = nn.DataParallel(self.aux_mlp, device_ids=[1, 0])

    # @staticmethod
    # def _init_weights(m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         m.bias.data.fill_(0.01)

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

        if self.training and self.with_aux:
            batch_size = data.num_graphs
            mask_polyline_indices = [random.randint(1, valid_lens[i] - 1) + i * time_step_len for i in range(batch_size)]
            aux_gt = sub_graph_out.x[mask_polyline_indices]
            sub_graph_out.x[mask_polyline_indices] = 0.0

        # TODO: fill the output of subgraph with correct data rather than create new data for global graph
        # reconstruct the batch global interaction graph data
        sub_graph_out.valid_lens = data.valid_len
        sub_graph_out.time_step_len = data.time_step_len
        # sub_graph_out.x = F.normalize(sub_graph_out.x, dim=0)

        edge_index = torch.empty((2, 0), device=self.device, dtype=torch.long)
        # print("[Debug]: data type: {}".format(type(data)))
        if isinstance(data, Batch):
            # mini-batch case
            for idx in range(data.num_graphs):
                node_list = torch.tensor([i for i in range(idx * time_step_len, idx * time_step_len + valid_lens[idx])],
                                         device=self.device).long()
                xx, yy = torch.meshgrid(node_list, node_list)
                xy = torch.vstack([xx.reshape(-1), yy.reshape(-1)])
                edge_index = torch.hstack([edge_index, xy[:, xy[0] != xy[1]]])
                # edge_index = torch.hstack([edge_index, torch.combinations(node_list, 2).T])

        elif isinstance(data, Data):
            # single batch case
            node_list = torch.tensor([i for i in range(valid_lens[0])], device=self.device).long()
            xx, yy = torch.meshgrid(node_list, node_list)
            edge_index = torch.vstack([xx.reshape(-1), yy.reshape(-1)])
            edge_index = edge_index[:, edge_index[0] != edge_index[1]]         # remove the self-loop
            # edge_index = torch.combinations(node_list, 2).T
        else:
            raise NotImplementedError
        sub_graph_out.edge_index = edge_index

        if self.training:
            # mask out the features for a random subset of polyline nodes
            # for one batch, we mask the same polyline features

            global_graph_out = self.global_graph(sub_graph_out, batch_size=data.num_graphs)
            # global_graph_out = self.global_graph(sub_graph_out)
            global_graph_out = global_graph_out.view(-1, time_step_len, self.global_graph_width)

            if self.with_aux:
                aux_in = global_graph_out.view(-1, self.global_graph_width)[mask_polyline_indices]
                aux_out = self.aux_mlp(aux_in)

                return global_graph_out, aux_out, aux_gt
            else:
                return global_graph_out, None, None

        else:
            global_graph_out = self.global_graph(sub_graph_out, batch_size=data.num_graphs)
            # global_graph_out = self.global_graph(sub_graph_out)
            global_graph_out = global_graph_out.view(-1, time_step_len, self.global_graph_width)

            return global_graph_out, None, None


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    decay_lr_factor = 0.9
    decay_lr_every = 10
    lr = 0.005
    in_channels, pred_len = 8, 30
    show_every = 10
    os.chdir('..')
    # get model
    model = VectorNetBackbone(in_channels, pred_len, with_aux=True).to(device)
    # model = OriginalVectorNet(in_channels, pred_len, with_aux=True).to(device)

    DATA_DIR = "dataset/interm_data"
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
