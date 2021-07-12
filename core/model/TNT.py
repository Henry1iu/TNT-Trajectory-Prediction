# TNT model
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from core.model.backbone.vectornet import VectorNetBackbone
from core.model.layers.target_prediction import TargetPred
from core.model.layers.motion_etimation import MotionEstimation
from core.model.layers.scoring_and_selection import TrajScoreSelection, distance_metric

from core.dataloader.argoverse_loader import Argoverse, GraphData, ArgoverseInMem


class TNT(nn.Module):
    def __init__(self,
                 in_channels=8,
                 horizon=30,
                 num_subgraph_layers=3,
                 num_global_graph_layer=1,
                 subgraph_width=64,
                 global_graph_width=64,
                 with_aux=False,
                 aux_width=64,
                 n=1000,
                 target_pred_hid=64,
                 m=50,
                 motion_esti_hid=64,
                 score_sel_hid=64,
                 temperature=0.01,
                 k=6,
                 lambda1=0.1,
                 lambda2=1.0,
                 lambda3=0.1,
                 device=torch.device("cpu"),
                 multi_gpu: bool = False):
        """
        TNT algorithm for trajectory prediction
        :param in_channels: int, the number of channels of the input node features
        :param horizon: int, the prediction horizon (prediction length)
        :param num_subgraph_layers: int, the number of subgraph layer
        :param num_global_graph_layer: the number of global interaction layer
        :param subgraph_width: int, the channels of the extrated subgraph features
        :param global_graph_width: int, the channels of extracted global graph feature
        :param with_aux: bool, with aux loss or not
        :param aux_width: int, the hidden dimension of aux recovery mlp
        :param n: int, the number of sampled target candidate
        :param target_pred_hid: int, the hidden dimension of target prediction
        :param m: int, the number of selected candidate
        :param motion_esti_hid: int, the hidden dimension of motion estimation
        :param score_sel_hid: int, the hidden dimension of score module
        :param temperature: float, the temperature when computing the score
        :param k: int, final output trajectories
        :param lambda1: float, the weight of candidate prediction loss
        :param lambda2: float, the weight of motion estimation loss
        :param lambda3: float, the weight of trajectory scoring loss
        :param device: the device for computation
        :param multi_gpu: the multi gpu setting
        """
        super(TNT, self).__init__()
        self.horizon = horizon
        self.n = n
        self.m = m
        self.k = k

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.with_aux = with_aux

        self.device = device

        # feature extraction backbone
        self.backbone = VectorNetBackbone(
            in_channels=in_channels,
            pred_len=horizon,
            num_subgraph_layres=num_subgraph_layers,
            subgraph_width=subgraph_width,
            num_global_graph_layer=num_global_graph_layer,
            global_graph_width=global_graph_width,
            with_aux=with_aux,
            aux_mlp_width=aux_width,
            device=device
        )

        self.target_pred_layer = TargetPred(
            in_channels=global_graph_width,
            hidden_dim=target_pred_hid,
            n=n,
            m=m,
            device=device
        )
        self.motion_estimator = MotionEstimation(
            in_channels=global_graph_width,
            horizon=horizon,
            hidden_dim=motion_esti_hid
        )
        self.traj_score_layer = TrajScoreSelection(
            feat_channels=global_graph_width,
            horizon=horizon,
            hidden_dim=score_sel_hid,
            temper=temperature,
            device=self.device
        )
        self._init_weight()

        # if multi_gpu:
        #     self.target_pred_layer = nn.DataParallel(self.target_pred_layer)
        #     self.motion_estimator = nn.DataParallel(self.motion_estimator)
        #     self.traj_score_layer = nn.DataParallel(self.traj_score_layer)

    def forward(self, data):
        """
        predict the top k most-likely trajectories
        :param data: observed sequence data
        :return:
        """
        target_candidate = data.candidate.view(-1, self.n, 2)    # [batch_size, N, 2]
        batch_size, _, _ = target_candidate.size()

        global_feat, _, _ = self.backbone(data)     # [batch_size, time_step_len, global_graph_width]
        target_feat = global_feat[:, 0]

        # predict the prob. of target candidates and selected the most likely M candidate
        target_pred, offset_pred = self.target_pred_layer(target_feat, target_candidate)

        # DEBUG
        gt = data.y.unsqueeze(1).view(batch_size, -1, 2).cumsum(axis=1)

        # trajectory estimation for the m predicted target location
        traj_pred = self.motion_estimator(target_feat, target_pred+offset_pred)

        # score the predicted trajectory and select the top k trajectory
        score = self.traj_score_layer(target_feat, traj_pred)

        return self.traj_selection(traj_pred, score)

    def loss(self, data, reduction="sum"):
        """
        compute loss according to the gt
        :param data: node feature data
        :param gt: ground truth data
        :param reduction: reduction method, "mean", "sum"
        :return:
        """
        target_candidate = data.candidate.view(-1, self.n, 2)   # [batch_size, N, 2]
        batch_size, _, _ = target_candidate.size()

        global_feat, aux_out, aux_gt = self.backbone(data)              # [batch_size, time_step_len, global_graph_width]
        target_feat = global_feat[:, 0]

        loss = 0.0
        if self.with_aux and self.training:
            aux_loss = F.smooth_l1_loss(aux_out, aux_gt, reduction=reduction)
            loss += aux_loss

        candidate_gt, offset_gt, target_gt, y = data.candidate_gt, data.offset_gt, data.target_gt, data.y

        # add the target prediction loss
        candidate_gt, offset_gt = candidate_gt.view(-1, self.n), offset_gt.view(-1, 2)
        target_loss = self.target_pred_layer.loss(
            target_feat,
            target_candidate,
            candidate_gt,
            offset_gt,
            reduction=reduction
        )
        loss += self.lambda1 * target_loss

        # add the motion estimation loss
        location_gt, traj_gt = target_gt.view(-1, 2).float(), y.view(-1, self.horizon * 2)
        traj_loss, _ = self.motion_estimator.loss(
            target_feat,
            location_gt,
            traj_gt,
            reduction=reduction
        )
        loss += self.lambda2 * traj_loss

        # add the score and selection loss
        pred_tar, pred_offset = self.target_pred_layer(target_feat, target_candidate)
        traj_pred = self.motion_estimator(target_feat, pred_tar+pred_offset)
        score_loss = self.traj_score_layer.loss(
            target_feat,
            traj_pred,
            traj_gt,
            reduction=reduction
        )
        loss += self.lambda3 * score_loss

        return loss, {"target_loss": target_loss, "traj_loss": traj_loss, "score_loss": score_loss}

    def inference(self, data):
        raise NotImplementedError

    def candidate_sampling(self, data):
        """
        sample candidates given the test data
        :param data:
        :return:
        """
        raise NotImplementedError

    # todo: determine appropiate threshold
    def traj_selection(self, traj_in, score, threshold=0.004):
        """
        select the top k trajectories according to the score and the distance
        :param traj_in: candidate trajectories, [batch, M, horizon * 2]
        :param score: score of the candidate trajectories, [batch, M]
        :param threshold: float, the threshold for exclude traj prediction
        :return: [batch_size, k, horizon * 2]
        """
        # re-arrange trajectories according the the descending order of the score
        _, batch_order = score.sort(descending=True)
        traj_pred = torch.cat([traj_in[i, order] for i, order in enumerate(batch_order)], dim=0).view(-1, self.m, self.horizon * 2)
        traj_selected = traj_pred[:, :self.k]           # [batch_size, k, horizon * 2]

        # check the distance between them, NMS
        for batch_id in range(traj_pred.shape[0]):                              # one batch for a time
            traj_cnt = 1
            for i in range(1, self.m):
                dis = distance_metric(traj_selected[batch_id, :traj_cnt], traj_pred[batch_id, i].unsqueeze(0))
                if not torch.any(dis < threshold):                       # not exist similar trajectory
                    traj_selected[batch_id, traj_cnt] = traj_pred[batch_id, i]  # add this trajectory
                    traj_cnt += 1

                if traj_cnt >= self.k:
                    break                                                       # break if collect enough traj

            # no enough traj, pad zero traj
            if traj_cnt < self.k:
                traj_selected[:, traj_cnt:] = 0.0

        return traj_selected

    def _init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                # module.weight.data.kaiming_uniform_(module)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


if __name__ == "__main__":
    batch_size = 256
    # DATA_DIR = "../../dataset/interm_tnt_with_filter"
    DATA_DIR = "../../dataset/interm_tnt_n_s"
    TRAIN_DIR = os.path.join(DATA_DIR, 'train_intermediate')
    # TRAIN_DIR = os.path.join(DATA_DIR, 'val_intermediate')
    # TRAIN_DIR = os.path.join(DATA_DIR, 'test_intermediate')

    dataset = ArgoverseInMem(TRAIN_DIR)
    data_iter = DataLoader(dataset, batch_size=batch_size, num_workers=16, pin_memory=True)

    n = 1000
    m = 50
    k = 6

    device = torch.device("cuda:1")
    # device = torch.device("cpu")

    model = TNT(in_channels=10, n=n, m=m, k=k, with_aux=True, device=device).to(device)
    model.train()

    for i, data in enumerate(tqdm(data_iter)):
        loss, _ = model.loss(data.to(device))
        # print("loss dtype:{}".format(loss.dtype))

        # pred = model(data.to(device))
        # print("\n[TNT/Debug]: shape of {}th pred: {}".format(i, pred.shape))
