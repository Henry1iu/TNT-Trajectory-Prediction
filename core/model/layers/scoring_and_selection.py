# score the predicted trajectories

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model.layers.basic_module import MLP


def distance_metric(traj_candidate: torch.Tensor, traj_gt: torch.Tensor):
    """
    compute the distance between the candidate trajectories and gt trajectory
    :param traj_candidate: torch.Tensor, [batch_size, M, horizon * 2] or [M, horizon * 2]
    :param traj_gt: torch.Tensor, [batch_size, horizon * 2] or [1, horizon * 2]
    :return: distance, torch.Tensor, [batch_size, M] or [1, M]
    """
    assert traj_gt.dim() == 2, "Error dimension in ground truth trajectory"
    if traj_candidate.dim() == 3:
        # batch case
        pass

    elif traj_candidate.dim() == 2:
        traj_candidate = traj_candidate.unsqueeze(1)
    else:
        raise NotImplementedError

    assert traj_candidate.size()[2] == traj_gt.size()[1], "Miss match in prediction horizon!"

    _, M, horizon_2_times = traj_candidate.size()
    dis = torch.pow(traj_candidate - traj_gt.unsqueeze(1), 2).view(-1, M, int(horizon_2_times / 2), 2)

    dis, _ = torch.max(torch.sum(dis, dim=3), dim=2)

    return dis


class TrajScoreSelection(nn.Module):
    def __init__(self,
                 feat_channels,
                 horizon=30,
                 hidden_dim=64,
                 temper=0.01,
                 device=torch.device("cpu")):
        """
        init trajectories scoring and selection module
        :param feat_channels: int, number of channels
        :param horizon: int, prediction horizon, prediction time x pred_freq
        :param hidden_dim: int, hidden dimension
        :param temper: float, the temperature
        """
        super(TrajScoreSelection, self).__init__()
        self.feat_channels = feat_channels
        self.horizon = horizon
        self.temper = temper

        self.device = device

        # self.score_mlp = nn.Sequential(
        #     nn.Linear(feat_channels + horizon * 2, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, 1)
        # )
        self.score_mlp = nn.Sequential(
            MLP(feat_channels + horizon * 2, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feat_in: torch.Tensor, traj_in: torch.Tensor):
        """
        forward function
        :param feat_in: input feature tensor, torch.Tensor, [batch_size, feat_channels]
        :param traj_in: candidate trajectories, torch.Tensor, [batch_size, M, horizon * 2]
        :return: [batch_size, M]
        """
        assert feat_in.dim() == 3, "[TrajScoreSelection]: Error in input feature dimension."
        assert traj_in.dim() == 3, "[TrajScoreSelection]: Error in candidate trajectories dimension"

        batch_size, M, _ = traj_in.size()
        input_tenor = torch.cat([feat_in.repeat(1, M, 1), traj_in], dim=2)

        return F.softmax(self.score_mlp(input_tenor).squeeze(-1), dim=-1)

    def loss(self, feat_in, traj_in, traj_gt):
        """
        compute loss
        :param feat_in: input feature, torch.Tensor, [batch_size, feat_channels]
        :param traj_in: candidate trajectories, torch.Tensor, [batch_size, M, horizon * 2]
        :param traj_gt: gt trajectories, torch.Tensor, [batch_size, horizon * 2]
        :return:
        """
        # batch_size = traj_in.shape[0]

        # compute ground truth score
        score_gt = F.softmax(-distance_metric(traj_in, traj_gt)/self.temper, dim=1)
        score_pred = self.forward(feat_in, traj_in)

        # return F.mse_loss(score_pred, score_gt, reduction='sum')
        logprobs = - torch.log(score_pred)

        # loss = torch.sum(torch.mul(logprobs, score_gt)) / batch_size
        loss = torch.sum(torch.mul(logprobs, score_gt))
        # if reduction == 'mean':
        #     loss = torch.sum(torch.mul(logprobs, score_gt)) / batch_size
        # else:
        #     loss = torch.sum(torch.mul(logprobs, score_gt))
        return loss

    def inference(self, feat_in: torch.Tensor, traj_in: torch.Tensor):
        """
        forward function
        :param feat_in: input feature tensor, torch.Tensor, [batch_size, feat_channels]
        :param traj_in: candidate trajectories, torch.Tensor, [batch_size, M, horizon * 2]
        :return: [batch_size, M]
        """
        return self.forward(feat_in, traj_in)


if __name__ == "__main__":
    feat_in = 64
    horizon = 30
    layer = TrajScoreSelection(feat_in, horizon)

    batch_size = 4

    feat_tensor = torch.randn((batch_size, feat_in))
    traj_in = torch.randn((batch_size, 50, horizon * 2))
    traj_gt = torch.randn((batch_size, horizon * 2))

    traj_in[:, 0, :] = traj_gt

    # forward
    score = layer(feat_tensor, traj_in)
    print("shape of score: ", score.size())

    # loss
    loss = layer.loss(feat_tensor, traj_in, traj_gt)
    print("Pass")
