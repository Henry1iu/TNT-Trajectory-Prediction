# motion estimation layer
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model.layers.basic_module import MLP


class MotionEstimation(nn.Module):
    def __init__(self,
                 in_channels,
                 horizon=30,
                 hidden_dim=64):
        """
        estimate the trajectories based on the predicted targets
        :param in_channels:
        :param horizon:
        :param hidden_dim:
        """
        super(MotionEstimation, self).__init__()
        self.in_channels = in_channels
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        # self.traj_pred = nn.Sequential(
        #     nn.Linear(in_channels + 2, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ReLU(inplace=True),
        #     # nn.LeakyReLU(inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, horizon * 2)
        # )
        self.traj_pred = nn.Sequential(
            MLP(in_channels + 2, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, horizon * 2)
        )

    def forward(self, feat_in: torch.Tensor, loc_in: torch.Tensor):
        """
        predict the trajectory according to the target location
        :param feat_in: encoded feature vector for the target agent, torch.Tensor, [batch_size, in_channels]
        :param loc_in: end location, torch.Tensor, [batch_size, M, 2] or [batch_size, 1, 2]
        :return: [batch_size, M, horizon * 2] or [batch_size, 1, horizon * 2]
        """
        assert feat_in.dim() == 3, "[MotionEstimation]: Error dimension in encoded feature input"
        assert feat_in.size()[-1] == self.in_channels, "[MotionEstimation]: Error feature, mismatch in the feature channels!"

        batch_size, M, _ = loc_in.size()
        if M > 1:
            # target candidates
            input = torch.cat([feat_in.repeat(1, M, 1), loc_in], dim=2)
        else:
            # targt ground truth
            input = torch.cat([feat_in, loc_in], dim=-1)

        return self.traj_pred(input)

    def loss(self, feat_in: torch.Tensor, loc_gt: torch.Tensor, traj_gt: torch.Tensor):
        """
        compute loss according to the ground truth target location input
        :param feat_in: feature input of the target agent, torch.Tensor, [batch_size, in_channels]
        :param loc_gt: final target location gt, torch.Tensor, [batch_size, 2]
        :param traj_gt: the gt trajectory, torch.Tensor, [batch_size, horizon * 2]
        :param reduction: reduction of the loss, str
        :return:
        """
        assert feat_in.dim() == 3, "[MotionEstimation]: Error in feature input dimension."
        assert traj_gt.dim() == 2, "[MotionEstimation]: Error in trajectory gt dimension."
        batch_size, _, _ = feat_in.size()

        traj_pred = self.forward(feat_in, loc_gt.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(traj_pred, traj_gt, reduction='sum')
        # loss /= batch_size          # average over batches

        # ====================================== DEBUG ====================================== #
        # print("[DEBUG]: traj_pred: \n{};".format(traj_pred.detach().cpu().numpy()))
        # print("[DEBUG]: traj_gt: \n{};".format(traj_gt.detach().cpu().numpy()))
        # print("[DEBUG]: difference: \n{};".format((traj_pred - traj_gt).detach().cpu().numpy()))
        # ====================================== DEBUG ====================================== #
        return loss

    def inference(self, feat_in: torch.Tensor, loc_in: torch.Tensor):
        """
        predict the trajectory according to the target location
        :param feat_in: encoded feature vector for the target agent, torch.Tensor, [batch_size, in_channels]
        :param loc_in: end location, torch.Tensor, [batch_size, M, 2] or [batch_size, 1, 2]
        :return: [batch_size, M, horizon * 2] or [batch_size, 1, horizon * 2]
        """
        return self.forward(feat_in, loc_in)


if __name__ == "__main__":
    in_ch = 64
    horizon = 30
    batch_size = 4

    layer = MotionEstimation(in_ch, horizon)

    feat_tensor = torch.randn((batch_size, in_ch))
    loc_pred_tensor = torch.randn((batch_size, 50, 2))
    loc_gt_tensor = torch.randn((batch_size, 2))
    traj_gt_tensor = torch.randn((batch_size, horizon * 2))

    # forward
    pred_traj = layer(feat_tensor, loc_pred_tensor)
    print("shape of pred_traj: ", pred_traj.size())

    # loss
    loss, pred_traj = layer.loss(feat_tensor.squeeze(1), loc_gt_tensor, traj_gt_tensor)
    print("shape of pred_traj: ", pred_traj.size())
    print("Pass!")
