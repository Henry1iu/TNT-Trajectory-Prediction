# trajectory prediction layer of TNT algorithm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class TargetPred(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dim: int = 64,
                 grid_size: float = 10.0,
                 interval: float = 0.5,
                 M: int = 50,
                 device=torch.device("cpu")):
        """"""
        super(TargetPred, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.device = device

        self.grid_size = grid_size      # unit in meter
        self.interval = interval        # unit in meter
        self.M = M

        range = np.arange(-int(self.grid_size), int(self.grid_size), self.interval)
        x_grid, y_grid = np.meshgrid(range, range)
        self.tar_candidate = torch.tensor(
            np.stack([x_grid.flatten(), y_grid.flatten()], axis=1),
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        self.N_tar = self.tar_candidate.size()[1]

        self.prob_mlp = nn.Sequential(
            nn.Linear(in_channels + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=2)
        )

        self.mean_mlp = nn.Sequential(
            nn.Linear(in_channels + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, feat_in: torch.Tensor):
        # dimension must be [batch size, 1, in_channels]
        assert feat_in.dim() == 3, "[TNT-TargetPred]: Error input feature dimension"
        batch_size, _, _ = feat_in.size()

        # stack the target candidates to the end of input feature
        feat_in_repeat = torch.cat(
            [feat_in.repeat(1, self.N_tar, 1), self.tar_candidate.repeat(batch_size, 1, 1)], dim=2
        )
        print("feat_in_repeat size: ", feat_in_repeat.size())

        # compute probability for each candidate
        tar_candit_pro = self.prob_mlp(feat_in_repeat).squeeze(-1)          # (batch_size, self.N_tar, 1)
        tar_offset_mean = self.mean_mlp(feat_in_repeat)         # (batch_size, self.N_tar, 2)
        print("tar_candit_pro size: ", tar_candit_pro.size())
        print("tar_offset_mean size: ", tar_offset_mean.size())

        # compute the prob. of normal distribution
        d_x_dist = Normal(tar_offset_mean[:, :, 0], torch.tensor([1.0]))    # (batch_size, self.N_tar)
        d_y_dist = Normal(tar_offset_mean[:, :, 1], torch.tensor([1.0]))    # (batch_size, self.N_tar)
        d_x = d_x_dist.sample()
        d_y = d_y_dist.sample()

        p = tar_candit_pro * d_x_dist.log_prob(d_x) * d_y_dist.log_prob(d_y)
        _, indices = p.topk(self.M, dim=1)
        return tar_candit_pro, d_x, d_y, indices

    def loss(self, feat_in, gt):
        tar_pred_pro, dx, dy, top_n_indices = self.forward(feat_in)


if __name__ == "__main__":
    in_channels = 64
    in_tensor = torch.randn((4, 1, in_channels)).float()
    layer = TargetPred(in_channels)
    print("total number of params: ", sum(p.numel() for p in layer.parameters()))

    pred, dx, dy = layer(in_tensor)
    print("shape of pred prob: ", pred.size())
    print("shape of dx and dy: ", dx.size())
