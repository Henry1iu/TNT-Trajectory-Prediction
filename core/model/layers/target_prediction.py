# trajectory prediction layer of TNT algorithm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class TargetPred(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dim: int = 64,
                 M: int = 50,
                 device=torch.device("cpu")):
        """"""
        super(TargetPred, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.M = M          # output candidate target

        self.device = device

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

    def forward(self, feat_in: torch.Tensor, tar_candidate: torch.Tensor):
        """
        predict the target end position of the target agent from the target candidates
        :param feat_in: the encoded trajectory features, [batch_size, inchannels]
        :param tar_candidate: the target position candidate (x, y), [batch_size, N, 2]
        :return:
        """
        # dimension must be [batch size, 1, in_channels]
        assert feat_in.dim() == 2, "[TNT-TargetPred]: Error input feature dimension"

        feat_in = feat_in.unsqueeze(1)
        batch_size, _, _ = feat_in.size()
        _, N, _ = tar_candidate.size()

        # stack the target candidates to the end of input feature
        feat_in_repeat = torch.cat([feat_in.repeat(1, N, 1), tar_candidate], dim=2)
        print("feat_in_repeat size: ", feat_in_repeat.size())

        # compute probability for each candidate
        tar_candit_pro = self.prob_mlp(feat_in_repeat).squeeze(-1)          # [batch_size, self.N_tar, 1]
        tar_offset_mean = self.mean_mlp(feat_in_repeat)                     # [batch_size, self.N_tar, 2]
        print("tar_candit_pro size: ", tar_candit_pro.size())
        print("tar_offset_mean size: ", tar_offset_mean.size())

        # compute the prob. of normal distribution
        d_x_dist = Normal(tar_offset_mean[:, :, 0], torch.tensor([1.0]))    # [batch_size, self.N_tar]
        d_y_dist = Normal(tar_offset_mean[:, :, 1], torch.tensor([1.0]))    # [batch_size, self.N_tar]
        d_x = d_x_dist.sample()
        d_y = d_y_dist.sample()

        p = tar_candit_pro * d_x_dist.log_prob(d_x) * d_y_dist.log_prob(d_y)
        _, indices = p.topk(self.M, dim=1)

        return tar_candit_pro, d_x, d_y, indices

    def loss(self,
             feat_in: torch.Tensor,
             tar_candidate: torch.Tensor,
             candidate_gt: torch.Tensor,
             offset_gt: torch.Tensor,
             reduction="mean"):
        """
        compute the loss for target prediction, classification gt is binary labels,
        only the closest candidate is labeled as 1
        :param feat_in: encoded feature for the target candidate, [batch_size, inchannels]
        :param tar_candidate: the target candidates for predicting the end position of the target agent, [batch_size, N, 2]
        :param tar_gt: target prediction ground truth, classification gt and offset gt, [batch_size, N]
        :param offset_gt: the offset ground truth, [batch_size, 2]
        :param reduction: the reduction to apply to the loss output
        :return:
        """
        batch_size, N, _ = tar_candidate.size()
        tar_pred_prob, dx, dy, top_m_indices = self.forward(feat_in, tar_candidate)

        # select the M output and gt
        index_offset = torch.arange(0, batch_size).view(batch_size, -1).repeat(1, self.M).view(-1)
        top_m_indices = top_m_indices.view(-1) + index_offset * N
        tar_pred_prob_selected = F.normalize(tar_pred_prob.view(-1)[top_m_indices].view(batch_size, -1))
        candidate_gt_selected = candidate_gt.view(-1)[top_m_indices].unsqueeze(1).view(batch_size, -1)

        # classfication output
        n_candidate_loss = F.binary_cross_entropy(tar_pred_prob, candidate_gt, reduction=reduction)
        m_candidate_loss = F.binary_cross_entropy(tar_pred_prob_selected, candidate_gt_selected, reduction=reduction)

        offset_loss = F.smooth_l1_loss(torch.cat([dx.unsqueeze(2), dy.unsqueeze(2)], dim=2), offset_gt, reduction=reduction)

        return n_candidate_loss + m_candidate_loss + offset_loss

    def inference(self,
                  feat_in: torch.Tensor,
                  tar_candidate: torch.Tensor):
        """
        output only the M predicted propablity of the predicted target
        :param feat_in: the encoded trajectory features, [batch_size, inchannels]
        :param tar_candidate: tar_candidate: the target position candidate (x, y), [batch_size, N, 2]
        :return:
        """
        batch_size, N, _ = tar_candidate.size()
        # get the prob, dx and dy
        tar_pred_prob, dx, dy, top_m_indices = self.forward(feat_in, tar_candidate)

        # select the top M candidate
        index_offset = torch.arange(0, batch_size).view(batch_size, -1).repeat(1, self.M).view(-1)
        top_m_indices = top_m_indices.view(-1) + index_offset * N

        tar_pred_prob_selected = F.normalize(tar_pred_prob.view(-1)[top_m_indices].view(batch_size, -1))
        dx_selected = dx.view(-1)[top_m_indices].unsqueeze(1).view(batch_size, -1)
        dy_selected = dy.view(-1)[top_m_indices].unsqueeze(1).view(batch_size, -1)

        return tar_pred_prob_selected, dx_selected, dy_selected


if __name__ == "__main__":
    batch_size = 4
    in_channels = 64
    N = 1000
    layer = TargetPred(in_channels)
    print("total number of params: ", sum(p.numel() for p in layer.parameters()))

    # forward
    print("test forward")
    feat_tensor = torch.randn((batch_size, in_channels)).float()
    tar_candi_tensor = torch.randn((batch_size, N, 2)).float()
    pred, dx, dy, indices = layer(feat_tensor, tar_candi_tensor)
    print("shape of pred prob: ", pred.size())
    print("shape of dx and dy: ", dx.size())
    print("shape of indices: ", indices.size())

    # loss
    print("test loss")
    candid_gt = torch.randn((batch_size, N))
    offset_gt = torch.randn((batch_size, N, 2))
    loss = layer.loss(feat_tensor, tar_candi_tensor, candid_gt, offset_gt)

    # inference
    print("test inference")
    pred_se, dx_se, dy_se = layer.inference(feat_tensor, tar_candi_tensor)
    print("shape of pred_se: ", pred_se.size())
    print("shape of dx, dy: ", dx_se.size())


