# trajectory prediction layer of TNT algorithm with Binary CE Loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from core.model.layers.utils import masked_softmax
from core.model.layers.basic_module import MLP


class TargetPred(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dim: int = 64,
                 m: int = 50,
                 device=torch.device("cpu")):
        """"""
        super(TargetPred, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.M = m          # output candidate target

        self.device = device

        self.prob_mlp = nn.Sequential(
            MLP(in_channels + 2, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

        self.mean_mlp = nn.Sequential(
            MLP(in_channels + 2, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, feat_in: torch.Tensor, tar_candidate: torch.Tensor, candidate_mask=None):
        """
        predict the target end position of the target agent from the target candidates
        :param feat_in:        the encoded trajectory features, [batch_size, inchannels]
        :param tar_candidate:  the target position candidate (x, y), [batch_size, N, 2]
        :param candidate_mask: the mask of valid target candidate
        :return:
        """
        # dimension must be [batch size, 1, in_channels]
        assert feat_in.dim() == 3, "[TNT-TargetPred]: Error input feature dimension"

        batch_size, n, _ = tar_candidate.size()

        # stack the target candidates to the end of input feature
        feat_in_repeat = torch.cat([feat_in.repeat(1, n, 1), tar_candidate], dim=2)

        # compute probability for each candidate
        prob_tensor = self.prob_mlp(feat_in_repeat).squeeze(2)
        if not isinstance(candidate_mask, torch.Tensor):
            tar_candit_prob = F.softmax(prob_tensor, dim=-1)                                    # [batch_size, n_tar]
        else:
            tar_candit_prob = masked_softmax(prob_tensor, candidate_mask, dim=-1)               # [batch_size, n_tar]
        tar_offset_mean = self.mean_mlp(feat_in_repeat)                                         # [batch_size, n_tar, 2]

        return tar_candit_prob, tar_offset_mean

    def loss(self,
             feat_in: torch.Tensor,
             tar_candidate: torch.Tensor,
             candidate_gt: torch.Tensor,
             offset_gt: torch.Tensor,
             candidate_mask=None):
        """
        compute the loss for target prediction, classification gt is binary labels,
        only the closest candidate is labeled as 1
        :param feat_in: encoded feature for the target candidate, [batch_size, inchannels]
        :param tar_candidate: the target candidates for predicting the end position of the target agent, [batch_size, N, 2]
        :param candidate_gt: target prediction ground truth, classification gt and offset gt, [batch_size, N]
        :param offset_gt: the offset ground truth, [batch_size, 2]
        :param candidate_mask:
        :return:
        """
        batch_size, n, _ = tar_candidate.size()
        _, num_cand = candidate_gt.size()

        assert num_cand == n, "The num target candidate and the ground truth one-hot vector is not aligned: {} vs {};".format(n, num_cand)

        # pred prob and compute cls loss
        tar_candit_prob, tar_offset_mean = self.forward(feat_in, tar_candidate, candidate_mask)

        # classfication loss in n candidates
        n_candidate_loss = F.cross_entropy(tar_candit_prob.transpose(1, 2), candidate_gt.long(), reduction='sum')

        # classification loss in m selected candidates
        _, indices = tar_candit_prob[:, :, 1].topk(self.M, dim=1)
        batch_idx = torch.vstack([torch.arange(0, batch_size, device=self.device) for _ in range(self.M)]).T
        # tar_pred_prob_selected = F.normalize(tar_candit_prob[batch_idx, indices], dim=-1)
        # tar_pred_prob_selected = tar_candit_prob[batch_idx, indices]
        # candidate_gt_selected = candidate_gt[batch_idx, indices]
        # m_candidate_loss = F.binary_cross_entropy(tar_pred_prob_selected, candidate_gt_selected, reduction='sum') / batch_size

        # pred offset with gt candidate and compute regression loss
        # feat_in_offset = torch.cat([feat_in.squeeze(1), tar_candidate[candidate_gt]], dim=-1)
        # offset_loss = F.smooth_l1_loss(self.mean_mlp(feat_in_offset), offset_gt, reduction='sum')

        # isolate the loss computation from the candidate target offset prediction
        offset_loss = F.smooth_l1_loss(tar_offset_mean[candidate_gt.bool()], offset_gt, reduction='sum')

        # ====================================== DEBUG ====================================== #
        # # select the M output and check corresponding gt
        # _, indices = tar_candit_prob.topk(self.M, dim=1)
        # batch_idx = torch.vstack([torch.arange(0, batch_size, device=self.device) for _ in range(self.M)]).T
        # tar_pred_prob_selected = F.normalize(tar_candit_prob[batch_idx, indices], dim=-1)
        # tar_pred_selected = tar_candidate[batch_idx, indices]
        # candidate_gt_selected = candidate_gt[batch_idx, indices]
        #
        # tar_candit_prob_cpu = tar_pred_prob_selected.detach().cpu().numpy()
        # candidate_gt_cpu = candidate_gt_selected.detach().cpu().numpy()
        #
        # print("\n[DEBUG]: tar_pred_prob_selected: \n{};\n[DEBUG]: candidate_gt_selected: \n{};".format(tar_candit_prob_cpu,
        #                                                                                                candidate_gt_cpu))
        # print("[DEBUG]: tar_pred_selected: \n{};\n[DEBUG]: tar_gt: \n{};".format(tar_pred_selected.detach().cpu().numpy(),
        #                                                                          tar_candidate[candidate_gt.bool()].detach().cpu().numpy()))
        # # check offset
        # tar_offset_mean_cpu = tar_offset_mean.detach().cpu().numpy()
        # offset_gt_cpu = offset_gt.detach().cpu().numpy()
        # print("[DEBUG]: tar_offset_mean: {};\n[DEBUG]: offset_gt: {};".format(tar_offset_mean_cpu, offset_gt_cpu))
        #
        # # check destination
        # dst_gt = tar_candidate[candidate_gt.bool()] + offset_gt
        # offset = torch.normal(self.mean_mlp(feat_in_prob), std=1.0)[batch_idx, indices]
        # dst_pred = tar_pred_selected + offset
        # print("[DEBUG]: dst_pred: \n{};\n[DEBUG]: dst_gt: \n{};".format(dst_pred.detach().cpu().numpy(),
        #                                                                 dst_gt.detach().cpu().numpy()))
        # ====================================== DEBUG ====================================== #
        # return n_candidate_loss + m_candidate_loss + offset_loss, tar_candidate[batch_idx, indices], tar_offset_mean[batch_idx, indices]
        return n_candidate_loss + offset_loss, tar_candidate[batch_idx, indices], tar_offset_mean[batch_idx, indices]
        # return m_candidate_loss + offset_loss, tar_candidate[batch_idx, indices], tar_offset_mean[batch_idx, indices]

    def inference(self,
                  feat_in: torch.Tensor,
                  tar_candidate: torch.Tensor,
                  candidate_mask=None):
        """
        output only the M predicted propablity of the predicted target
        :param feat_in:        the encoded trajectory features, [batch_size, inchannels]
        :param tar_candidate:  tar_candidate: the target position candidate (x, y), [batch_size, N, 2]
        :param candidate_mask: the mask of valid target candidate
        :return:
        """
        """
        predict the target end position of the target agent from the target candidates
        :param feat_in: the encoded trajectory features, [batch_size, inchannels]
        :param tar_candidate: the target position candidate (x, y), [batch_size, N, 2]
        :param candidate_mask:
        :return:
        """
        return self.forward(feat_in, tar_candidate, candidate_mask)


if __name__ == "__main__":
    batch_size = 4
    in_channels = 64
    N = 1000
    layer = TargetPred(in_channels)
    print("total number of params: ", sum(p.numel() for p in layer.parameters()))

    # forward
    print("test forward")
    feat_tensor = torch.randn((batch_size, 1, in_channels)).float()
    tar_candi_tensor = torch.randn((batch_size, N, 2)).float()
    tar_pred, offset_pred = layer(feat_tensor, tar_candi_tensor)
    print("shape of pred prob: ", tar_pred.size())
    print("shape of dx and dy: ", offset_pred.size())

    # loss
    print("test loss")
    candid_gt = torch.zeros((batch_size, N), dtype=torch.bool)
    candid_gt[:, 5] = 1.0
    offset_gt = torch.randn((batch_size, 2))
    loss = layer.loss(feat_tensor, tar_candi_tensor, candid_gt, offset_gt)

    # inference
    print("test inference")
    tar_candidate, offset = layer.inference(feat_tensor, tar_candi_tensor)
    print("shape of tar_candidate: ", tar_candidate.size())
    print("shape of offset: ", offset.size())


