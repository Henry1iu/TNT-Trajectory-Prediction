from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch_geometric.data import DataLoader
from torch_geometric.nn import DataParallel
from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate

from core.trainer.trainer import Trainer
from core.model.TNT import TNT
from core.optim_schedule import ScheduledOptim
from core.util.viz_utils import show_pred_and_gt


class TNTTrainer(Trainer):
    """
    VectorNetTrainer, train the vectornet with specified hyperparameters and configurations
    """
    def __init__(self,
                 trainset,
                 evalset,
                 testset,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 num_global_graph_layer=1,
                 horizon: int = 30,
                 lr: float = 1e-3,
                 betas=(0.9, 0.999),
                 weight_decay: float = 0.01,
                 warmup_epoch=30,
                 lr_update_freq=5,
                 lr_decay_rate=0.3,
                 aux_loss: bool = False,
                 with_cuda: bool = False,
                 cuda_device=None,
                 enable_log=True,
                 log_freq: int = 2,
                 save_folder: str = "",
                 model_path: str = None,
                 ckpt_path: str = None,
                 verbose: bool = True
                 ):
        """
        trainer class for vectornet
        :param train_loader: see parent class
        :param eval_loader: see parent class
        :param test_loader: see parent class
        :param lr: see parent class
        :param betas: see parent class
        :param weight_decay: see parent class
        :param warmup_steps: see parent class
        :param with_cuda: see parent class
        :param multi_gpu: see parent class
        :param log_freq: see parent class
        :param model_path: str, the path to a trained model
        :param ckpt_path: str, the path to a stored checkpoint to be resumed
        :param verbose: see parent class
        """
        super(TNTTrainer, self).__init__(
            trainset=trainset,
            evalset=evalset,
            testset=testset,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            warmup_epoch=warmup_epoch,
            with_cuda=with_cuda,
            cuda_device=cuda_device,
            enable_log=enable_log,
            log_freq=log_freq,
            save_folder=save_folder,
            verbose=verbose
        )

        # init or load model
        self.aux_loss = aux_loss
        # input dim: (20, 8); output dim: (30, 2)
        # model_name = VectorNet
        model_name = TNT
        self.model = model_name(
            self.trainset.num_features,
            horizon,
            num_global_graph_layer=num_global_graph_layer,
            with_aux=aux_loss,
            device=self.device,
            multi_gpu=self.multi_gpu
        )

        # resume from model file or maintain the original
        if model_path:
            self.load(model_path, 'm')

        if self.multi_gpu:
            # self.model = DataParallel(self.model)
            if self.verbose:
                print("[TNTTrainer]: Train the mode with multiple GPUs: {}.".format(self.cuda_id))
        else:
            if self.verbose:
                print("[TNTTrainer]: Train the mode with single device on {}.".format(self.device))
        self.model = self.model.to(self.device)

        # init optimizer
        self.optim = AdamW(self.model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        self.optm_schedule = ScheduledOptim(
            self.optim,
            self.lr,
            n_warmup_epoch=self.warmup_epoch,
            update_rate=lr_update_freq,
            decay_rate=lr_decay_rate
        )
        # record the init learning rate
        self.write_log("LR", self.lr, 0)

        # resume training from ckpt
        if ckpt_path:
            self.load(ckpt_path, 'c')

    def iteration(self, epoch, dataloader):
        training = self.model.training
        avg_loss = 0.0
        num_sample = 0

        data_iter = tqdm(
            enumerate(dataloader),
            desc="{}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}".format("train" if training else "eval",
                                                                   epoch,
                                                                   0.0,
                                                                   avg_loss),
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}"
        )

        for i, data in data_iter:
            if not self.multi_gpu:
                n_graph = data.num_graphs
            else:
                n_graph = len(data)
            # ################################### DEBUG ################################### #
            # if epoch > 0:
            #     print("\nsize of x: {};".format(data.x.shape))
            #     print("size of cluster: {};".format(data.cluster.shape))
            #     print("valid_len: {};".format(data.valid_len))
            #     print("time_step_len: {};".format(data.time_step_len))
            #
            #     print("size of candidate: {};".format(data.candidate.shape))
            #     print("size of candidate_mask: {};".format(data.candidate_mask.shape))
            #     print("candidate_len_max: {};".format(data.candidate_len_max))
            # ################################### DEBUG ################################### #

            if training:
                if self.multi_gpu:
                    # loss, loss_dict = self.model.module.loss(data.to(self.device))
                    loss, loss_dict = self.model.loss(data.to(self.device))
                    # loss, loss_dict = self.model.module.loss(data)
                else:
                    loss, loss_dict = self.model.loss(data.to(self.device))

                self.optm_schedule.zero_grad()
                loss.backward()
                self.optim.step()

                # writing loss
                self.write_log("Train_Loss", loss.detach().item() / n_graph, i + epoch * len(dataloader))
                self.write_log("Target_Loss", loss_dict["target_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                self.write_log("Traj_Loss", loss_dict["traj_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                self.write_log("Score_Loss", loss_dict["score_loss"].detach().item() / n_graph, i + epoch * len(dataloader))

            else:
                with torch.no_grad():
                    if self.multi_gpu:
                        # loss, loss_dict = self.model.module.loss(data.to(self.device))
                        loss, loss_dict = self.model.loss(data.to(self.device))
                    else:
                        loss, loss_dict = self.model.loss(data.to(self.device))

                    # writing loss
                    self.write_log("Eval_Loss", loss.item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("Target_Loss(Eval)",
                                   loss_dict["target_loss"].item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("Traj_Loss(Eval)",
                                   loss_dict["traj_loss"].item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("Score_Loss(Eval)",
                                   loss_dict["score_loss"].item() / n_graph, i + epoch * len(dataloader))

            num_sample += n_graph
            avg_loss += loss.detach().item()

            desc_str = "[Info: {}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}]".format("train" if training else "eval",
                                                                                 epoch,
                                                                                 loss.detach().item() / n_graph,
                                                                                 avg_loss / num_sample)
            data_iter.set_description(desc=desc_str, refresh=True)

        if training:
            learning_rate = self.optm_schedule.step_and_update_lr()
            self.write_log("LR", learning_rate, epoch)

        return avg_loss

    # todo: the inference of the model
    def test(self, miss_threshold=2.0):
        self.model.eval()

        forecasted_trajectories, gt_trajectories = {}, {}
        seq_id = 0

        # k = self.model.k if not self.multi_gpu else self.model.module.k
        k = self.model.k
        # horizon = self.model.horizon if not self.multi_gpu else self.model.module.horizon
        horizon = self.model.horizon

        with torch.no_grad():
            for data in tqdm(self.test_loader):
                batch_size = data.num_graphs
                gt = data.y.unsqueeze(1).view(batch_size, -1, 2).cumsum(axis=1).numpy()

                # inference and transform dimension
                if self.multi_gpu:
                    # out = self.model.module(data.to(self.device))
                    out = self.model(data.to(self.device))
                else:
                    out = self.model(data.to(self.device))
                dim_out = len(out.shape)
                pred_y = out.unsqueeze(dim_out).view((batch_size, k, horizon, 2)).cumsum(axis=2).cpu().numpy()

                # record the prediction and ground truth
                for batch_id in range(batch_size):
                    forecasted_trajectories[seq_id] = [pred_y_k for pred_y_k in pred_y[batch_id]]
                    gt_trajectories[seq_id] = gt[batch_id]
                    seq_id += 1

        # compute the metric
        metric_results = get_displacement_errors_and_miss_rate(
            forecasted_trajectories,
            gt_trajectories,
            k,
            horizon,
            miss_threshold
        )
        print("[TNTTrainer]: The test result: {};".format(metric_results))

        # plot the result
        fig, ax = plt.subplots()
        for key in forecasted_trajectories.keys():
            ax.set_xlim(-15, 15)
            show_pred_and_gt(ax, gt_trajectories[key], forecasted_trajectories[key])
            plt.pause(3)
            ax.clear()
