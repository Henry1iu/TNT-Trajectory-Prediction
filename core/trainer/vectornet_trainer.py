from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch_geometric.nn import DataParallel

from core.trainer.trainer import Trainer
from core.model.vectornet import VectorNet, OriginalVectorNet
from core.optim_schedule import ScheduledOptim


class VectorNetTrainer(Trainer):
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
                 warmup_epoch=15,
                 lr_update_freq=5,
                 lr_decay_rate=0.3,
                 aux_loss: bool = False,
                 with_cuda: bool = False,
                 cuda_device=None,
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
        super(VectorNetTrainer, self).__init__(
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
            log_freq=log_freq,
            save_folder=save_folder,
            verbose=verbose
        )

        # init or load model
        self.aux_loss = aux_loss
        # input dim: (20, 8); output dim: (30, 2)
        model_name = VectorNet
        # model_name = OriginalVectorNet
        self.model = model_name(
            self.trainset.num_features,
            horizon,
            num_global_graph_layer=num_global_graph_layer,
            with_aux=aux_loss,
            device=self.device
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
        self.optim = Adam(self.model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        self.optm_schedule = ScheduledOptim(
            self.optim,
            self.lr,
            n_warmup_epoch=self.warmup_epoch,
            update_rate=lr_update_freq,
            decay_rate=lr_decay_rate
        )

        # load ckpt
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
            n_graph = data.num_graphs
            if training:
                if self.multi_gpu:
                    # loss = self.model.module.loss(data.to(self.device))
                    loss = self.model.loss(data.to(self.device))
                else:
                    loss = self.model.loss(data.to(self.device))

                self.optm_schedule.zero_grad()
                loss.backward()
                self.optim.step()
                self.write_log("Train Loss", loss.detach().item() / n_graph, i + epoch * len(dataloader))

            else:
                with torch.no_grad():
                    if self.multi_gpu:
                        # loss = self.model.module.loss(data.to(self.device))
                        loss = self.model.loss(data.to(self.device))
                    else:
                        loss = self.model.loss(data.to(self.device))
                    self.write_log("Eval Loss", loss.item() / n_graph, i + epoch * len(dataloader))

            num_sample += n_graph
            avg_loss += loss.detach().item()

            # print log info
            desc_str = "[Info: {}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}]".format("train" if training else "eval",
                                                                                 epoch,
                                                                                 loss.item() / n_graph,
                                                                                 avg_loss / num_sample)
            data_iter.set_description(desc=desc_str, refresh=True)

        if training:
            learning_rate = self.optm_schedule.step_and_update_lr()
            self.write_log("LR", learning_rate, epoch)

        return avg_loss / num_sample

    # todo: the inference of the model
    def test(self, data):
        raise NotImplementedError
