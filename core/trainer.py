# trainner to train the models

import os
from typing import Dict

from tqdm import tqdm
import json

import torch
import torch.nn as nn
from torch.optim import Adam
# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader
from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate

from core.model.vectornet import VectorNet
from core.optim_schedule import ScheduledOptim
from core.loss import VectorLoss


class Trainer(object):
    """
    Parent class for all the trainer class
    """
    def __init__(self,
                 train_loader: DataLoader,
                 eval_loader: DataLoader,
                 test_laoder: DataLoader = None,
                 batch_size: int = 1,
                 lr: float = 1e-4,
                 betas=(0.9, 0.999),
                 weight_decay: float = 0.01,
                 warmup_epoch=5,
                 with_cuda: bool = True,
                 multi_gpu: bool = False,
                 log_freq: int = 2,
                 verbose: bool = True
                 ):
        """
        :param train_loader: train dataset dataloader
        :param eval_loader: eval dataset dataloader
        :param test_laoder: dataset dataloader
        :param lr: initial learning rate
        :param betas: Adam optiimzer betas
        :param weight_decay: Adam optimizer weight decay param
        :param warmup_steps: optimizatioin scheduler param
        :param with_cuda: tag indicating whether using gpu for training
        :param multi_gpu: tag indicating whether multiple gpus are using
        :param log_freq: logging frequency in epoch
        :param verbose: whether printing debug messages
        """
        # cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() and with_cuda else "cpu")

        # dataset
        self.trainset = train_loader
        self.evalset = eval_loader
        self.testset = test_laoder
        self.batch_size = batch_size

        # model
        self.model = None
        self.multi_gpu = multi_gpu

        # optimizer params
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_epoch = warmup_epoch
        self.optim = None
        self.optm_schedule = None

        # criterion and metric
        self.criterion = None
        self.min_eval_loss = None

        # log
        self.log_freq = log_freq
        self.verbose = verbose

    def train(self, epoch):
        raise NotImplementedError

    def eval(self, epoch):
        raise NotImplementedError

    def test(self, data):
        raise NotImplementedError

    def iteration(self, epoch, dataloader):
        raise NotImplementedError

    # todo: save the model and current training status
    def save(self, save_folder, iter_epoch, loss):
        """
        save current state of the training and update the minimum loss value
        :param save_folder: str, the destination folder to store the ckpt
        :param iter_epoch: int, ith epoch of current saving checkpoint
        :param loss: float, the loss of current saving state
        :return:
        """
        self.min_eval_loss = loss
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        torch.save({
            "epoch": iter_epoch,
            "model_state_dict": self.model.state_dict() if not self.multi_gpu else self.model.module.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "min_eval_loss": loss
        }, os.path.join(save_folder, "checkpoint_iter{}.ckpt".format(iter_epoch)))
        if self.verbose:
            print("[Trainer]: Saving checkpoint to {}...".format(save_folder))

    def save_model(self, save_folder, prefix=""):
        """
        save current state of the model
        :param save_folder: str, the folder to store the model file
        :return:
        """
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        torch.save(
            self.model.state_dict() if not self.multi_gpu else self.model.module.state_dict(),
            os.path.join(save_folder, "{}_{}.pth".format(prefix, type(self.model).__name__))
        )
        if self.verbose:
            print("[Trainer]: Saving model to {}...".format(save_folder))

        # compute the metrics and save
        _ = self.compute_metric(stored_file=os.path.join(save_folder, "{}_metrics.txt".format(prefix)))

    def load(self, load_path, mode='c'):
        """
        loading function to load the ckpt or model
        :param mode: str, "c" for checkpoint, or "m" for model
        :param load_path: str, the path of the file to be load
        :return:
        """
        if mode == 'c':
            # load ckpt
            ckpt = torch.load(load_path)
            try:
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.optim.load_state_dict(ckpt["optimizer_state_dict"])
                self.min_eval_loss = ckpt["min_eval_loss"]
            except:
                raise Exception("[Trainer]: Error in loading the checkpoint file {}".format(load_path))
        elif mode == 'm':
            try:
                self.model.load_state_dict(torch.load(load_path))
            except:
                raise Exception("[Trainer]: Error in loading the model file {}".format(load_path))
        else:
            raise NotImplementedError

    def compute_metric(self, miss_threshold=2.0, stored_file=None):
        """
        compute metric for test dataset
        :param miss_threshold: float,
        :param stored_file: str, store the result metric in the file
        :return:
        """
        assert self.model, "[Trainer]: No valid model, metrics can't be computed!"
        assert self.testset, "[Trainer]: No test dataset, metrics can't be computed!"

        forecasted_trajectories, gt_trajectories = {}, {}
        seq_id = 0
        self.model.eval()
        with torch.no_grad():
            for data in self.testset:
                gt = data.y.view(-1, 2).cumsum(axis=0).numpy()

                # inference and transform dimension
                out = self.model(data.to(self.device))
                pred_y = out.view((-1, 2)).cumsum(axis=0).cpu().numpy()

                # record the prediction and ground truth
                forecasted_trajectories[seq_id] = [pred_y]
                gt_trajectories[seq_id] = gt
                seq_id += 1

            metric_results = get_displacement_errors_and_miss_rate(
                forecasted_trajectories,
                gt_trajectories,
                self.model.max_n_guesses,
                self.model.pred_len,
                miss_threshold
            )
        if stored_file:
            with open(stored_file, 'w+') as f:
                assert isinstance(metric_results, dict), "[Trainer] The metric evaluation result is not valid!"
                f.write(json.dumps(metric_results))
        return metric_results


class VectorNetTrainer(Trainer):
    """
    VectorNetTrainer, train the vectornet with specified hyperparameters and configurations
    """
    def __init__(self,
                 train_loader: DataLoader,
                 eval_loader: DataLoader,
                 test_laoder: DataLoader = None,
                 batch_size: int = 1,
                 num_global_graph_layer=1,
                 lr: float = 1e-4,
                 betas=(0.9, 0.999),
                 weight_decay: float = 0.01,
                 warmup_epoch=5,
                 aux_loss: bool = False,
                 with_cuda: bool = True,
                 multi_gpu: bool = False,
                 log_freq: int = 2,
                 model_path: str = None,
                 ckpt_path: str = None,
                 verbose: bool = True
                 ):
        """
        trainer class for vectornet
        :param train_loader: see parent class
        :param eval_loader: see parent class
        :param test_laoder: see parent class
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
            train_loader=train_loader,
            eval_loader=eval_loader,
            test_laoder=test_laoder,
            batch_size=batch_size,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            warmup_epoch=warmup_epoch,
            with_cuda=with_cuda,
            multi_gpu=multi_gpu,
            log_freq=log_freq,
            verbose=verbose
        )

        # init or load model
        self.aux_loss = aux_loss
        # input dim: (20, 8); output dim: (30, 2)
        self.model = VectorNet(8,                                   # input 20 time step with 8 features each time step
                               30,                                  # output 30 time step with 2 offset each time step
                               num_global_graph_layer=num_global_graph_layer,
                               with_aux=aux_loss,
                               device=self.device)

        if not model_path:
            if self.multi_gpu:
                self.model = nn.DataParallel(self.model)
            self.model.to(self.device)
        else:
            self.load(model_path, 'm')

        # init optimizer
        self.optim = Adam(self.model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        self.optm_schedule = ScheduledOptim(self.optim, self.lr, n_warmup_epoch=self.warmup_epoch)


        # loss function
        self.criterion = VectorLoss(aux_loss=aux_loss)

        # load ckpt
        if ckpt_path:
            self.load(ckpt_path, 'c')

    def train(self, epoch):
        self.model.train()
        return self.iteration(epoch, self.trainset)

    def eval(self, epoch):
        self.model.eval()
        return self.iteration(epoch, self.evalset)

    def iteration(self, epoch, dataloader):
        training = self.model.training
        avg_loss = 0.0
        num_sample = 0

        data_iter = tqdm(enumerate(dataloader),
                         desc="{}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}".format("train" if training else "eval",
                                                                               epoch,
                                                                               0.0,
                                                                               avg_loss),
                         total=len(dataloader),
                         bar_format="{l_bar}{r_bar}")

        for i, data in data_iter:
            if training:
                pred, aux_out, aux_gt = self.model(data.to(self.device))
                loss = self.criterion(pred,
                                      data.y.view(-1, self.model.out_channels * self.model.pred_len),
                                      aux_out,
                                      aux_gt)

                self.optm_schedule.zero_grad()
                loss.backward()

            else:
                with torch.no_grad():
                    pred = self.model(data.to(self.device))
                    loss = self.criterion(pred,
                                          data.y.view(-1, self.model.out_channels * self.model.pred_len))

            num_sample += self.batch_size
            avg_loss += loss.item()

            # print log info
            # log = {
            #     "iter": i,
            #     "loss": loss.item(),
            #     "avg_loss": avg_loss / num_sample
            # }
            # data_iter.write(str(log))
            desc_str = "{}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}".format("train" if training else "eval",
                                                                     epoch,
                                                                     loss.item(),
                                                                     avg_loss / num_sample)
            data_iter.set_description(desc=desc_str, refresh=True)

        self.optm_schedule.step_and_update_lr()
        return avg_loss / num_sample

    # todo: the inference of the model
    def test(self, data):
        raise NotImplementedError
