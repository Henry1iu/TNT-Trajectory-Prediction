# trainner to train the models

import os
from tqdm import tqdm

import json

import torch

import torch.distributed as dist
from torch.utils.data import distributed
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader, DataListLoader
from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate

import gc


class Trainer(object):
    """
    Parent class for all the trainer class
    """
    def __init__(self,
                 trainset,
                 evalset,
                 testset,
                 loader=DataLoader,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 lr: float = 1e-4,
                 betas=(0.9, 0.999),
                 weight_decay: float = 0.01,
                 warmup_epoch=30,
                 with_cuda: bool = False,
                 cuda_device=None,
                 multi_gpu: bool = False,
                 enable_log: bool = False,
                 log_freq: int = 2,
                 save_folder: str = "",
                 verbose: bool = True
                 ):
        """
        :param trainset: train dataset
        :param evalset: eval dataset
        :param testset: dataset
        :param loader: data loader
        :param lr: initial learning rate
        :param betas: Adam optiimzer betas
        :param weight_decay: Adam optimizer weight decay param
        :param warmup_epoch: optimizatioin scheduler param
        :param with_cuda: tag indicating whether using gpu for training
        :param cuda_device: tag indicating whether multiple gpus are using
        :param log_freq: logging frequency in epoch
        :param verbose: whether printing debug messages
        """
        # determine cuda device id
        self.cuda_id = cuda_device if with_cuda and cuda_device else 0
        self.device = torch.device("cuda:{}".format(self.cuda_id) if torch.cuda.is_available() and with_cuda else "cpu")
        torch.backends.cudnn.benchmark = True if torch.cuda.is_available() and with_cuda else False     # boost cudnn
        if 'WORLD_SIZE' in os.environ and multi_gpu:
            self.multi_gpu = True if int(os.environ['WORLD_SIZE']) > 1 else False
        else:
            self.multi_gpu = False

        torch.manual_seed(self.cuda_id)
        if self.multi_gpu:
            torch.cuda.set_device(self.cuda_id)
            dist.init_process_group(backend='nccl', init_method='env://')

        # dataset
        self.trainset = trainset
        self.evalset = evalset
        self.testset = testset
        self.batch_size = batch_size
        # self.loader = loader if not self.multi_gpu else DataListLoader
        self.loader = loader
        # print("[Debug]: using {} to load data".format(self.loader))

        if self.multi_gpu:
            # datset sampler when training with distributed data parallel model
            self.train_sampler = distributed.DistributedSampler(
                self.trainset,
                num_replicas=int(os.environ['WORLD_SIZE']),
                rank=self.cuda_id
            )
            self.eval_sampler = distributed.DistributedSampler(
                self.evalset,
                num_replicas=int(os.environ['WORLD_SIZE']),
                rank=self.cuda_id
            )
            self.test_sampler = distributed.DistributedSampler(
                self.testset,
                num_replicas=int(os.environ['WORLD_SIZE']),
                rank=self.cuda_id
            )

            self.train_loader = self.loader(
                self.trainset,
                batch_size=self.batch_size,
                num_workers=0,
                pin_memory=True,
                shuffle=False,
                sampler=self.train_sampler
            )
            self.eval_loader = self.loader(self.evalset, batch_size=self.batch_size, num_workers=0, sampler=self.eval_sampler)
            # self.eval_loader = self.loader(self.evalset, batch_size=self.batch_size)
            self.test_loader = self.loader(self.testset, batch_size=self.batch_size, num_workers=0, sampler=self.test_sampler)
            # self.test_loader = self.loader(self.testset, batch_size=self.batch_size)
        else:
            self.train_loader = self.loader(
                self.trainset,
                batch_size=self.batch_size,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=True
            )
            self.eval_loader = self.loader(self.evalset, batch_size=self.batch_size, num_workers=num_workers)
            # self.eval_loader = self.loader(self.evalset, batch_size=self.batch_size)
            self.test_loader = self.loader(self.testset, batch_size=self.batch_size, num_workers=num_workers)
            # self.test_loader = self.loader(self.testset, batch_size=self.batch_size)

        # model
        self.model = None

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
        self.best_metric = None

        # log
        self.enable_log = enable_log
        self.save_folder = save_folder
        if not self.multi_gpu or (self.multi_gpu and self.cuda_id == 1):
            self.logger = SummaryWriter(log_dir=os.path.join(self.save_folder, "log"))
        self.log_freq = log_freq
        self.verbose = verbose

        gc.enable()

    def train(self, epoch):
        gc.collect()

        self.model.train()
        return self.iteration(epoch, self.train_loader)

    def eval(self, epoch):
        gc.collect()

        self.model.eval()
        return self.iteration(epoch, self.eval_loader)

    def test(self):
        raise NotImplementedError

    def iteration(self, epoch, dataloader):
        raise NotImplementedError

    def compute_loss(self, data):
        raise NotImplementedError

    def write_log(self, name_str, data, epoch):
        if not self.enable_log:
            return
        self.logger.add_scalar(name_str, data, epoch)

    # todo: save the model and current training status
    def save(self, iter_epoch, loss):
        """
        save current state of the training and update the minimum loss value
        :param save_folder: str, the destination folder to store the ckpt
        :param iter_epoch: int, ith epoch of current saving checkpoint
        :param loss: float, the loss of current saving state
        :return:
        """
        if self.multi_gpu and self.cuda_id != 1:
            return

        self.min_eval_loss = loss
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)
        torch.save({
            "epoch": iter_epoch,
            "model_state_dict": self.model.state_dict() if not self.multi_gpu else self.model.module.state_dict(),
            # "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "min_eval_loss": loss
        }, os.path.join(self.save_folder, "checkpoint_iter{}.ckpt".format(iter_epoch)))
        if self.verbose:
            print("[Trainer]: Saving checkpoint to {}...".format(self.save_folder))

    def save_model(self, prefix=""):
        """
        save current state of the model
        :param prefix: str, the prefix to the model file
        :return:
        """
        if self.multi_gpu and self.cuda_id != 1:
            return

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)

        # compute the metrics and save
        metric = self.compute_metric()

        # skip model saving if the minADE is not better
        if self.best_metric and isinstance(metric, dict):
            if metric["minADE"] >= self.best_metric["minADE"]:
                print("[Trainer]: Best minADE: {}; Current minADE: {}; Skip model saving...".format(
                    self.best_metric["minADE"],
                    metric["minADE"]))
                return

        # save best metric
        if self.verbose:
            print("[Trainer]: Best minADE: {}; Current minADE: {}; Saving model to {}...".format(
                self.best_metric["minADE"] if self.best_metric else "Inf",
                metric["minADE"],
                self.save_folder))
        self.best_metric = metric
        metric_stored_file = os.path.join(self.save_folder, "{}_metrics.txt".format(prefix))
        with open(metric_stored_file, 'a+') as f:
            f.write(json.dumps(self.best_metric))
            f.write("\n")

        # save model
        torch.save(
            self.model.state_dict() if not self.multi_gpu else self.model.module.state_dict(),
            # self.model.state_dict(),
            os.path.join(self.save_folder, "{}_{}.pth".format(prefix, type(self.model).__name__))
        )

    def load(self, load_path, mode='c'):
        """
        loading function to load the ckpt or model
        :param mode: str, "c" for checkpoint, or "m" for model
        :param load_path: str, the path of the file to be load
        :return:
        """
        if mode == 'c':
            # load ckpt
            ckpt = torch.load(load_path, map_location=self.device)
            try:
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.optim.load_state_dict(ckpt["optimizer_state_dict"])
                self.min_eval_loss = ckpt["min_eval_loss"]
            except:
                raise Exception("[Trainer]: Error in loading the checkpoint file {}".format(load_path))
        elif mode == 'm':
            try:
                self.model.load_state_dict(torch.load(load_path, map_location=self.device))
            except:
                raise Exception("[Trainer]: Error in loading the model file {}".format(load_path))
        else:
            raise NotImplementedError

    def compute_metric(self, miss_threshold=2.0):
        """
        compute metric for test dataset
        :param miss_threshold: float,
        :return:
        """
        assert self.model, "[Trainer]: No valid model, metrics can't be computed!"
        assert self.testset, "[Trainer]: No test dataset, metrics can't be computed!"

        forecasted_trajectories, gt_trajectories = {}, {}
        seq_id = 0

        # k = self.model.k if not self.multi_gpu else self.model.module.k
        # horizon = self.model.horizon if not self.multi_gpu else self.model.module.horizon
        k = self.model.k if not self.multi_gpu else self.model.module.k
        horizon = self.model.horizon if not self.multi_gpu else self.model.module.horizon

        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                batch_size = data.num_graphs
                gt = data.y.unsqueeze(1).view(batch_size, -1, 2).cumsum(axis=1).numpy()

                # inference and transform dimension
                if self.multi_gpu:
                    out = self.model.module.inference(data.to(self.device))
                else:
                    out = self.model.inference(data.to(self.device))
                pred_y = out.cpu().numpy()

                # record the prediction and ground truth
                for batch_id in range(batch_size):
                    forecasted_trajectories[seq_id] = [pred_y_k for pred_y_k in pred_y[batch_id]]
                    gt_trajectories[seq_id] = gt[batch_id]
                    seq_id += 1

            metric_results = get_displacement_errors_and_miss_rate(
                forecasted_trajectories,
                gt_trajectories,
                k,
                horizon,
                miss_threshold
            )
        return metric_results
