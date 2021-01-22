from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import DataLoader

from core.trainer import Trainer
from core.model.vectornet import VectorNet, OriginalVectorNet
from core.optim_schedule import ScheduledOptim
from core.loss import VectorLoss


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
                 lr: float = 1e-3,
                 betas=(0.9, 0.999),
                 weight_decay: float = 0.01,
                 warmup_epoch=15,
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
            8,
            30,
            num_global_graph_layer=num_global_graph_layer,
            with_aux=aux_loss,
            device=self.device
        )

        if not model_path:
            if self.multi_gpu:
                self.model = nn.DataParallel(self.model)
                if self.verbose:
                    print("[VectorNetTrainer]: Train the mode with multiple GPUs.")
            else:
                print("[VectorNetTrainer]: Train the mode with single device on {}.".format(self.device))
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
            if training:
                pred, aux_out, aux_gt = self.model(data.to(self.device))
                loss = self.criterion(
                    pred,
                    data.y.view(-1, self.model.out_channels * self.model.pred_len),
                    aux_out,
                    aux_gt
                )

                self.optm_schedule.zero_grad()
                loss.backward()
                self.optim.step()
                self.write_log("Train Loss", loss.item(), i + epoch * len(dataloader))

            else:
                with torch.no_grad():
                    pred = self.model(data.to(self.device))
                    loss = self.criterion(pred,
                                          data.y.view(-1, self.model.out_channels * self.model.pred_len))
                    self.write_log("Eval Loss", loss.item(), i + epoch * len(dataloader))

            num_sample += self.batch_size
            avg_loss += loss.item()

            # print log info
            # log = {
            #     "iter": i,
            #     "loss": loss.item(),
            #     "avg_loss": avg_loss / num_sample
            # }
            # data_iter.write(str(log))
            desc_str = "[Info: {}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}]".format("train" if training else "eval",
                                                                                 epoch,
                                                                                 loss.item(),
                                                                                 avg_loss / num_sample)
            data_iter.set_description(desc=desc_str, refresh=True)

        self.optm_schedule.step_and_update_lr()
        return avg_loss / num_sample

    # todo: the inference of the model
    def test(self, data):
        raise NotImplementedError
