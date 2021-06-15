# A wrapper class for optimizer
# source: https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/trainer/optim_schedule.py
import numpy as np


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling
    """

    def __init__(self, optimizer, init_lr, n_warmup_epoch=10, update_rate=5, decay_rate=0.9):
        self._optimizer = optimizer
        self.n_warmup_epoch = n_warmup_epoch
        self.n_current_steps = 0
        self.init_lr = init_lr
        self.update_rate = update_rate
        self.decay_rate = decay_rate

    def step_and_update_lr(self):
        """Step with the inner optimizer"""
        self.n_current_steps += 1
        rate = self._update_learning_rate()
        return rate
        # self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.power(self.decay_rate, max((self.n_current_steps - self.n_warmup_epoch) // self.update_rate + 1, 0))

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """

        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        return lr
