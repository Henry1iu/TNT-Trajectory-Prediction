# A wrapper class for optimizer
# source: https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/trainer/optim_schedule.py
import numpy as np


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling
    """

    def __init__(self, optimizer, init_lr, n_warmup_epoch=5, decay_rate=0.3):
        self._optimizer = optimizer
        self.n_warmup_epoch = n_warmup_epoch
        self.n_current_steps = 0
        self.init_lr = init_lr
        self.decay_rate = decay_rate

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.power(self.decay_rate, self.n_current_steps)

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
