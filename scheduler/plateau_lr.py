"""
Learning rate scheduler
Refer from https://github.com/rwightman/pytorch-image-models/blob/master/timm/scheduler/scheduler.py
"""

import torch

from .scheduler import Scheduler


class PlateauLRScheduler(Scheduler):
    def __init__(self,
                 optimizer,
                 decay_rate=0.1,
                 patience_t=10,
                 warmup_lr_init=0.,
                 warmup_prefix=False,
                 t_in_epochs=True):
        super(StepLRScheduler, self).__init__(optimizer)
        self.decay_t = decay_t
        self.decay_rate = decay_rate

        self.warmup_steps = warmup_steps
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix

        self.t_in_epochs = t_in_epochs

        assert self.warmup_lr_init < max(self.base_values), "warm_lr_init must be smaller than lr!"

        if self.warmup_steps:
            self.warmup_step_size = [(v - warmup_lr_init) / self.warmup_steps for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_step_size = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_steps:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_step_size]
        else:
            t = t - self.warmup_steps if self.warmup_prefix else t
            lrs = [v * (self.decay_rate ** (t // self.decay_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates):
        if self.t_in_epochs:
            return None
        else:
            return self._get_lr(num_updates)
