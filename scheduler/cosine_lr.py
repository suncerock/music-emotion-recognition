import math
import torch

from .scheduler import Scheduler


class CosineLRScheduler(Scheduler):
    def __init__(self,
                 optimizer,
                 t_initial,
                 lr_min=0.,
                 cycle_mul=1.,
                 cycle_decay=1.,
                 cycle_limit=1.,
                 warmup_steps=0,
                 warmup_lr_init=0.,
                 warmup_prefix=False,
                 t_in_epochs=True):
        super(CosineLRScheduler, self).__init__(optimizer)

        self.t_initial = t_initial
        self.lr_min = lr_min

        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit

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
            if self.cycle_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul))
                t_i = self.cycle_mul ** i * self.t_initial
                t_curr = t - (1 - self.cycle_mul ** i) / (1 - self.cycle_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.cycle_decay ** i
            lr_max_values = [v * gamma for v in self.base_values]

            if i < self.cycle_limit:
                lrs = [
                    self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.cos(math.pi * t_curr / t_i))
                     for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

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
