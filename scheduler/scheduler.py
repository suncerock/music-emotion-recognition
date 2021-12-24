"""
Learning rate scheduler
Refer from https://github.com/rwightman/pytorch-image-models/blob/master/timm/scheduler/scheduler.py
"""

import torch


class Scheduler(object):
    """
    Parameter Scheduler Base Class
    A scheduler base class that can be used to schedule any optimizer parameter groups.

    Unlike the builtin PyTorch schedulers, this is intended to be consistently called
    * At the END of each epoch, before incrementing the epoch count, to calculate next epoch's value
    * At the END of each optimizer update, after incrementing the update count, to calculate next update's value
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_values = [group["lr"] for group in self.optimizer.param_groups]

    def get_epoch_values(self, epoch):
        raise NotImplementedError("Never use the base scheduler class")

    def get_update_values(self, num_updates):
        raise NotImplementedError("Never use the base scheduler class")

    def epoch_update(self, epoch):
        values = self.get_epoch_values(epoch)
        if values is not None:
            self.update_groups(values)

    def step_update(self, num_updates):
        values = self.get_update_values(num_updates)
        if values is not None:
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_groups, value in zip(self.optimizer.param_groups, values):
            param_groups["lr"] = value