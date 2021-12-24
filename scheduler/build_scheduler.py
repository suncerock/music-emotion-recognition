"""
Learning rate scheduler
Refer from https://github.com/rwightman/pytorch-image-models/blob/master/timm/scheduler/scheduler.py
"""

import warnings

from .step_lr import StepLRScheduler
from .cosine_lr imoprt CosineLRScheduler


def build_scheduler(config, optimizer):
    if config.type is None:
        return None
    elif config.type == 'step':
        scheduler = StepLRScheduler(
            optimizer,
            decay_t=getattr(config, "decay_t"),
            decay_rate=getattr(config, "decay_rate", 0.1),
            warmup_steps=getattr(config, "warmup_steps", 0),
            warmup_lr_init=getattr(config, "warmup_lr_init", 0),
            warmup_prefix=getattr(config, "warmup_prefix", False),
            t_in_epochs=getattr(config, "t_in_epochs", True)
        )
    elif config.type == 'cosine':
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=getattr(config, "t_initial"),
            lr_min=getattr(config, "lr_min", 0.),
            cycle_mul=getattr(config, "cycle_mul", 1.),
            cycle_decay=getattr(config, "cycle_decay", 1.),
            cycle_limit=getattr(config, "cycle_limit", 1),
            warmup_steps=getattr(config, "warmup_steps", 0),
            warmup_lr_init=getattr(config, "warmup_lr_init", 0.),
            warmup_prefix=getattr(config, "warmup_prefix", False),
            t_in_epochs=getattr(config, "t_in_epochs", True)
        )
    else:
        warnings.warn("Unknown scheduler type {}".format(config.type))
        return None
    
    return scheduler



