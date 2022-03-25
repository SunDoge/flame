from typing import Any, Callable
from oneflow import Tensor
from oneflow.optim import Optimizer
from oneflow.optim.lr_scheduler import _LRScheduler as LrScheduler
from oneflow.utils.data.dataloader import DataLoader


OptimizerFn = Callable[[Any, float], Optimizer]
LrSchedulerFn = Callable[[Optimizer], LrScheduler]


__all__ = [
    'Tensor',
    'Optimizer',
    'LrScheduler',
    'DataLoader',
    'OptimizerFn',
    'LrSchedulerFn',
]
