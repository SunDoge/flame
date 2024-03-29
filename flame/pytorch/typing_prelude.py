from typing import Any, Callable
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LrScheduler
from torch.utils.data.dataloader import DataLoader


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
