import functools
from typing import Callable, Optional
from flame.config_parser import ConfigParser
import torch.distributed as dist
import logging
from torch.optim import Optimizer

_logger = logging.getLogger(__name__)


def _scale_lr(lr: float) -> float:
    world_size = _get_world_size()
    new_lr = lr * world_size
    return new_lr


def create_optimizer_from_config(config: dict, params, scale_lr: Callable[[float], float] = _scale_lr):
    config_parser = ConfigParser()

    config_copied = config.copy()
    old_lr = config_copied['lr']
    new_lr = scale_lr(old_lr)
    _logger.info('scale lr: %s -> %s', old_lr, new_lr)
    config_copied['params'] = params

    optimizer = config_parser.parse(config_copied)
    return optimizer


def get_learning_rate_from_optimizer(optimizer: Optimizer) -> float:
    lr = optimizer.param_groups[0]['lr']
    return lr


def _get_world_size() -> int:
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()

    return world_size


def scale_lr_linearly(
    lr: float,
    batch_size: int,
    world_size: int = 1,
    base: int = 256
) -> float:
    new_lr = lr * float(batch_size * world_size) / base
    _logger.info('linearly scale lr: %s -> %s', lr, new_lr)
    return new_lr


def scale_lr_for_partial(
    func: functools.partial,
    world_size: Optional[int] = None
):
    lr: float = func.keywords['lr']
    if not world_size:
        world_size = _get_world_size()
        _logger.info('auto infer world_size: %s', world_size)

    new_lr = lr * world_size
    func.keywords['lr'] = new_lr
