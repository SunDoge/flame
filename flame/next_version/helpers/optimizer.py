from flame.next_version.config_parser import ConfigParser
import torch.distributed as dist
import logging

_logger = logging.getLogger(__name__)


def create_optimizer_from_config(config: dict, params):
    config_parser = ConfigParser()
    config['params'] = params

    optimizer = config_parser.parse(config)
    return optimizer


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
