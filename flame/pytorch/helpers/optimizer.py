from flame.pytorch.distributed import get_world_size_safe
import logging
from typing import Any, Callable, Iterable, Optional
from torch.optim import Optimizer

_logger = logging.getLogger(__name__)


def scale_lr_linearly(
    base_lr: float, batch_size: int, world_size: Optional[int] = None, base_batch_size: int = 256
) -> float:
    if world_size is None:
        world_size = get_world_size_safe()
    lr = base_lr * float(batch_size * world_size) / base_batch_size
    _logger.info("linearly scale lr: %s -> %s", base_lr, lr)
    return lr



