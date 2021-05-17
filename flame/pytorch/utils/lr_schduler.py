from torch.optim.lr_scheduler import _LRScheduler as BaseLrScheduler
from typing import Optional
import enum


class LrScheduler(BaseLrScheduler):
    """
    用来占位，没有实际作用
    """

    def __init__(self) -> None:
        pass

    def step(self, epoch: Optional[int] = None) -> None:
        pass


class SchedulePosition(enum.Enum):
    STEP = enum.auto()
    EPOCH = enum.auto()


class LrSchedulerManager:

    def __init__(self) -> None:
        pass
