from numbers import Number
from typing import Any, Dict, Optional
from .base_meter import Meter
from flame.pytorch.distributed import (
    is_dist_available_and_initialized,
    reduce_numbers,
)
import logging
from .time_meter import EstimatedTimeOfArrival
import numpy as np
from contextlib import AbstractContextManager
from torch.utils.tensorboard.writer import SummaryWriter

_logger = logging.getLogger(__name__)


class AverageMeter(Meter):
    def __init__(self, name: str, fmt: str = ":f") -> None:
        super().__init__()

        self.name = name
        self.fmt = fmt
        self.history = []

        self.reset()
        self.reset_local()

    def reset(self):
        self.val = 0  # val永远是local的
        self.sum = 0
        self.count = 0
        self.synchronized = True

    def reset_local(self):
        self._local_sum = 0
        self._local_count = 0
        self.synchronized = True

    def update(self, val: float, n: int = 1):
        # 每次更新local计数器，更新完后需要sync
        self.val = val
        self._local_count += n
        self._local_sum += val * n
        self.synchronized = False

    def sync(self):
        if self.synchronized:
            return

        if is_dist_available_and_initialized():
            global_sum, global_count = reduce_numbers(
                [self._local_sum, self._local_count]
            )
            self.sum += global_sum
            self.count += global_count
        else:
            self.sum += self._local_sum
            self.count += self._local_count

        self.reset_local()

    @property
    def avg(self) -> float:
        if self.synchronized:
            return self.sum / self.count
        else:
            return self._local_sum / self._local_count

    def __str__(self) -> str:
        fmt_str = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        # _logger.debug('fmt_str: %s', fmt_str)
        return fmt_str.format(name=self.name, val=self.val, avg=self.avg)

    def state_dict(self) -> dict:
        state = {
            "history": self.history,
        }
        return state

    def load_state_dict(self, state: dict):
        self.history = state["history"]

    def record(self):
        self.history.append(self.avg)

    def min(self):
        return min(self.history)

    def max(self):
        return max(self.history)

    def argmin(self):
        return np.argmin(self.history)

    def argmax(self):
        return np.argmax(self.history)

    def is_best(self, higher_is_better: bool = True) -> bool:
        last_metric = self.history[-1]
        best_metric = self.max() if higher_is_better else self.min()
        return last_metric is best_metric


class AverageMeterGroup(Meter):
    def __init__(self, meters: Dict[str, AverageMeter], delimiter: str = "\t") -> None:
        super().__init__()
        self._meters = meters
        self.delimiter = delimiter

    def update(self, metrics: Dict[str, Any], n: int = 1):
        for key, meter in self._meters.items():
            value = metrics[key]
            meter.update(value, n=n)

    def sync(self):
        for meter in self._meters.values():
            meter.sync()

    def reset(self):
        for meter in self._meters.values():
            meter.reset()

    def __str__(self) -> str:
        fmt_str = self.delimiter.join([str(m) for m in self._meters.values()])
        return fmt_str

    def __getitem__(self, key: str) -> AverageMeter:
        return self._meters[key]


class DynamicAverageMeterGroup(Meter):
    def __init__(self, delimiter: str = "\t") -> None:
        super().__init__()
        self.delimiter = delimiter
        self._meters: Dict[str, AverageMeter] = {}

    def update(self, name: str, value: float, n: int = 1, fmt: str = ":f"):
        if name not in self._meters:
            self._meters[name] = AverageMeter(name, fmt=fmt)

        meter = self._meters[name]
        meter.update(value, n=n)

    def sync(self):
        _logger.info(f"sync {self.__class__}")
        for meter in self._meters.values():
            meter.sync()

    def reset(self):
        _logger.info(f"reset {self.__class__}")
        for meter in self._meters.values():
            meter.reset()

    def __str__(self) -> str:
        fmt_str = self.delimiter.join([str(m) for m in self._meters.values()])
        return fmt_str

    def __getitem__(self, key: str) -> AverageMeter:
        return self._meters[key]

    def state_dict(self) -> dict:
        state = {k: v.state_dict() for k, v in self._meters.items()}
        return state

    def load_state_dict(self, state: dict):
        for k, v in state.items():
            self._meters[k].load_state_dict(v)

    def record(self):
        for v in self._meters.values():
            v.record()

    def write_summary(self, summary_writer: SummaryWriter, global_step: int):
        for name, meter in self._meters.items():
            summary_writer.add_scalar(name, meter.avg, global_step=global_step)

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, *args, **kwargs):
        # 先同步，再记录
        self.sync()
        self.record()


class LazyAverageMeters(DynamicAverageMeterGroup):
    """
    self._meters: Dict[prefix, DynamicAverageMeterGroup[name, AverageMeter]]
    """

    def __init__(self, delimiter: str = " ") -> None:
        super().__init__(delimiter=delimiter)
        self._meters: Dict[str, DynamicAverageMeterGroup] = {}

    def update(self, prefix: str, name: str, value: float, n: int = 1, fmt: str = ":f"):
        if prefix not in self._meters:
            self._meters[prefix] = DynamicAverageMeterGroup(delimiter=self.delimiter)

        meters = self._meters[prefix]
        meters.update(name, value, n=n, fmt=fmt)

    # def with_prefix(self, prefix: str):
    #     return self._meters.get(prefix)

    def __getitem__(self, key: str) -> DynamicAverageMeterGroup:
        if key not in self._meters:
            meters = DynamicAverageMeterGroup(delimiter=self.delimiter)
            self._meters[key] = meters
        return self._meters[key]

    def write_summary(
        self, prefix: str, summary_writer: SummaryWriter, global_step: int
    ):
        meters = self._meters[prefix]

        for name, meter in meters._meters.items():
            summary_writer.add_scalar(
                f"{prefix}/{name}", meter.avg, global_step=global_step
            )
