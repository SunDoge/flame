
from numbers import Number
from typing import Any, Dict
from .base_meter import BaseMeter
from flame.pytorch.utils.distributed import is_dist_available_and_initialized, reduce_numbers
import logging

_logger = logging.getLogger(__name__)


class AverageMeter(BaseMeter):

    def __init__(self, name: str, fmt: str = ':.f') -> None:
        super().__init__()

        self.name = name
        self.fmt = fmt

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

    def update(self, val: Number, n: int = 1):
        # 每次更新local计数器，更新完后需要sync
        self.val = val
        self._local_count += n
        self._local_sum += val * n
        self.synchronized = False

    def sync(self):
        if self.synchronized:
            return

        if is_dist_available_and_initialized():
            global_sum, global_count = reduce_numbers([
                self._local_sum, self._local_count
            ])
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
        fmt_str = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmt_str.format(name=self.name, fmt=self.fmt)


class AverageMeterGroup(BaseMeter):

    def __init__(self, meters: Dict[str, AverageMeter], delimiter: str = "\t") -> None:
        super().__init__()
        self.meters = meters
        self.delimiter = delimiter

    def update(self, output: Dict[str, Any], n: int = 1):
        for key, meter in self.meters.items():
            value = output[key]
            meter.update(value, n=n)

    def sync(self):
        for meter in self.meters.values():
            meter.sync()

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def __str__(self) -> str:
        fmt_str = self.delimiter.join([str(m) for m in self.meters.values()])
        return fmt_str
