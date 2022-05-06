import logging
from collections import defaultdict
from contextlib import contextmanager
from typing import DefaultDict, Dict, List, Optional, Tuple

import oneflow as flow
from oneflow import Tensor

_logger = logging.getLogger(__name__)


_DEFAULT_DEVICE = flow.device("cpu")
_HistoryRecord = Tuple[float, Optional[int]]


class AverageMeter:

    val: Tensor
    sum: Tensor
    count: Tensor

    def __init__(
        self,
        name: str,
        fmt: str = ":f",
        device: flow.device = _DEFAULT_DEVICE,
    ) -> None:

        self.name = name
        self.fmt = fmt
        self.device = device

        # Prevent syncing multiple times
        self._synchronized = True
        self.val = flow.tensor(0, dtype=flow.float, device=device)
        self.sum = flow.tensor(0, dtype=flow.float, device=device)
        self.count = flow.tensor(0, dtype=flow.int, device=device)

    def reset(self):
        self.val = flow.zeros_like(self.val)
        self.sum = flow.zeros_like(self.sum)
        self.count = flow.zeros_like(self.count)

    @flow.inference_mode()
    def update(self, val: Tensor, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n

        self._synchronized = False

    @property
    def avg(self) -> Tensor:
        return self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(
            name=self.name,
            val=self.val.item(),
            avg=self.avg.item(),
        )

    def sync(self):

        if self._synchronized:
            _logger.debug(f"meter {self.name} is already synchronized")
            return

        flow.comm.all_reduce(self.count)
        flow.comm.all_reduce(self.sum)

        self._synchronized = True


class LazyAverageMeterDict:
    def __init__(
        self,
        delimiter: str = "\t",
        device: flow.device = _DEFAULT_DEVICE,
    ) -> None:
        self._history: DefaultDict[str, List[_HistoryRecord]] = defaultdict(
            list
        )
        # Don't know if flatten dict is a good choice
        self._meters: Dict[str, AverageMeter] = {}
        self._delimiter = delimiter
        self._device = device

    def sync(self, prefix: Optional[str] = None):
        for key, meter in self.named_meters(prefix=prefix):
            _logger.info(f'sync {key}')
            meter.sync()

    def reset(self, prefix: Optional[str] = None):
        for key, meter in self.named_meters(prefix=prefix):
            meter.reset()

    def save_history(self, epoch: Optional[int] = None, prefix: Optional[str] = None):
        for key, meter in self.named_meters(prefix=prefix):
            self._history[key].append((meter.avg.item(), epoch))

    def named_meters(self, prefix: Optional[str] = None):
        if prefix:
            for key, meter in self._meters.items():
                if key.startswith(prefix):
                    yield key, meter
        else:
            for key, meter in self._meters.items():
                yield key, meter

    @contextmanager
    def record(self, epoch: Optional[int] = None, prefix: Optional[str] = None):
        self.reset(prefix=prefix)
        yield self
        self.sync(prefix=prefix)
        self.save_history(epoch=epoch, prefix=prefix)

    def get(self, name: str, key: Optional[str] = None, fmt: str = ':f') -> AverageMeter:
        if key is None:
            key = name

        if key in self._meters:
            return self._meters[key]
        else:
            meter = AverageMeter(name, fmt=fmt, device=self._device)
            self._meters[key] = meter
            return meter

    def state_dict(self) -> Dict:
        return {'history': self._history}

    def load_state_dict(self, state_dict: Dict):
        history = state_dict['history']

    def max(self, key: str) -> _HistoryRecord:
        return max(self._history[key], key=lambda x: x[0])

    def min(self, key: str) -> _HistoryRecord:
        return min(self._history[key], key=lambda x: x[0])

    def last(self, key: str) -> _HistoryRecord:
        return self._history[key][-1]

    def is_highest(self, key: str) -> bool:
        return self.last(key)[0] >= self.max(key)[0]

    def is_lowest(self, key: str) -> bool:
        return self.last(key)[0] <= self.min(key)[0]

    def __str__(self) -> str:
        fmt_str = self._delimiter.join([str(m) for m in self._meters.values()])
        return fmt_str

    def to_str(self, prefix: Optional[str] = None) -> str:
        fmt_str = self._delimiter.join(
            [str(m) for k, m in self.named_meters(prefix=prefix)])
        return fmt_str
