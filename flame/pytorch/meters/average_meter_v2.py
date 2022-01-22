from collections import defaultdict
from contextlib import contextmanager
from typing import DefaultDict, Dict, List, Optional
from torch import Tensor
import torch
import torch.distributed as dist
import logging

_logger = logging.getLogger(__name__)


_DEFAULT_DEVICE = torch.device("cpu")


class AverageMeter:
    def __init__(
        self,
        name: str,
        fmt: str = ":f",
        device: torch.device = _DEFAULT_DEVICE,
    ) -> None:

        self.name = name
        self.fmt = fmt
        self.device = device

        # Prevent syncing multiple times
        self._synchronized = True
        self.val = torch.tensor(0, dtype=torch.float, device=device)
        self.sum = torch.tensor(0, dtype=torch.float, device=device)
        self.count = torch.tensor(0, dtype=torch.int, device=device)

    def reset(self):
        self.val = torch.zeros_like(self.val)
        self.sum = torch.zeros_like(self.sum)
        self.count = torch.zeros_like(self.count)

    @torch.inference_mode()
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

        fut_count = dist.all_reduce(self.count, op=dist.ReduceOp.SUM, async_op=True)
        fut_sum = dist.all_reduce(self.sum, op=dist.ReduceOp.SUM, async_op=True)
        fut_count.wait()
        fut_sum.wait()

        self._synchronized = True


class LazyAverageMeterDict:
    def __init__(
        self,
        delimiter: str = "\t",
        separator: str = "/",
        device: torch.device = _DEFAULT_DEVICE,
    ) -> None:
        self._history: DefaultDict[str, List[float]] = defaultdict(list)
        # Don't know if flatten dict is a good choice
        self._meters: Dict[str, AverageMeter] = {}
        self._delimiter = delimiter
        self._separator = separator
        self._device = device

    def update(self, prefix: str, name: str, val: Tensor, n: int = 1, fmt: str = ":f"):
        key = prefix + self._separator + name

        if key not in self._meters:
            self._meters[key] = AverageMeter(name, fmt=fmt, device=self._device)

        self._meters[key].update(val, n=n)

    def sync(self):
        for key, meter in self._meters.items():
            meter.sync()

    def reset(self):
        for key, meter in self._meters.items():
            meter.reset()

    def save_history(self, prefix: str):
        for key, meter in self.named_meters(prefix=prefix):
            self._history[key].append(meter.avg.item())

    def named_meters(self, prefix: Optional[str] = None):
        if prefix:
            for key, meter in self._meters.items():
                if key.startswith(prefix):
                    yield key, meter
        else:
            for key, meter in self._meters.items():
                yield key, meter

    @contextmanager
    def record(self, prefix: str):
        self.reset()
        yield self
        self.sync()
        self.save_history(prefix)
