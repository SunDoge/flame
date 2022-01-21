from optparse import Option
from typing import Optional
from torch import Tensor
import torch
import torch.distributed as dist
import logging

_logger = logging.getLogger(__name__)


class AverageMeter:

    def __init__(
        self,
        name: str,
        fmt: str = ':f',
        device: Optional[torch.device] = None,
    ) -> None:

        self.name = name
        self.fmt = fmt

        self._synchronized = True
        self.val = torch.tensor(0, dtype=torch.float, device=device)
        self.sum = torch.tensor(0, dtype=torch.float,  device=device)
        self.count = torch.tensor(0, dtype=torch.int, device=device)

    def reset(self):
        self.val = torch.zeros_like(self.val)
        self.sum = torch.zeros_like(self.sum)
        self.count = torch.zeros_like(self.count)

    @torch.no_grad()
    def update(self, val: Tensor, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n

        self._synchronized = False

    @property
    def avg(self) -> Tensor:
        return self.sum / self.count

    def __str__(self) -> str:
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(
            name=self.name,
            val=self.val.item(),
            avg=self.avg.item(),
        )

    def sync(self):

        if self._synchronized:
            _logger.debug(f'meter {self.name} is already synchronized')
            return

        fut_count = dist.all_reduce(
            self.count, op=dist.ReduceOp.SUM, async_op=True)
        fut_sum = dist.all_reduce(
            self.sum, op=dist.ReduceOp.SUM, async_op=True)
        fut_count.wait()
        fut_sum.wait()

        self._synchronized = True


class AverageMeterDict:
    pass
