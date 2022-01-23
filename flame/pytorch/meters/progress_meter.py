

from sys import prefix
from typing import Optional

from torch import Tensor
from flame.core.meters.naive_average_meter import NaiveAverageMeter
from tqdm import tqdm
from flame.pytorch.distributed import get_rank_safe
from flame.core.helpers.tqdm import tqdm_get_rate
import logging


_logger = logging.getLogger(__name__)


class ProgressMeter:

    def __init__(self, epoch: int, meters, prefix: str, print_freq: int = 10, no_tqdm: bool = False) -> None:
        self._epoch = epoch
        self._prefix = prefix
        self._batch_size = 1
        self._no_tqdm = no_tqdm
        self._print_freq = print_freq

        self.sample_per_second_meter = NaiveAverageMeter('spl/s', fmt=':.2f')

    def enumerate(self, iterable, num_iters: Optional[int] = None):
        if not num_iters:
            num_iters = len(iterable)

        with tqdm(
            desc=self._prefix,
            total=num_iters,
            dynamic_ncols=True,
            ascii=True,
            disable=self._no_tqdm or get_rank_safe() != 0  # 如果不是 rank0，就关掉 tqdm
        ) as pbar:
            for batch_idx, batch in enumerate(iterable, start=1):
                yield batch_idx, batch

                pbar.update()

                num_samples = self._batch_size * tqdm_get_rate(pbar)
                self.sample_per_second_meter.update(
                    num_samples
                )

            _logger.info(
                f'{prefix} complete [{self._epoch}]: {self.sample_per_second_meter}'
            )

    def update(self, name: str, val: Tensor, n: int = 1):
        pass

    def update_batch_size(self, batch_size: int):
        self._batch_size = batch_size

