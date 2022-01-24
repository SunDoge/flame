

import logging
from typing import Optional

import torch
from torch import Tensor
from tqdm import tqdm

from flame.core.helpers.tqdm import tqdm_get_rate
from flame.core.meters.naive_average_meter import NaiveAverageMeter
from flame.pytorch.distributed import get_rank_safe
from flame.pytorch.meters.average_meter_v2 import (AverageMeter,
                                                   LazyAverageMeterDict)

from torch.utils.tensorboard.writer import SummaryWriter

from .state import State


_logger = logging.getLogger(__name__)


class ProgressMeter:

    def __init__(
        self,
        meters: LazyAverageMeterDict,
        state: State,
        prefix: str,
        device: torch.device,
        print_freq: int = 10,
        no_tqdm: bool = False,
        separator: str = '/',
        debug: bool = False,
        num_valid_samples: int = 0,
    ) -> None:
        self._state = state
        self._prefix = prefix
        self._batch_size = 1
        self._no_tqdm = no_tqdm
        self._print_freq = print_freq
        self._device = device
        self._separator = separator
        self._meters = meters
        self._debug = debug

        self._num_remaining_samples = num_valid_samples

        self.sample_per_second_meter = NaiveAverageMeter('spl/s', fmt=':.2f')

    def enumerate(self, iterable, num_iters: Optional[int] = None):
        if not num_iters:
            num_iters = len(iterable)

        self.sample_per_second_meter.reset()
        epoch = self._state.epoch
        prefix = self._prefix

        with tqdm(
            desc=self._prefix.capitalize(),
            total=num_iters,
            dynamic_ncols=True,
            ascii=True,
            position=0,
            disable=self._no_tqdm or get_rank_safe() != 0  # 如果不是 rank0，就关掉 tqdm
        ) as pbar, self._meters.record(epoch=self._state.epoch, prefix=self._prefix):
            for batch_idx, batch in enumerate(iterable, start=1):

                if self._state.training:
                    self._state.step += 1

                yield batch_idx, batch

                pbar.update()

                num_samples = self._batch_size * tqdm_get_rate(pbar)
                self.sample_per_second_meter.update(
                    num_samples
                )

                if batch_idx % self._print_freq == 0:
                    meter_str = self._meters.to_str(prefix=self._prefix)
                    _logger.info(
                        f'{self._prefix} [{epoch}][{batch_idx}/{num_iters}]\t{self.sample_per_second_meter}\t{meter_str}'
                    )
                    if self._debug:
                        break

        # Wait for meters sync
        meter_str = self._meters.to_str(prefix=self._prefix)
        _logger.info(
            f'{prefix} complete [{epoch}]\t{self.sample_per_second_meter}\t{meter_str}'
        )

    def update_batch_size(self, batch_size: int) -> int:
        """
        Return:
            num valid samples in this batch
        """
        self._batch_size = batch_size
        self._num_remaining_samples -= batch_size

        if self._num_remaining_samples > 0:
            return min(self._num_remaining_samples, batch_size)
        else:
            _logger.info('no remaining samples left')
            return 0

    def get(self, name: str, fmt: str = ':f') -> AverageMeter:
        key = self._prefix + self._separator + name
        return self._meters.get(name, key=key, fmt=fmt)

    def write_summary(self, summary_writer: SummaryWriter, trainer_name: Optional[str] = None):
        if trainer_name is None:
            trainer_prefix = ''
        else:
            trainer_prefix = trainer_name + '/'

        for key, meter in self._meters.named_meters(prefix=self._prefix):
            tag = trainer_prefix + key
            _logger.info('summary writer add scalar: %s', tag)
            summary_writer.add_scalar(
                tag, meter.avg.item(), global_step=self._state.epoch
            )
