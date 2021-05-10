

from dataclasses import dataclass
import dataclasses

from typing import Any, Callable, Dict, Iterable, NamedTuple, Optional, Tuple
import logging
from enum import Enum

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import functools
import torch
from torch import nn
from flame.pytorch.typing_prelude import Optimizer, LrScheduler, Model


_logger = logging.getLogger(__name__)


@dataclass
class State:
    step: int = 0
    epoch: int = 0
    training: bool = False

    def reset(self):
        for field in self.__dataclass_fields__.values():
            field: dataclasses.Field
            assert field.default is not dataclasses.MISSING
            setattr(self, field.name, field.default)


class Mode(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class BaseEngine:

    state: State
    model: Model
    optimizer: Optimizer
    scheduler: LrScheduler

    def _init(self, **kwargs):
        """
        example::
            self._init(**locals())
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _iteration(self, i: int, batch, mode: str):
        pass

    def _loop(self, data_laoder: Iterable):
        for i, batch in enumerate(data_laoder, start=1):
            if self.state.training:
                self.state.step += 1

            self._iteration(batch)

        _logger.debug('epoch %s finished', self.state.epoch)

    def prepare_data(self, batch: Any) -> Tuple[Any, int]:
        """
        Return:
            data: data for training
            batch_size: batch size
        """
        pass

    def training_loop(self, next: Callable):
        # model.train()

        next()

        # log something

    def validation_loop(self, next: Callable):

        next()

    def forward(self, batch) -> dict:
        pass

    def training_step(self, batch) -> dict:
        pass

    def validation_step(self, batch) -> dict:
        pass

    def train(self, loader: Iterable, epoch_length: Optional[int] = None, mode: str = 'train'):
        self.state.training = True
        self.state.epoch += 1

        self._auto_set_epoch(loader, self.state.epoch)

        self.training_loop(
            functools.partial(self._loop, loader)
        )

    def validate(self, loader: Iterable, epoch_length: Optional[int] = None, mode: str = 'val'):
        self.state.training = False
        with torch.no_grad():
            self.validation_loop(
                functools.partial(self._loop, loader)
            )

    @staticmethod
    def _auto_set_epoch(loader: DataLoader, epoch: int):
        if isinstance(loader, DataLoader) and isinstance(loader.sampler, DistributedSampler):
            _logger.debug('automatic loader.sampler.set_epoch(%d)', epoch)
            loader.sampler.set_epoch(epoch)

    @staticmethod
    def _auto_infer_epoch_length(loader: DataLoader):
        if hasattr(loader, '__len__'):
            return len(loader)

    def exit(self):
        pass

    @staticmethod
    def every(i: int, n: int) -> bool:
        return i > 0 and i % n == 0

    def every_n_steps(self, n: int = 1) -> bool:
        return self.every(self.state.step, n)

    def unfinished(self, max_epochs: int) -> bool:
        return self.state.epoch < max_epochs
