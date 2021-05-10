

from dataclasses import dataclass
import dataclasses

from pydantic.schema import model_schema


from flame.pytorch.meters.average_meter import AverageMeterGroup

from typing import Any, Callable, Dict, Iterable, NamedTuple, Optional, Tuple
import logging
from enum import Enum

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import functools
import torch
from torch import nn, Tensor
from flame.pytorch.typing_prelude import Optimizer, LrScheduler, Model
from pydantic import BaseModel

_logger = logging.getLogger(__name__)


@dataclass
class State:
    step: int = 0
    epoch: int = 0
    training: bool = False
    mode: str = 'train'
    epoch_length: Optional[int] = None

    def reset(self):
        for field in self.__dataclass_fields__.values():
            field: dataclasses.Field
            assert field.default is not dataclasses.MISSING
            setattr(self, field.name, field.default)


class Mode(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class BaseEngineConfig(BaseModel):
    update_interval: int = 1  # optimizer更新频率，用来控制梯度lei ji
    log_interval: int = 10
    max_norm: float = -1.  # 如果大于0，就开始clip


class BaseEngine:

    state: State
    model: Model
    optimizer: Optimizer
    scheduler: LrScheduler
    cfg: BaseEngineConfig
    meters: AverageMeterGroup

    def __init__(self, **kwargs):
        """
        example::
            self._init(**locals())
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    # def _iteration(self, i: int, batch, mode: str):
    #     pass

    def _loop(self, data_loader: Iterable, step_fn: Callable[[Any, int], None]):
        for batch_idx, batch in enumerate(data_loader, start=1):
            if self.state.training:
                self.state.step += 1

            step_fn(batch, batch_idx)

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

    def forward(self, batch, batch_idx: int) -> dict:
        pass

    def update(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def training_step(self, batch, batch_idx: int) -> dict:
        output = self.forward(batch, batch_idx)
        loss: Tensor = output['loss']
        loss.backward()

        self.clip_grad_norm_if_needed()

        if self.every_n_steps(n=self.cfg.update_interval):
            self.update()

    def validation_step(self, batch, batch_idx: int) -> dict:
        output = self.forward(batch, batch_idx)
        return output

    def clip_grad_norm_if_needed(self):
        if self.cfg.max_norm > 0.0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.max_norm,
            )

    @staticmethod
    def output(loss: Tensor = None, batch_size: int = None, **kwargs) -> dict:
        output = {
            'loss': loss,
            'batch_size': batch_size,
            **kwargs
        }
        return output

    def train(self, loader: Iterable, epoch_length: Optional[int] = None, mode: str = 'train'):
        self.state.training = True
        self.state.epoch += 1
        self.state.mode = mode

        if epoch_length is None:
            self.state.epoch_length = self._auto_infer_epoch_length(
                loader
            )

        self._auto_set_epoch(loader, self.state.epoch)

        self.training_loop(
            functools.partial(self._loop, loader)
        )

    def validate(self, loader: Iterable, epoch_length: Optional[int] = None, mode: str = 'val'):
        self.state.training = False
        self.state.mode = mode

        if epoch_length is None:
            self.state.epoch_length = self._auto_infer_epoch_length(
                loader
            )

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
