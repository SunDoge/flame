

from dataclasses import dataclass
import dataclasses


from flame.pytorch.meters.average_meter import AverageMeterGroup

from typing import Any, Callable, Iterable, Optional, Tuple
import logging
from enum import Enum

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import functools
import torch
from torch import nn, Tensor
from flame.pytorch.typing_prelude import Criterion, Optimizer, LrScheduler, Model
from pydantic import BaseModel
import pydantic

_logger = logging.getLogger(__name__)

Middleware = Callable[[Callable], Any]


@dataclass
class State:
    step: int = 0
    epoch: int = 0
    training: bool = False
    mode: str = 'train'
    epoch_length: Optional[int] = None
    batch: Optional[Any] = None
    batch_idx: int = 0
    output: Optional[Any] = None
    loader: Optional[Iterable] = None

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
    type: str = pydantic.Field(alias='_type')
    update_interval: int = 1  # optimizer更新频率，用来控制梯度lei ji
    log_interval: int = 10
    max_norm: float = -1.  # 如果大于0，就开始clip


class BaseEngine:

    state: State
    model: Model
    optimizer: Optimizer
    scheduler: LrScheduler
    criterion: Criterion
    meters: AverageMeterGroup

    def __init__(
        self,
        model: Model = None,
        optimizer: Optimizer = None,
        criterion: Criterion = None,
        scheduler: Optional[LrScheduler] = None,
        update_interval: int = 1,
        log_interval: int = 1,
        max_norm: float = -1.,
    ):
        """
        example::
            self._init(**locals())
        """

        self.state = State()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.update_interval = update_interval
        self.log_interval = log_interval
        self.max_norm = max_norm
        # self.cfg = self.__annotations__['cfg'](**self.cfg)

    # def _iteration(self, i: int, batch, mode: str):
    #     pass

    def loop(self, next: Callable):
        # FIXME: 这里不应该用enumerate，应该根据epoch_length来循环，后面有空再改
        for batch_idx, batch in enumerate(self.state.loader, start=1):
            if self.state.training:
                self.state.step += 1

            self.state.batch_idx = batch_idx
            self.state.batch = batch

            next()

            self.state.batch = None
            self.state.output = None

        _logger.info('epoch %s finished', self.state.epoch)
        _logger.info(
            f'{self.state.mode} complete: {self.meters}'
        )

    def prepare_data(self, batch: Any) -> Tuple[Any, int]:
        """
        Return:
            data: data for training
            batch_size: batch size
        """
        pass

    def forward(self, next) -> dict:
        pass

    def update(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def training_step(self, next) -> dict:
        self.forward(next)
        loss: Tensor = self.state.output['loss']
        loss.backward()

        self.clip_grad_norm_if_needed()

        if self.every_n_steps(n=self.update_interval):
            self.update()

    def validation_step(self, next) -> dict:
        self.forward(next)

    def clip_grad_norm_if_needed(self):
        if self.max_norm > 0.0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_norm,
            )

    # @staticmethod
    def output(self, loss: Tensor = None, batch_size: int = None, **kwargs) -> dict:
        self.state.output = {
            'loss': loss,
            'batch_size': batch_size,
            **kwargs
        }

    def prepare_training(
        self,
        loader: Iterable,
        next: Callable,
        epoch_length: Optional[int] = None,
        mode: str = 'train'
    ):
        self.state.training = True
        self.state.epoch += 1
        self.state.mode = mode
        self.state.loader = loader
        self.model.train()

        if epoch_length is None:
            self.state.epoch_length = self._auto_infer_epoch_length(
                loader
            )

        self._auto_set_epoch(loader, self.state.epoch)

        # self.training_loop(
        #     functools.partial(self._loop, loader, self.training_step)
        # )

        next()

    def prepare_validation(
        self,
        loader: Iterable,
        next: Callable,
        epoch_length: Optional[int] = None,
        mode: str = 'val'
    ):
        self.state.training = False
        self.state.mode = mode
        self.state.loader = loader
        self.model.eval()

        if epoch_length is None:
            self.state.epoch_length = self._auto_infer_epoch_length(
                loader
            )

        with torch.no_grad():
            # self.validation_loop(
            #     functools.partial(self._loop, loader, self.validation_step)
            # )
            next()

    def train(self, loader: Iterable, epoch_length: Optional[int] = None, mode: str = 'train'):
        prepare_training = functools.partial(
            self.prepare_training, loader, epoch_length=epoch_length, mode=mode
        )

        fn = self.compose(
            prepare_training,
            self.loop,
            self.training_step
        )

        fn()

    def validate(self, loader: Iterable, epoch_length: Optional[int] = None, mode: str = 'val'):
        prepare_validation = functools.partial(
            self.prepare_validation, loader, epoch_length=epoch_length, mode=mode,
        )
        fn = self.compose(
            prepare_validation,
            self.loop,
            self.validation_step
        )

        fn()

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

    @staticmethod
    def compose(*middlewars: Middleware) -> Middleware:

        def fn(): pass

        for middleware in middlewars[::-1]:
            fn = functools.partial(middleware, fn)

        return fn
