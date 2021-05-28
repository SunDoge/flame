from flame.pytorch.typing_prelude import Model
from flame.pytorch.meters.time_meter import EstimatedTimeOfArrival
from typing import Callable, Iterable, Optional
from pydantic import BaseModel, Field
from torch import nn, Tensor
from torch.optim import Optimizer
import torch
from flame.pytorch.meters import Meter


class State(BaseModel):
    step: int = 0
    epoch: int = 0
    # epoch_length: Optional[int] = None
    # max_epochs: int = 0
    training: bool = True
    metrics: dict = {}


class BaseEngineConfig(BaseModel):
    update_freq: int = 1
    print_freq: int = 1
    max_epochs: int = 1


class BaseEngine:
    """
    继承这个类，然后实现两个方法
    forward
    run
    """

    model: Model
    optimizer: Optimizer
    state: State
    meters: Meter
    cfg: BaseEngineConfig

    def __init__(
        self,
        state: State,
        # cfg: dict
    ) -> None:

        # self.cfg = BaseEngineConfig(**cfg)
        self.state = state

        self.epoch_eta: EstimatedTimeOfArrival = None
        self.step_eta: EstimatedTimeOfArrival = None

    def training_step(self, batch, batch_idx: int):
        self.state.step += 1
        output = self.forward(batch, batch_idx)
        loss: Tensor = output['loss']
        loss.backward()

        if self.every_n_steps(self.cfg.update_freq):
            self.update(output)

    def validation_step(self, batch, batch_idx: int):
        output = self.forward(batch, batch_idx)
        return output

    def update(self, output: dict):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def loop(self, loader: Iterable, epoch_length: int, step_fn: Callable):

        self.step_eta = EstimatedTimeOfArrival(epoch_length)
        self.meters.reset()

        for batch_idx, batch in enumerate(loader, start=1):
            output = step_fn(batch, batch_idx)

    @staticmethod
    def output(loss: Tensor = None, batch_size: int = None, **kwargs) -> dict:
        output = {
            'loss': loss,
            'batch_size': batch_size,
            **kwargs
        }
        return output

    @staticmethod
    def every(i: int, n: int) -> bool:
        return i > 0 and i % n == 0

    def every_n_steps(self, n: int = 1) -> bool:
        return self.every(self.state.step, n)

    def unfinished(self) -> bool:
        return self.state.epoch < self.cfg.max_epochs

    @staticmethod
    def _auto_infer_epoch_length(loader: Iterable):
        if hasattr(loader, '__len__'):
            return len(loader)
        else:
            raise Exception('fail to infer epoch length')

    def train(self, loader: Iterable, epoch_length: Optional[int] = None, mode: str = 'train'):
        self.state.training = True
        self.state.epoch += 1

        if epoch_length is None:
            epoch_length = self._auto_infer_epoch_length(
                loader
            )

        self.model.train()

        self.loop(loader, epoch_length, self.training_step)

    def validate(self, loader: Iterable, epoch_length: Optional[int] = None, mode: str = 'val'):
        self.state.training = False

        if epoch_length is None:
            epoch_length = self._auto_infer_epoch_length(
                loader
            )

        self.model.eval()

        with torch.no_grad():
            self.loop(loader, epoch_length, self.validation_step)

    def run(self):
        # self.epoch_eta = EstimatedTimeOfArrival(self.state.max_epochs)
        pass

    def forward(self, batch, batch_idx: int) -> dict:
        pass
