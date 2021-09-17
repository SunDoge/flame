from typing import Iterable
from flame.pytorch.meters.time_meter import EstimatedTimeOfArrival
from torch import nn
import logging

_logger = logging.getLogger(__name__)


class BaseState:

    def __init__(self) -> None:
        self.epoch = 0
        self.step = 0
        self.batch_idx = 0
        self.epoch_length = -1
        self.training = False
        self.max_epochs = 1
        self.epoch_eta = EstimatedTimeOfArrival(0)
        self.iter_eta = EstimatedTimeOfArrival(0)

    def train(self, mode=True):
        self.training = mode
        for k, v in self.__dict__.items():
            if isinstance(v, nn.Module):
                _logger.info(f'set {k}.train(mode={mode})')
                v.train(mode=mode)

    def eval(self):
        self.train(mode=False)

    def epoch_wrapper(self, max_epochs: int):
        self.max_epochs = max_epochs
        self.epoch_eta = EstimatedTimeOfArrival(max_epochs, initial=self.epoch)
        while self.epoch < self.max_epochs:
            self.epoch += 1
            yield self.epoch
            self.epoch_eta.update()

    def iter_wrapper(self, loader: Iterable):
        self.epoch_length = self.get_length(loader)
        self.iter_eta = EstimatedTimeOfArrival(self.epoch_length)
        loader_iter = iter(loader)
        for batch_idx in range(self.epoch_length):
            if self.training:
                self.step += 1
            self.batch_idx = batch_idx
            batch = next(loader_iter)
            batch_size = self.get_batch_size(batch)
            yield batch, batch_size
            self.iter_eta.update(n=batch_size)

    def get_length(self, loader: Iterable) -> int:
        return len(loader)

    def get_batch_size(self, batch) -> int:
        return 1

    def every_n_iters(self, n: int = 1) -> bool:
        return self.batch_idx > 0 and self.batch_idx % n == 0
