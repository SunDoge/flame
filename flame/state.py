from typing import Callable, Dict, Iterable, Iterator

from torch.utils.data.dataloader import DataLoader
from flame.pytorch.meters.time_meter import EstimatedTimeOfArrival
from torch import nn
import logging
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel, DataParallel


_logger = logging.getLogger(__name__)

ToStateDictFunction = Callable[[], dict]
LoadStateDictFunction = Callable[[], dict]


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
        self.metrics = {}

        self._state_dict_functions = {}

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
        if self.training:
            self.set_epoch(loader)
        self.epoch_length = self.get_length(loader)
        self.iter_eta = EstimatedTimeOfArrival(self.epoch_length)
        loader_iter = iter(loader)
        for batch_idx in range(self.epoch_length):
            if self.training:
                self.step += 1

            # 1 base
            self.batch_idx = batch_idx + 1
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

    def set_epoch(self, loader: DataLoader):
        if isinstance(loader, DataLoader) and isinstance(loader.sampler, DistributedSampler):
            _logger.info('loader.sampler.set_epoch(%d)', self.epoch)
            loader.sampler.set_epoch(self.epoch)
        else:
            _logger.warning('fail to set_epoch for sampler')

    def state_dict(self):
        state_dict = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict'):
                # if isinstance(value, (DistributedDataParallel, DataParallel)):
                #     state_dict[key] = value.module.state_dict()
                # else:
                value: nn.Module
                state_dict[key] = value.state_dict()
            else:
                state_dict[key] = value

        return state_dict

    def load_state_dict(self, state_dict: dict):
        # FIXME: 这里没有检查state_dict是否有缺失state
        for key, value in state_dict.items():
            attribute = getattr(self, key)
            if hasattr(attribute, 'load_state_dict'):
                # if isinstance(attribute, (DistributedDataParallel, DataParallel)):
                #     attribute.module.load_state_dict(value)
                # else:
                attribute: nn.Module
                attribute.load_state_dict(value)
            else:
                setattr(self, key, value)

    def get_weights(self) -> Dict[str, dict]:
        weights = {}
        for key, value in self.__dict__.items():
            if isinstance(value, nn.Module):
                if isinstance(value, (DataParallel, DistributedDataParallel)):
                    weights[key] = value.module.state_dict()
                else:
                    weights[key] = value.state_dict()

        return weights

    def _get_training_state(self) -> Dict[str,  bool]:
        """
        检查是否model被设为train
        """
        res = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'training'):
                res[key] = getattr(value, 'training')

        return res
