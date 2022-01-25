import logging
from typing import TypeVar

import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from flame.core.helpers.tqdm import tqdm_get_total_time
from flame.pytorch.arguments import BaseArgs
from flame.pytorch.distributed import get_rank_safe
from flame.pytorch.meters.average_meter import LazyAverageMeterDict
from flame.pytorch.trainer.state_manager import StateManager

from .progress_meter import ProgressMeter
from .state import State

_logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self, args: BaseArgs) -> None:

        state = State()
        meters = LazyAverageMeterDict(device=args.device)

        state_manager = StateManager()
        state_manager.register(
            'state',
            state.state_dict,
            state.load_state_dict,
            state.train
        )
        state_manager.register(
            'meters',
            meters.state_dict,
            meters.load_state_dict
        )

        self.args = args
        self.device = args.device
        self.debug = args.debug
        self.state = state
        self.state_manager = state_manager
        self.meters = meters

    T = TypeVar('T')

    def to_device(self, x: T, non_blocking: bool = True) -> T:
        return _to_device(x, self.device, non_blocking=non_blocking)

    def run(self, max_epochs: int):
        pass

    def train(self, loader, prefix: str = "train"):
        pass

    def validate(self, loader, prefix: str = "val"):
        pass

    def test(self, loader, prefix: str = "test"):
        return self.validate(loader, prefix=prefix)

    def _disable_tqdm(self) -> bool:
        return self.args.no_tqdm or get_rank_safe() != 0

    def epoch_range(self, max_epochs: int):

        with tqdm(
            desc='Epoch',
            total=max_epochs,
            initial=self.state.epoch,
            dynamic_ncols=True,
            ascii=True,
            position=-1,
            disable=self._disable_tqdm()
        ) as pbar:

            # init epoch = 0
            while self.state.epoch < max_epochs:
                self.state.epoch += 1

                # 1-based epoch
                yield self.state.epoch

                pbar.update()

                _logger.info(
                    f'Epoch complete [{self.state.epoch}/{max_epochs}]'
                )

                if self.debug:
                    break

            _logger.info('Total time: %s', tqdm_get_total_time(pbar))

    def progress_meter(self, prefix: str) -> ProgressMeter:
        return ProgressMeter(
            self.meters,
            self.state,
            prefix,
            self.device,
            print_freq=self.args.print_freq,
            no_tqdm=self.args.no_tqdm,
            debug=self.args.debug,
            separator='/',
        )

    def set_sampler_epoch(self, loader: DataLoader):
        sampler = loader.sampler
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(self.state.epoch)
            _logger.info(
                'loader.sampler.set_epoch(%s)', self.state.epoch
            )
        else:
            _logger.warning(
                'loader.sampler is not DistributedSampler, fail to set epoch'
            )

    @property
    def name(self) -> str:
        return self.__class__.__qualname__

    @property
    def module_name(self) -> str:
        """
        lib.xxx_trainer -> xxx_trainer
        """
        return self.__module__.split('.')[-1]


T = TypeVar('T')


def _to_device(x: T, device: torch.device, non_blocking: bool = True) -> T:
    if isinstance(x, tuple):
        return tuple(_to_device(v, device, non_blocking=non_blocking) for v in x)
    elif isinstance(x, dict):
        return {k: _to_device(v, device, non_blocking=non_blocking) for k, v in x.items()}
    elif isinstance(x, list):
        return [_to_device(v, device, non_blocking=non_blocking) for v in x]
    elif isinstance(x, Tensor):
        return x.to(device, non_blocking=non_blocking)
    else:
        raise Exception()
