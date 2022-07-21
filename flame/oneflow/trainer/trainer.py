import logging
from datetime import datetime
from typing import TypeVar

import oneflow as flow
import oneflow.env
from oneflow import Tensor
from oneflow.utils.data.dataloader import DataLoader
from oneflow.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# from flame.core.helpers.tqdm import tqdm_get_total_time
from flame.oneflow.arguments import BaseArgs
from flame.oneflow.meters.average_meter import LazyAverageMeterDict

from .progress_meter import ProgressMeter
from .state import State
from .state_manager import StateManager

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

    def to_device(self, x: T) -> T:
        return _to_device(x, self.device)

    def run(self, max_epochs: int):
        pass

    def train(self, loader, prefix: str = "train"):
        pass

    def validate(self, loader, prefix: str = "val"):
        pass

    def test(self, loader, prefix: str = "test"):
        return self.validate(loader, prefix=prefix)

    def _disable_tqdm(self) -> bool:
        return self.args.no_tqdm or oneflow.env.get_rank() != 0

    def epoch_range(self, max_epochs: int):

        start_time = datetime.now()

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

            # _logger.info('Total time: %s', tqdm_get_total_time(pbar))
        duration = datetime.now() - start_time
        _logger.info('Total time: %s', duration)

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


def _to_device(x: T, device: flow.device) -> T:
    if isinstance(x, tuple):
        return tuple(_to_device(v, device) for v in x)
    elif isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        return [_to_device(v, device) for v in x]
    elif isinstance(x, Tensor):
        return x.to(device)
    else:
        raise Exception()
