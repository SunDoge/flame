import logging
from turtle import position
from typing import TypeVar

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from flame.pytorch.arguments import BaseArgs
from flame.pytorch.distributed import get_rank_safe
from flame.pytorch.meters.average_meter_v2 import LazyAverageMeterDict
from flame.pytorch.trainer.state_manager import StateManager
from .progress_meter import ProgressMeter
from .state import State
from .trainer import _to_device

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
            position=1,
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

    def progress_meter(self, prefix: str) -> ProgressMeter:
        return ProgressMeter(
            self.meters,
            self.state,
            prefix,
            self.device,
            print_freq=self.args.print_freq,
            no_tqdm=self.args.no_tqdm,
            debug=self.args.debug,
        )

    def set_sampler_epoch(self, sampler: DistributedSampler):
        sampler.set_epoch(self.state.epoch)

    @property
    def name(self) -> str:
        return self.__class__.__qualname__
