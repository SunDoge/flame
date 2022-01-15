import functools
import logging
from typing import Callable, Optional

from torch import nn
import torch
from torch.functional import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from flame.pytorch.meters.average_meter import LazyAverageMeters

from .data_module import DataModule

from .checkpoint_manager import CheckpointManager
from .coroutine_scheduler import CoroutineScheduler
from .state import State
from flame.pytorch.meters.time_meter import EstimatedTimeOfArrival
from torch.utils.data.distributed import DistributedSampler

_logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self) -> None:
        state = State()
        checkpoint_manager = CheckpointManager()
        meters = LazyAverageMeters()
        checkpoint_manager.register(
            "trainer_state", state.state_dict, state.load_state_dict, state.train
        )
        # checkpoint_manager.register(
        #     "meters",
        #     meters.state_dict,
        #     meters.load_state_dict,
        # )
        coroutine_scheduler = CoroutineScheduler()

        self.state = state
        self.checkpoint_manager = checkpoint_manager
        self.coroutine_scheduler = coroutine_scheduler

        self.iter_eta = None
        self.epoch_eta = None
        self.meters = meters
        self._break = False

    def train(self, loader, epoch_length: Optional[int] = None, prefix: str = "train"):

        # Update state
        self.state.epoch += 1

        # Switch to training mode
        self.checkpoint_manager.train()

        # Get epoch length
        if not epoch_length:
            self.state.epoch_length = self._try_infer_epoch_length(loader)

        # Set epoch
        if isinstance(loader, DataLoader) and isinstance(
            loader.sampler, DistributedSampler
        ):
            loader.sampler.set_epoch(self.state.epoch)
            _logger.info("set_epoch: %s", self.state.epoch)

        self.iter_eta = EstimatedTimeOfArrival(epoch_length)

        self.stage_middleware(
            prefix, functools.partial(
                self._loop, loader, self.iter_eta, prefix)
        )

        self.state.last_prefix = prefix

    @torch.no_grad()
    def validate(self, loader, epoch_length: Optional[int] = None, prefix: str = "val"):
        # Switch to eval mode
        self.checkpoint_manager.eval()

        # Get epoch length
        if not epoch_length:
            self.state.epoch_length = self._try_infer_epoch_length(loader)

        self.iter_eta = EstimatedTimeOfArrival(epoch_length)

        self.stage_middleware(
            prefix, functools.partial(
                self._loop, loader, self.iter_eta, prefix)
        )

        self.state.last_prefix = prefix

    def test(self, loader, epoch_length: Optional[int] = None, prefix: str = "test"):
        self.validate(loader, epoch_length=epoch_length, prefix=prefix)

    def forward(self, batch, batch_idx: int, prefix: str):
        raise NotImplementedError

    def _loop(self, loader, eta: EstimatedTimeOfArrival, prefix: str):
        # with self.meters, self.coroutine_scheduler as coroutine_scheduler:
        meters = self.meters[prefix]
        with meters:
            for batch_idx, batch in enumerate(loader):

                if self.state.training:
                    # Update state
                    self.state.step += 1

                # batch_size = coroutine_scheduler.run(
                #     self.forward(batch, batch_idx, prefix)
                # )
                batch_size = self.forward(batch, batch_idx, prefix)

                eta.update(n=batch_size if batch_size else 1)

                if self.state.debug:
                    break

        _logger.info(f"{prefix} complete: {meters}")

    @staticmethod
    def _try_infer_epoch_length(loader) -> Optional[int]:
        epoch_length = None
        if isinstance(loader, DataLoader):
            epoch_length = len(loader)
        elif hasattr(loader, "__len__"):
            epoch_length = len(loader)
        _logger.info("try infer epoch_length=%s", epoch_length)
        return epoch_length

    @staticmethod
    def every_n_iters(batch_idx: int, n: int = 1, debug: bool = False) -> bool:
        return (batch_idx > 0 and batch_idx % n == 0) or debug

    # def _on_epoch_start(self, prefix: str):
    #     pass

    # def _on_epoch_end(self, prefix: str):
    #     pass

    def stage_middleware(self, prefix: str, next_fn: Callable):
        next_fn()

    def epoch_middleware(self, next_fn: Callable):
        next_fn()

    def run(self, data_module: DataModule, max_epochs: int, debug: bool = False):
        _logger.info("checkpoint manager: %s", self.checkpoint_manager)

        self.state.debug = debug
        self.epoch_eta = EstimatedTimeOfArrival(
            max_epochs, initial=self.state.epoch)

        while self.state.epoch < max_epochs:

            def f():
                if data_module.train_loader:
                    self.train(
                        data_module.train_loader,
                        epoch_length=data_module.train_loader_len,
                    )

                if data_module.val_loader:
                    self.validate(
                        data_module.val_loader,
                        epoch_length=data_module.val_loader_len,
                    )

                if data_module.test_loader:
                    self.test(
                        data_module.test_loader,
                        epoch_length=data_module.test_loader_len,
                    )
                self.epoch_eta.update()
                _logger.info(f"epoch complete: {self.epoch_eta}")

            self.epoch_middleware(f)

            if debug:
                break

        _logger.info("total time: %s", self.epoch_eta.elapsed_time)

    def set_coroutine_delay(self, delay: int):
        self.coroutine_scheduler.delay = delay

    def backward(self, loss: Tensor, optimizer: Optimizer):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def break_iter(self):
        self._break = True
