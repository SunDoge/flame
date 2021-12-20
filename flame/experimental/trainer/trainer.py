import logging
from typing import Optional

from torch import nn
from torch.utils.data.dataloader import DataLoader

from .checkpoint_manager import CheckpointManager
from .coroutine_scheduler import CoroutineScheduler
from .state import State

_logger = logging.getLogger(__name__)

nn.Module


class BaseTrainer:
    def __init__(self) -> None:
        state = State()
        checkpoint_manager = CheckpointManager()

        checkpoint_manager.register(
            "trainer_state", state.state_dict, state.load_state_dict, state.train
        )
        coroutine_scheduler = CoroutineScheduler()

        self.state = state
        self.checkpoint_manager = checkpoint_manager
        self.coroutine_scheduler = coroutine_scheduler

    def train(self, loader, epoch_length: Optional[int] = None, prefix: str = "train"):

        # Update state
        self.state.epoch += 1

        # Switch to training mode
        self.checkpoint_manager.train()

        # Get epoch length
        if not epoch_length:
            self.state.epoch_length = self._try_infer_epoch_length(loader)

        # Start epoch
        self.on_epoch_start(prefix)

        with self.coroutine_scheduler as coroutine_scheduler:

            for batch_idx, batch in enumerate(loader):

                # Update state
                self.state.step += 1

                coroutine_scheduler.run(self.forward(batch, batch_idx, prefix))

        # End epoch
        self.on_epoch_end(prefix)

    def validate(self, loader, epoch_length: Optional[int] = None, prefix: str = "val"):
        # Switch to eval mode
        self.checkpoint_manager.eval()

        # Get epoch length
        if not epoch_length:
            self.state.epoch_length = self._try_infer_epoch_length(loader)

        # Start epoch
        self.on_epoch_start(prefix)

        with self.coroutine_scheduler as coroutine_scheduler:

            for batch_idx, batch in enumerate(loader):
                coroutine_scheduler.run(self.forward(batch, batch_idx, prefix))

        # End epoch
        self.on_epoch_end(prefix)

    def test(self, loader, epoch_length: Optional[int] = None, prefix: str = "test"):
        self.validate(loader, epoch_length=epoch_length, prefix=prefix)

    def forward(self, batch, batch_idx: int, prefix: str):
        raise NotImplementedError

    @staticmethod
    def _try_infer_epoch_length(loader) -> Optional[int]:
        epoch_length = None
        if isinstance(loader, DataLoader):
            epoch_length = len(loader)
        elif hasattr(loader, "__len__"):
            epoch_length = len(loader)
        _logger.info("try infer epoch_length=%s", epoch_length)
        return epoch_length

    # def _on_epoch_start(self, prefix: str):
    #     pass

    # def _on_epoch_end(self, prefix: str):
    #     pass

    def on_epoch_start(self, prefix: str):
        pass

    def on_epoch_end(self, prefix: str):
        pass

    def run(self):
        raise NotImplementedError

    def set_coroutine_delay(self, delay: int):
        self.coroutine_scheduler.delay = delay
