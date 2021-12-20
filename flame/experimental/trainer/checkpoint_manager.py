from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass
import logging
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LrScheduler
from torch.cuda.amp.grad_scaler import GradScaler

_logger = logging.getLogger(__name__)


ToStateDict = Callable[[], Dict]
LoadStateDict = Callable[[Dict], Any]
ToTraining = Callable[[bool], Any]


def _default_to_training(training: bool):
    pass


@dataclass
class Entry:
    to_state_dict: ToStateDict
    load_state_dict: LoadStateDict
    to_training: ToTraining
    # ref: Any = None


class CheckpointManager:
    def __init__(self) -> None:
        self.registry: Dict[str, Entry] = dict()

    def register(
        self,
        name: str,
        to_state_dict: ToStateDict,
        load_state_dict: LoadStateDict,
        to_training: ToTraining = _default_to_training,
    ):
        self.registry[name] = Entry(
            to_state_dict,
            load_state_dict,
            to_training,
        )

    def train(self, mode: bool = True):
        for name, value in self.registry.items():
            value.to_training(mode)
            _logger.info(f"{name}.training={mode}")

    def eval(self):
        self.train(mode=False)

    def state_dict(self):
        sd = {k: v.to_state_dict() for k, v in self.registry.items()}
        return sd

    def load_state_dict(self, state_dict: Dict):
        for key, value in state_dict.items():
            self.registry[key].load_state_dict(value)

    def register_model(self, model: nn.Module, name: str = "model"):
        if isinstance(
            model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)
        ):
            model = model.module

        self.register(name, model.state_dict, model.load_state_dict, model.train)

    def register_optimizer(self, optimizer: Optimizer, name: str = "optimizer"):
        self.register(name, optimizer.state_dict, optimizer.load_state_dict)

    def register_lr_scheduler(
        self, lr_scheduler: LrScheduler, name: str = "lr_scheduler"
    ):
        self.register(name, lr_scheduler.state_dict, lr_scheduler.load_state_dict)

    def register_grad_scaler(self, grad_scaler: GradScaler, name: str = "grad_scaler"):
        self.register(name, grad_scaler.state_dict, grad_scaler.load_state_dict)
