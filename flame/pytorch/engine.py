from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional
import logging

_logger = logging.getLogger(__name__)


@dataclass
class IterationState:
    # 总的iteration
    iteration: int = 0
    # 一般情况下没用
    max_iterations: Optional[int] = None
    # 一般情况下有用
    epoch_length: Optional[int] = None


@dataclass
class EpochState:
    epoch: int = 0
    max_epochs: Optional[int] = None

    # 记录best_acc之类的
    metrics: Dict[str, Any] = field(default_factory=dict)


class IterationEngine:

    def __init__(self, state: IterationState) -> None:
        self.state = state

    def update_state(self, epoch_length: int, max_iterations: Optional[int] = None):
        self.state.epoch_length = epoch_length
        self.state.max_iterations = max_iterations

    @staticmethod
    def enumerate(iterable: Iterable):
        return enumerate(iterable, start=1)

    @staticmethod
    def every_n_iterations(i: int, n: int = 1) -> bool:
        return i > 0 and i % n == 0


class EpochEngine:

    def __init__(self, state: EpochState) -> None:
        self.state = state

    def update_state(self, max_epochs: int):
        self.state.max_epochs = max_epochs

    @staticmethod
    def every_n_epochs(epoch: int, n: int = 1) -> bool:
        return epoch > 0 and epoch % n == 0
