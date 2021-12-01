import logging
from dataclasses import Field, dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Type

_logger = logging.getLogger(__name__)


@dataclass
class State:

    def state_dict(self) -> dict:
        """将 State 序列化成 nested dict
        """
        ret = {}
        for key, value in self.__dict__.items():
            if isinstance(value, State):
                ret[key] = asdict(value)
            else:
                ret[key] = value

        return ret

    def load_state_dict(self, state_dict: dict):
        """将 nested dict 加载回 State
        """
        for key, value in state_dict.items():
            if isinstance(value, dict):
                state = getattr(self, key)
                if isinstance(state, State):
                    state.load_state_dict(value)
                else:
                    setattr(self, key, value)
            else:
                setattr(self, key, value)


@dataclass
class IterationState(State):
    # 总的iteration
    iteration: int = 0
    # 一般情况下没用
    max_iterations: Optional[int] = None
    # 一般情况下有用
    epoch_length: Optional[int] = None


@dataclass
class EpochState(State):
    epoch: int = 0
    max_epochs: Optional[int] = None

    # 记录best_acc之类的
    metrics: Dict[str, Any] = field(default_factory=dict)

    train_state: IterationState = field(default_factory=IterationState)
    val_state: IterationState = field(default_factory=IterationState)
    test_state: IterationState = field(default_factory=IterationState)


class Engine:

    @staticmethod
    def enumerate(iterable: Iterable):
        return enumerate(iterable, start=1)

    @staticmethod
    def every(i: int, n: int = 1) -> bool:
        return i > 0 and i % n == 0

    def step(self, state: IterationState, epoch: int):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
