from enum import Enum
from typing import Any, Dict, Iterable, Iterator, List, Optional
from dataclasses import dataclass, field


class Events(Enum):
    # For epoch
    EPOCH_STARTED = 'epoch_started'
    EPOCH_COMPLETED = 'epoch_completed'

    # For training
    STARTED = 'started'
    COMPLETED = 'completed'

    # For iteration
    ITERATION_STARTED = 'iteration_started'
    ITERATION_COMPLETED = 'iteration_completed'
    EXCEPTION_RAISED = 'exception_raised'  # FIXME: 不确定有什么用

    # For data
    GET_BATCH_STARTED = 'get_batch_started'
    GET_BATCH_COMPLETED = 'get_batch_completed'

    # 这几个理论上都用不到
    DATALOADER_STOP_ITERATION = 'dataloader_stop_iteration'
    TERMINATE = 'terminate'
    TERMINATE_SINGLE_EPOCH = 'terminate_single_epoch'


class EventsList:

    def __init__(self) -> None:
        self._events: List[Events] = []

    def _append(self, event: Events):
        self._events.append(event)

    def __getitem__(self, index: int) -> Events:
        return self._events[index]

    def __iter__(self) -> Iterator[Events]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def __or__(self, other: Events) -> 'EventsList':
        self._append(other)
        return self


@dataclass
class State:
    # For epoch engine
    epoch: int = 0
    max_epochs: Optional[int] = None

    # For iteration engine
    # 内循环的iter
    local_iteration: int = 0
    # 总的iter
    global_iteration: int = 0

    # dataloader的长度，和local_iteration相关
    epoch_length: Optional[int] = None

    # 最多跑多少iter，和global_iteration相关
    max_iterations: Optional[int] = None

    batch: Optional[Any] = None  # model input
    output: Optional[Any] = None  # model output
    dataloader: Optional[Iterable[Any]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def update_local_iteration(self, iteration: int):
        self.local_iteration = iteration

    def update_global_iteration(self):
        self.global_iteration += 1

    