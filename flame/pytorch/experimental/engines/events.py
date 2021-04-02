from enum import Enum
from typing import Any, Dict, Iterable, Iterator, List, Optional
from dataclasses import dataclass, field
import logging

_logger = logging.getLogger(__name__)


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

    def update_max_iterations(self):
        # 自动计算max_iterations
        if self.max_epochs is not None and self.epoch_length is not None and self.max_iterations is None:
            old_max_iterations = self.max_iterations
            self.max_iterations = self.max_epochs * self.epoch_length
            _logger.info(
                'update max_iterations: %s => %s',
                old_max_iterations, self.max_iterations
            )
        else:
            _logger.warning('max_iterations is not updated')

    def update_local_iteration(self, iteration: int):
        self.local_iteration = iteration

    def update_global_iteration(self):
        self.global_iteration += 1

    def is_done_iterations(self) -> bool:
        return self.max_iterations is not None and self.global_iteration >= self.max_iterations

    def is_done_epochs(self) -> bool:
        return self.max_epochs is not None and self.epoch >= self.max_epochs

    def is_done_count(self) -> bool:
        return (
            self.epoch_length is not None
            and self.max_epochs is not None
            and self.global_iteration >= self.epoch_length * self.max_epochs
        )

    def is_done(self) -> bool:
        # 终止条件
        return self.is_done_iterations() or self.is_done_count() or self.is_done_epochs()

    def reset(self):
        self.epoch = 0
        self.max_epochs = None

        self.local_iteration = 0
        self.global_iteration = 0

        self.epoch_length = None
        self.max_iterations = None

        self.batch = None
        self.output = None
        self.dataloader = None
        self.metrics = {}

    def every_iterations(self, n: int) -> bool:
        return self.global_iteration > 0 and self.global_iteration % n == 0

    def every_epochs(self, n: int) -> bool:
        return self.epoch > 0 and self.epoch % n == 0
