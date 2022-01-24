from datetime import datetime, timedelta
from typing import Optional
from flame.pytorch.experimental.engine.events import Events, State
from flame.pytorch.experimental.engine import Engine
from injector import inject
import logging
from flame.core.timing import format_timedelta
from tqdm import tqdm

_logger = logging.getLogger(__name__)


# class EstimatedTimeOfArrival:

#     def __init__(self) -> None:
#         self._start_time = None
#         self._end_time = datetime.now()
#         self._start_iteration = 0
#         self._max_iterations = 0
#         self._current_iteration = 0

#     def update(self):

#         if self._start_time is None:
#             self._start_time = datetime.now()

#         self._end_time = datetime.now()
#         self._current_iteration += 1

#     @property
#     def duration(self) -> datetime:
#         return self._end_time - self._start_time

#     @property
#     def remain_time(self) -> datetime:
#         remain = max(self._max_iterations - self._current_iteration, 0)
#         past = max(self._current_iteration - self._start_iteration, 1)
#         return self.duration / past * remain

#     @property
#     def arrival_time(self) -> datetime:
#         return self._end_time + self.remain_time

#     @property
#     def elasped_time(self) -> datetime:
#         return self._end_time - self._start_time

#     def __str__(self) -> str:
#         arrival_time_str = self.arrival_time.strftime('%Y-%m-%d %H:%M:%S')
#         fmt_str = f'ETA: {self.remain_time}/{arrival_time_str}'
#         return fmt_str

#     def attach(self, engine: Engine):
#         """
#         如果有evaluator，evaluator也需要attach
#         """
#         self._start_iteration += engine.state.global_iteration
#         self._max_iterations += engine.state.max_iterations
#         self._current_iteration = self._start_iteration

#         _logger.info('start: %s, current: %s, max: %s', self._start_iteration,
#                      self._current_iteration, self._max_iterations)

#         engine.add_event_handler(Events.ITERATION_COMPLETED, self.update)


class ExponentialMovingAverage:

    def __init__(self, smoothing: float = 0.3) -> None:
        self.alpha = smoothing
        self.calls = 0
        self.last = 0

    def __call__(self, x: Optional[float] = None) -> float:
        beta = 1. - self.alpha
        if x is not None:
            self.last = self.alpha * x + beta * self.last
            self.calls += 1

        return self.last / (1. - beta ** self.calls) if self.calls else self.last


class EstimatedTimeOfArrival:

    def __init__(self, total: Optional[int] = None, initial: int = 0, prefix: str = '') -> None:
        self._start_time = datetime.now()
        self._end_time = datetime.now()
        self._total = total
        self._inital = initial
        self._n = initial  # num iterations
        self._prefix = prefix

    def update(self, n: int = 1):
        # if self._start_time is None:
        #     self._start_time = datetime.now()

        self._n += n
        self._end_time = datetime.now()

    @property
    def remaining_time(self) -> timedelta:
        return self.elapsed_time / self.elapsed * self.remaining

    @property
    def arrival_time(self) -> datetime:
        return datetime.now() + self.remaining_time

    @property
    def elapsed_time(self) -> timedelta:
        return self._end_time - self._start_time

    @property
    def remaining(self) -> int:
        return self._total - self._n

    @property
    def elapsed(self) -> int:
        return self._n - self._inital

    @property
    def rate(self) -> float:
        return self.elapsed / self.elapsed_time.total_seconds()

    def reset(self):
        self._start_time = None
        self._inital = 0
        self._n = 0

    def __str__(self) -> str:
        remaining_time_str = format_timedelta(self.remaining_time)

        arrival_time_str = self.arrival_time.strftime('%Y-%m-%d %H:%M:%S')

        fmt_str = f'{self._prefix}: [{self._n}/{self._total}] {self.rate:.2f}it/s R={remaining_time_str} A={arrival_time_str}'
        return fmt_str


class EpochEta(EstimatedTimeOfArrival):
    """

    如果有evaluator，必须放在evaluator之后
    """

    def attach(self, engine: Engine, prefix: str = 'Epoch'):
        self._prefix = prefix
        self._inital = engine.state.epoch
        self._n = engine.state.epoch
        self._total = engine.state.max_epochs

        engine.add_event_handler(Events.EPOCH_COMPLETED, self.update)


class IterationEta(EstimatedTimeOfArrival):

    def attach(self, engine: Engine, prefix: str = 'Train'):
        self._prefix = prefix
        self._inital = engine.state.local_iteration
        self._n = engine.state.local_iteration
        self._total = engine.state.epoch_length

        engine.add_event_handler(Events.ITERATION_COMPLETED, self.update)
        engine.add_event_handler(Events.EPOCH_STARTED, self.reset)
