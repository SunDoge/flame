from datetime import datetime
from typing import Optional
from flame.pytorch.experimental.engine.events import State
from injector import inject


class EstimatedTimeOfArrival:

    def __init__(self, start_iteration: int, max_iteration: int) -> None:
        self._start_time = None
        self._end_time = datetime.now()
        self._start_iteration = start_iteration
        self._max_iterations = max_iteration
        self._current_iteration = start_iteration

    @inject
    def update(self):

        if self._start_time is None:
            self._start_time = datetime.now()

        self._end_time = datetime.now()
        self._current_iteration += 1

    @property
    def duration(self) -> datetime:
        return self._end_time - self._start_time

    @property
    def remain_time(self) -> datetime:
        remain = max(self._max_iterations - self._current_iteration, 0)
        past = max(self._current_iteration - self._start_iteration, 0)
        return self.duration / past * remain

    @property
    def arrival_time(self) -> datetime:
        return self._end_time + self.remain_time

    def __str__(self) -> str:
        arrival_time_str = self.arrival_time.strftime('%Y-%m-%d %H:%M:%S')
        fmt_str = f'ETA: {self.remain_time}/{arrival_time_str}'
        return fmt_str
