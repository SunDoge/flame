from typing import Optional
from .base_meter import Meter
from datetime import timedelta, datetime
import time
from flame.core.timing import format_timedelta


class EstimatedTimeOfArrival(Meter):

    def __init__(self, total: int, initial: int = 0) -> None:
        super().__init__()
        # self.prefix = prefix
        self.total = total
        self.initial = initial
        self.count = initial
        self.num_samples = 0
        self.start_time = time.time()
        self.end_time = time.time()

    def update(self, n: int = 1):
        # if self._start_time is None:
        #     self._start_time = datetime.now()

        self.count += 1
        self.end_time = time.time()
        self.num_samples = n

    @property
    def remaining_seconds(self) -> float:
        return self.elapsed_seconds / self.elapsed * self.remaining

    @property
    def remaining_time(self) -> timedelta:
        return timedelta(seconds=self.remaining_seconds)

    @property
    def arrival_time(self) -> datetime:
        return datetime.now() + timedelta(seconds=self.remaining_seconds)

    @property
    def elapsed_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def elapsed_time(self) -> timedelta:
        return timedelta(seconds=self.elapsed_seconds)

    @property
    def remaining(self) -> int:
        return self.total - self.count

    @property
    def elapsed(self) -> int:
        return self.count - self.initial

    @property
    def rate(self) -> float:
        # print(self.elapsed_seconds)
        # elapsed_seconds = 0.1 if self.elapsed_seconds == 0 else self.elapsed_seconds
        return self.elapsed * self.num_samples / self.elapsed_seconds

    def reset(self):
        self.start_time = time.time()
        self.initial = 0
        self.count = 0

    def __str__(self) -> str:
        try:
            remaining_time_str = format_timedelta(self.remaining_time)
            arrival_time_str = self.arrival_time.strftime('%Y-%m-%d %H:%M:%S')
            rate_str = f'{self.rate:.2f}'
        except ZeroDivisionError:
            remaining_time_str = '?'
            arrival_time_str = '?'
            rate_str = '?'

        fmt_str = f'[{self.count}/{self.total}] {rate_str} it/s R={remaining_time_str} A={arrival_time_str}'
        return fmt_str


if __name__ == '__main__':
    eta = EstimatedTimeOfArrival(10)
    print(eta)
