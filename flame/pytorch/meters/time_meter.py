from typing import Optional
from .base_meter import Meter
from datetime import timedelta, datetime
import time
from tqdm import tqdm
from flame.utils.timing import format_timedelta


class EstimatedTimeOfArrival(Meter):

    def __init__(self, prefix: str, total: int, initial: int = 0) -> None:
        super().__init__()
        self.prefix = prefix
        self.total = total
        self.initial = initial
        self.count = initial
        self.start_time = time.time()
        self.end_time = time.time()

    def update(self, n: int = 1):
        # if self._start_time is None:
        #     self._start_time = datetime.now()

        self.count += n
        self.end_time = time.time()

    @property
    def remaining_time(self) -> float:
        return self.elapsed_time / self.elapsed * self.remaining

    @property
    def arrival_time(self) -> datetime:
        return datetime.now() + timedelta(seconds=self.remaining_time)

    @property
    def elapsed_time(self) -> float:
        return self.end_time - self.start_time

    @property
    def remaining(self) -> int:
        return self.total - self.count

    @property
    def elapsed(self) -> int:
        return self.count - self.initial

    @property
    def rate(self) -> float:
        return self.elapsed / self.elapsed_time

    def reset(self):
        self.start_time = time.time()
        self.initial = 0
        self.count = 0

    def __str__(self) -> str:
        remaining_time_str = format_timedelta(
            timedelta(seconds=self.remaining_time)
        )

        arrival_time_str = self.arrival_time.strftime('%Y-%m-%d %H:%M:%S')

        fmt_str = f'{self.prefix}: [{self.count}/{self.total}] {self.rate:.2f}it/s R={remaining_time_str} A={arrival_time_str}'
        return fmt_str
