from typing import Coroutine, Deque
from contextlib import AbstractContextManager


class CoroutineScheduler(AbstractContextManager):
    def __init__(self, delay: int = 1) -> None:
        self.delay = delay
        self.queue: Deque[Coroutine] = Deque()

    def run(self, coro: Coroutine):
        output = next(coro)
        if len(self.queue) >= self.delay:
            prev_coro = self.queue.popleft()
            for _ in prev_coro:
                pass
        self.queue.append(coro)
        return output

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.empty_tasks()

    def empty_tasks(self):
        while self.queue:
            coro = self.queue.popleft()
            for _ in coro:
                pass
