import functools
from typing import Callable
from injector import inject, Injector
from .events import State


def every_epochs(func: Callable, n: int = 1):

    @inject
    @functools.wraps(func)
    def wrapper(state: State, container: Injector, *args, **kwargs):
        if state.every_epochs(n):
            return container.call_with_injection(
                func,
                args=args,
                kwargs=kwargs,
            )

    return wrapper


def every_iterations(func: Callable, n: int = 1):

    @inject
    @functools.wraps(func)
    def wrapper(state: State, container: Injector, *args, **kwargs):
        if state.every_iterations(n):
            return container.call_with_injection(
                func,
                args=args,
                kwargs=kwargs,
            )

    return wrapper
