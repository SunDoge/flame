from typing import TypeVar
import inspect

T = TypeVar('T')


def main_fn(func: T) -> T:
    caller = inspect.stack()[1]

    name = caller.frame.f_globals['__name__']

    if name == '__main__':
        func()
    else:
        return func
