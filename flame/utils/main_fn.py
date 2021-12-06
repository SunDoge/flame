from typing import Any, Callable, Optional, TypeVar, Union
import inspect

T = TypeVar('T')


def _main_fn(func: T) -> T:
    # 拿到调用这个函数的 FrameInfo
    caller = inspect.stack()[2]

    # 拿到 __name__
    name = caller.frame.f_globals['__name__']

    if name == '__main__':
        func()
    else:
        return func


def main_fn(func: Optional[T]) -> Union[Callable[[T], T], T]:
    if func:
        return _main_fn(func)
    else:
        return _main_fn
