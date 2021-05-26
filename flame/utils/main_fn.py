from typing import TypeVar
import inspect

T = TypeVar('T')


def main_fn(func: T) -> T:
    # 拿到调用这个函数的 FrameInfo
    caller = inspect.stack()[1]

    # 拿到 __name__
    name = caller.frame.f_globals['__name__']

    if name == '__main__':
        func()
    else:
        return func
