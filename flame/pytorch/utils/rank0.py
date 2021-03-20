from typing import Callable
import torch.distributed as dist
import functools


class FakeObject:

    def __getattr__(self, _name: str) -> Callable:
        # 将所有函数替换成do nothing
        return do_nothing


def do_nothing(*args, **kwargs) -> FakeObject:
    # 只要被调用，就返回FakeObject
    return FakeObject()


def rank0(func):
    """run func only on rank 0

    You can use it as a decorator

    .. code-block:: python

        @rank0
        def my_print(*args, **kwargs):
            print(*args, **kwargs)

    
    Args:
        func: function or lambda
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if dist.get_rank() == 0:
            return func(*args, **kwargs)
        else:
            return FakeObject()

    return wrapper
