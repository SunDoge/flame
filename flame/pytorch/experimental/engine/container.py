from injector import Injector
from typing import Any, Callable, Optional
import functools


class Container(Injector):

    def __init__(
        self,
        modules: Any = None,
        auto_bind: bool = True,
        parent: Optional[Injector] = None
    ) -> None:
        super().__init__(modules=modules, auto_bind=auto_bind, parent=parent)
        self.binder.bind(Container, to=self)

    def with_injection(self, func: Callable) -> Callable:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call_with_injection(func, args=args, kwargs=kwargs)

        return wrapper
