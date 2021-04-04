from injector import Injector, get_bindings
from typing import Any, Callable, Dict, Optional, Tuple
import functools
import inspect


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

    def get_dependencies(self, callable: Callable, args=(), kwargs={}) -> Tuple[Tuple[Any], Dict[str, Any]]:
        bindings = get_bindings(callable)
        signature = inspect.signature(callable)

        full_args = args

        bound_arguments = signature.bind_partial(*full_args)

        needed = dict(
            (k, v) for (k, v) in bindings.items() if k not in kwargs and k not in bound_arguments.arguments
        )

        dependencies = self.args_to_inject(
            function=callable,
            bindings=needed,
            owner_key=callable.__module__,
        )

        dependencies.update(kwargs)

        return full_args, dependencies


def get_dependencies(container: Injector, callable: Callable, **kwargs) -> Dict[str, Any]:
    bindings = get_bindings(callable)

    needed = dict(
        (k, v) for (k, v) in bindings.items() if k not in kwargs
    )

    dependencies = container.args_to_inject(
        function=callable,
        bindings=needed,
        owner_key=callable.__module__,
    )

    dependencies.update(kwargs)

    return dependencies
