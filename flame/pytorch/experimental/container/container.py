from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from injector import Injector, inject, get_bindings
import types

Kwargs = Dict[str, Any]
CachedFunction = Tuple[Callable, Kwargs]
T = TypeVar('T')


def cached(func: T) -> T:
    func.__cached__ = True
    return func


def cached_inject(func: T) -> T:
    return cached(inject(func))

# def cache_functions(obj, injector: Injector) -> Dict[str, CachedFunction]:
#     pass


def get_dependencies(container: Injector, callable: Callable, **kwargs):
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


def cache_injections(func: Callable, injector: Injector, self_) -> Callable:
    dependencies = get_dependencies(injector, func)
    # print('dep:', dependencies)

    def f(*args, **kwargs):
        kwargs.update(dependencies)
        # return func(self_, *args, **kwargs)
        # print('func:', func)
        # print('kwargs:', kwargs)

        # dependencies.update(kwargs)
        return func(*args, **kwargs)

    return f


class Container:

    def __init__(self, injector: Optional[Injector] = None) -> None:

        if injector is None:
            injector = Injector()

        self._injector = injector

        for key in dir(self):
            value = getattr(self, key)
            if getattr(value, '__cached__', False) == True:
                # print(key, value)
                new_value = cache_injections(value, injector, self)
                # print('new_value: ', new_value)
                setattr(self, key, new_value)


if __name__ == '__main__':
    from injector import singleton

    @singleton
    class A:
        pass

    class B(Container):

        def __init__(self, injector: Optional[Injector] = None) -> None:
            super().__init__(injector=injector)

        @cached_inject
        def call_with_a(self, a: A, input_a=None):
            print(a)

        @cached_inject
        def call_with_kwargs(self, a: A, train=False, input_a=None):
            print(a)
            print('train', train)

    b = B()

    a1 = A()
    print('a1', a1)

    b.call_with_a()
    b.call_with_kwargs(train=True)

    B.call_with_a(None, a1)
    B.call_with_kwargs(None, a1, train=True)
