
from flame.auto_builder import import_from_path
from typing import Any, Callable, Dict
from injector import Binder, Injector, NoScope, singleton, SingletonScope, inject
import flame
from pydantic import BaseModel, Field
import copy

BUILDER_KEY = '_builder'
SCOPE_KEY = '_scope'

SCOPE_MAP = {
    'no': lambda x: x,
    'singleton': singleton
}


def builder(config: dict) -> Callable[[Injector], Any]:

    scope_name = config.pop(SCOPE_KEY, 'no')
    scope = SCOPE_MAP[scope_name]

    @scope
    @inject
    def f(container: Injector):

        config_cloned = copy.deepcopy(config)

        builder_path = config_cloned.pop(BUILDER_KEY)

        builder = flame.auto_builder.import_from_path(builder_path)

        kwargs = {}
        for key, value in config_cloned.items():
            if isinstance(value, str) and value.startswith('@'):
                instance = container.get(value[1:])
                kwargs[key] = instance
            else:
                kwargs[key] = value

        return builder(**kwargs)

    return f


def configure_singletons(config: Dict[str, dict]) -> Callable[[Binder], None]:

    def configure(binder: Binder):
        for key, value in config.items():
            binder.bind(key, to=builder(value))

    return configure
