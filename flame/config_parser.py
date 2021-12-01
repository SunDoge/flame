import inspect
import logging
from typing import Any, Callable, Dict, Union
import importlib
import copy
import rich
import functools

KEY_NAME = '_name'
KEY_USE = '_use'
KEY_CALL = '_call'
PREFIX_PLACEHOLDER = '$'
IMPORT_PLACEHOLDER = '@'

_logger = logging.getLogger(__name__)


class ConfigParser:

    def __init__(self, **kwargs) -> None:
        self.placeholders = kwargs

    def _parse_object(self, config: dict, depth: int):
        if depth < 0:
            return config

        # config_copied = copy.deepcopy(config)
        # config_copied = config.copy()
        # name = config_copied.pop(KEY_NAME)

        name = config[KEY_NAME]
        func = require(name)

        # kwargs = {k: self.parse(v) for k, v in config.items()}
        kwargs = {}
        for k, v in config.items():
            if isinstance(v, str) and v.startswith(PREFIX_PLACEHOLDER):
                kwargs[k] = self.placeholders[k]
            elif k != KEY_NAME:  # 过滤掉_name
                kwargs[k] = self.parse(v, depth=depth)
        # rich.print(kwargs)
        return func(**kwargs)

    def parse(self, config: Union[dict, list, float, int, str], depth: int = 64):
        if isinstance(config, dict):
            if KEY_NAME in config:
                obj = self._parse_object(config, depth - 1)
                return obj

        if isinstance(config, list):
            return [self.parse(c, depth=depth) for c in config]

        if isinstance(config, str):
            # auto import
            if config.startswith(IMPORT_PLACEHOLDER):
                return require(config[1:])

        return config


def require(name: str) -> Any:
    """
    根据路径名自动import

    torch.nn.Conv2d -> torch.nn + Conv2d
    """
    module_name, _sep, attribute_name = name.rpartition('.')
    module = importlib.import_module(module_name)
    attribute = getattr(module, attribute_name)
    return attribute


# class DependencyInjector:
#     """
#     Rules:
#     1. $var will lookup `var` in dict
#     2. _call for call method
#     3. _use for import method
#     4. _use with args means partial
#     5. we use `_` to concat names
#     6. finally recursive dict will be flatten
#     """


#     def __init__(self) -> None:
#         self.container = dict()


#     def parse(self, config: Union[dict, list]):
#         if isinstance(config, list):
#             # FIXME
#             return [self.parse(c) for c in config]

#         elif isinstance(config, dict):
#             if KEY_USE in config:
#                 func = self.parse_key_use(config)

#             elif KEY_CALL in config:
#                 pass
#             else:
#                 pass


#     def parse_key_use(self, config: dict) -> Callable:
#         use = config.pop(KEY_USE)
#         func = require(use)
#         kwargs = {k: self.parse(v) for k, v in config.items()}
#         if kwargs:
#             return functools.partial(func, **kwargs)
#         else:
#             return func


#     def parse_key_call(self, config: dict) -> Any:
#         call = config.pop(KEY_CALL)
#         func = require(call)
#         kwargs = {k: self.parse(v) for k, v in config.items()}
