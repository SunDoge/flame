import inspect
import logging
from typing import Any, Callable, Dict
import importlib
import copy
import rich

KEY_NAME = '_name'
PREFIX_PLACEHOLDER = '$'

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

    def parse(self, config: dict, depth: int = 64):
        if isinstance(config, dict):
            if KEY_NAME in config:
                obj = self._parse_object(config, depth - 1)
                return obj

        if isinstance(config, list):
            return [self.parse(c, depth=depth) for c in config]

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
