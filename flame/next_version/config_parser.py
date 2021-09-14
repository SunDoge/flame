import inspect
import logging
from typing import Any, Dict
import importlib
import copy

KEY_NAME = '_name'

_logger = logging.getLogger(__name__)


class ConfigParser:

    def __init__(self) -> None:
        pass

    def parse_object(self, config: dict):
        config = copy.deepcopy(config)
        name = config.pop(KEY_NAME)
        func = include_by_name(name)

        kwargs = {k: self.parse(v) for k, v in config.items()}

        return func(**kwargs)

    def parse(self, config: dict):
        if isinstance(config, dict):
            if KEY_NAME in config:
                return self.parse_object(config)

        if isinstance(config, list):
            return [self.parse(c) for c in config]

        return config


def include_by_name(name: str) -> Any:
    """
    根据路径名自动import

    torch.nn.Conv2d -> torch.nn + Conv2d
    """
    module_name, _sep, attribute_name = name.rpartition('.')
    module = importlib.import_module(module_name)
    attribute = getattr(module, attribute_name)
    return attribute
