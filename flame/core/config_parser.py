import functools
import importlib
import logging
from typing import Any, Union

_logger = logging.getLogger(__name__)

KEY_NAME = '_name'
KEY_CALL = '_call'
KEY_USE = '_use'
PREFIX_PLACEHOLDER = '$'
PREFIX_IMPORT = '@'

ParsableConfig = Union[dict, list, str, float, int]


class ConfigParser:

    def __init__(self, **kwargs) -> None:
        self.container = kwargs

    def parse_root_config(self, root_config: dict):
        for key, value in root_config.items():
            if key not in self.container:
                self.container[key] = self.dispatch(value, root_config)

        if KEY_CALL in root_config:
            name = root_config[KEY_CALL]
            func = require(name)
            return func(**self.container)

        return self.container

    def parse(self, config: ParsableConfig):
        return self.dispatch(config, config)

    def dispatch(self, value: ParsableConfig, root_config: dict):
        if isinstance(value, str):
            return self._parse_str(value, root_config)
        elif isinstance(value, dict):
            if KEY_CALL in value:
                return self._parse_object(value, root_config)
            elif KEY_USE in value:
                return self._parse_function(value, root_config)
            else:
                return self._parse_dict(value, root_config)
        elif isinstance(value, list):
            return self._parse_list(value, root_config)
        elif isinstance(value, (float, int)):
            return value

    def _parse_list(self, value: list, root_config: dict):
        return [self.dispatch(v, root_config) for v in value]

    def _parse_dict(self, value: dict, root_config: dict):
        return {k: self.dispatch(v, root_config) for k, v in value.items()}

    def _parse_object(self, value: dict, root_config: dict):
        name = value[KEY_CALL]
        func = require(name)

        kwargs = {k: v for k, v in value.items() if k != KEY_CALL}

        # TODO: raise better exception
        try:
            return func(**self._parse_dict(kwargs, root_config))
        except Exception as e:
            raise Exception(func, e)

    def _parse_function(self, value: dict, root_config: dict):
        name = value[KEY_USE]
        func = require(name)

        kwargs = {k: v for k, v in value.items() if k != KEY_USE}
        if kwargs:
            # TODO: raise exception
            return functools.partial(
                func, **self._parse_dict(kwargs, root_config)
            )
        else:
            return func

    def _parse_str(self, value: str, root_config: dict):
        if value.startswith(PREFIX_PLACEHOLDER):
            name = value[1:]
            if name in self.container:
                return self.container[name]
            else:
                self.container[name] = self.dispatch(
                    root_config[name],
                    root_config
                )
                return self.container[name]
        elif value.startswith(PREFIX_IMPORT):
            name = value[1:]
            return require(name)
        else:
            return value


def require(name: str) -> Any:
    """
    根据路径名自动import

    torch.nn.Conv2d -> torch.nn + Conv2d
    """
    module_name, _sep, attribute_name = name.rpartition('.')
    module = importlib.import_module(module_name)
    attribute = getattr(module, attribute_name)
    return attribute
