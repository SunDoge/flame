from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Tuple, Union
from tabulate import tabulate
import logging

_logger = logging.getLogger(__name__)


class Registry(Iterable[Tuple[str, Callable]]):

    def __init__(self, name: str, parent: Optional['Registry'] = None) -> None:
        self._name = name
        self._func_mapping: Dict[str, Callable] = {}
        self.parent = parent

    def _do_register(self, name: str, func: Callable):
        assert (
            name not in self._func_mapping
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )

        self._func_mapping[name] = func

    def register(self, func: Optional[Callable] = None):
        if func is None:
            def deco(func_or_class: Callable):
                name = func_or_class.__name__
                self._do_register(name, func)
                return func_or_class

            return deco
        else:
            name = func.__name__
            self._do_register(name, func)

    def get(self, name: str) -> Callable:
        ret = self._func_mapping.get(name)

        if ret is None:
            if self.parent is not None:
                # 如果当前registry没找到，但是有parent，就去上一级找
                ret = self.parent.get(name)
            else:
                # 否则就是没找到
                raise KeyError(
                    "No object named '{}' found in '{}' registry!".format(
                        name, self._name)
                )

        return ret

    def __contains__(self, key: Union[str, Callable]) -> bool:
        name = self._to_name(key)
        return name in self._func_mapping

    def __repr__(self) -> str:
        table_headers = ["Names", "Objects"]
        table = tabulate(
            self._func_mapping.items(), headers=table_headers, tablefmt="fancy_grid"
        )
        return "Registry of {}:\n".format(self._name) + table

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        return iter(self._func_mapping.items())

    def replace(self, name: str, func: Callable):
        assert self.get(name), 'Cannot replace unregistered func'
        _logger.debug('replace %s from %s to %s', name, self.get(name), func)
        self._func_mapping[name] = func

    @staticmethod
    def _to_name(cls: Callable):
        if isinstance(cls, str):
            return cls
        else:
            return cls.__name__

    def build_from_cfg(self, cfg: dict, *args, type_key: str = '_type') -> Any:
        if not isinstance(cfg, dict):
            raise TypeError(f'cfg must be a dict, but got {type(cfg)}')

        if type_key not in cfg:
            raise KeyError()

        func_key = cfg[type_key]
        func = self.get(func_key)
        kwargs = {}

        for key, value in cfg.items():
            if key == type_key:
                continue

            if isinstance(value, dict) and type_key in value:
                # 递归创建
                new_value = self.build_from_cfg(value, type_key=type_key)
                kwargs[key] = new_value
            else:
                kwargs[key] = value

        return func(*args, **kwargs)
