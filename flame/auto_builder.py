import importlib
from typing import Any, Callable
import inspect
import copy


def import_from_path(path: str) -> Any:
    """
    根据路径名自动import
    """
    module_name, _sep, attribute_name = path.rpartition('.')
    module = importlib.import_module(module_name)
    attribute = getattr(module, attribute_name)
    return attribute


def signature_contains(obj: Callable, key: str) -> bool:
    """
    测试函数签名里面是否存在特定key
    """
    signature = inspect.signature(obj)
    return key in signature.parameters


def build_from_config(cfg: dict, type_key='_type') -> Any:
    """
    默认_type指定路径
    """
    assert type_key in cfg, f'no path_key: `{type_key}` found in config'

    obj = import_from_path(cfg[type_key])
    assert not signature_contains(
        obj, type_key), f'signature contains illegal key: {type_key}'

    kwargs = get_kwargs_from_config(cfg, type_key=type_key)

    return obj(**kwargs)


def get_kwargs_from_config(cfg: dict, type_key='_type') -> dict:
    kwargs = {}
    for arg_key, arg_value in cfg.items():
        if arg_key == type_key:
            # 过滤掉type_key
            continue

        if isinstance(arg_value, dict) and type_key in arg_value:
            # 递归创建
            new_arg_value = build_from_config(
                arg_value, type_key=type_key
            )
            kwargs[arg_key] = new_arg_value
            continue

        kwargs[arg_key] = arg_value

    return kwargs
