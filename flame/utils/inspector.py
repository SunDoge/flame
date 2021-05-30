"""
从函数或类生成对应的jsonnet config
"""
from collections import namedtuple
from typing import Callable
from flame.auto_builder import import_from_path
import inspect


JSONNET_OBJECT_TEMPLATE = """
{name}: {
    {contents}
}
"""

JSONNET_FUNCTION_TEMPLATE = """
{{
    {name}({args}):: {{
{contents}
    }}
}}
"""


def generate_contents(**kwargs) -> str:

    ret = ''

    for key, value in kwargs.items():
        ret += f'\t{key}: {value},\n'

    return ret


def generate_args(*args) -> str:
    return ', '.join(args)


def generate_function(path: str) -> str:
    func: Callable = import_from_path(path)
    func_name = func.__name__
    signature = inspect.signature(func)

    args = []
    kwargs = {}

    for sig in signature.parameters.values():
        if sig.default is sig.empty:
            # args
            args.append(sig.name)
            kwargs[sig.name] = sig.name
        else:
            if isinstance(sig.default, str):
                kwargs[sig.name] = "'{}'".format(sig.default)
            else:
                kwargs[sig.name] = sig.default

    args_str = generate_args(*args)
    contents = generate_contents(**kwargs)

    return JSONNET_FUNCTION_TEMPLATE.format(
        name=func_name, args=args_str, contents=contents
    )


if __name__ == '__main__':
    # output = generate_function('flame.config.from_snippet')
    output = generate_function('torchvision.transforms.RandomResizedCrop')
    print(output)
