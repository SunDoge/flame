from typing import Any, TypeVar
from rich.console import Console
import rich

T = TypeVar('T')


def rich_format(obj: Any) -> str:
    with rich.get_console().capture() as capture:
        rich.print(obj)
    return capture.get()
