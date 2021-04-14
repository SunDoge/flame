from typing import Any
import copy


class BaseMeter:
    """
    不需要了，定义在reset里面就可以了，vscode可以提示
    """

    def reset(self):
        pass

    def sync(self):
        pass

    def __str__(self) -> str:
        pass

    