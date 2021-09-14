import inspect
import typed_args as ta
from dataclasses import dataclass
import rich
from .config_parser import require
import pydoc
from icecream import ic

@dataclass
class Args(ta.TypedArgs):
    name: str = ta.add_argument(
        help='func or class name, e.g. torch.nn.Linear'
    )


def main():
    args = Args.from_args()
    func = require(args.name)
    signature = inspect.signature(func)
    doc = inspect.getdoc(func)
    rich.print(signature)
    rich.print(doc)


if __name__ == '__main__':
    main()
