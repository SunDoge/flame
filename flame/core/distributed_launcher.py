import typed_args as ta
from dataclasses import dataclass


class Args(ta.TypedArgs):
    no_python: bool = ta.add_argument(
        '--no-python'
    )



def launch():
    pass


