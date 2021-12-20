import argparse
from dataclasses import dataclass

import typed_args as ta

import flame


@dataclass
class Args(ta.TypedArgs):
    pass


@flame.main_fn
def main():
    parser = argparse.ArgumentParser("flame")
    args = Args.from_args(parser=parser)
    print(args)
