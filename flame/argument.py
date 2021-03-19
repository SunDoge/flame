"""
我先用标准库，后面再考虑要不要用typed-args
"""

import argparse
from typing import List, Optional
from pathlib import Path
from argparse import ArgumentParser


class BasicArgs:

    config: List[str]
    output_dir: Path
    experiment_name: str
    debug: bool
    yes: bool
    world_size: int


def add_basic_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        '-c', '--config',
        type=str, action='append',
        help='config file or snippet'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        help='dir to host all your experiments'
    )
    parser.add_argument(
        '-e', '--experiment-name',
        type=str,
        help='experiment name'
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='activate debug mode'
    )
    parser.add_argument(
        '--world-size',
        type=int, default=1,
        help='number of GPUs/processes for distributed training'
    )
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='to skip some steps'
    )


def parse_basic_args(parser: Optional[ArgumentParser] = None) -> BasicArgs:
    if parser is None:
        parser = ArgumentParser()

    parser = add_basic_arguments(parser)
    args = parser.parse_args()
    
    return args
