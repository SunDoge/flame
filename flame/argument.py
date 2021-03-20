"""
我先用标准库，后面再考虑要不要用typed-args
"""

import argparse
from typing import List, Optional
from pathlib import Path
from argparse import ArgumentParser
import shlex
import sys
import os
import logging

_logger = logging.getLogger(__name__)


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


def get_command() -> str:
    # 记录实验路径
    cmd = [f'cd {shlex.quote(os.getcwd())}']

    # 记录环境变量
    envs = ['CUDA_VISIBLE_DEVICES']

    for env in envs:
        value = os.environ.get(env)

        if value is not None:
            cmd.append(f'export {env}={shlex.quote(value)}')

    # 记录命令，遇到-开头就换个行
    args = ''
    for arg in sys.argv:
        if arg.startswith('-'):
            args += ' \\n'

        args += arg
        args += ' '

    cmd.append(f'{sys.executable} {args}')

    return '\n'.join(cmd)


def save_command(filename: str):
    """

    保存实验命令，包括实验目录，实验用到的环境变量，实验命令

    Args:
        filename: e.g. experiment_dir / run.sh
    """

    cmd_str = get_command()

    _logger.info('save command to %s', filename)
    with open(filename, 'w') as f:
        f.write(cmd_str)
