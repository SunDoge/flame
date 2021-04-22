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
import typed_args as ta
from dataclasses import dataclass

_logger = logging.getLogger(__name__)


@dataclass
class BasicArgs(ta.TypedArgs):
    config: List[str] = ta.add_argument(
        '-c', '--config', type=str, action='append',
        default=[],
        help='config file or snippet',
    )
    local: List[str] = ta.add_argument(
        '-l', '--local', type=str, action='append',
        default=[],
        help='local variables'
    )
    output_dir: Path = ta.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=Path('exps'),
        help='dir to host all your experiments'
    )
    experiment_name: str = ta.add_argument(
        '-e', '--experiment-name',
        type=str,
        default='000',
        help='experiment name'
    )
    debug: bool = ta.add_argument(
        '-d', '--debug',
        action='store_true',
        help='activate debug mode'
    )
    yes: bool = ta.add_argument(
        '-y', '--yes',
        action='store_true',
        help='to skip some steps'
    )


# class BasicArgs:

#     config: List[str]
#     local: List[str]
#     output_dir: Path
#     experiment_name: str
#     debug: bool
#     yes: bool
#     world_size: int


def add_basic_arguments(parser: Optional[ArgumentParser] = None) -> ArgumentParser:

    if parser is None:
        parser = ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        type=str, action='append', default=[],
        help='config file or snippet'
    )
    parser.add_argument(
        '-l', '--local',
        type=str, action='append', default=[],
        help='local variables'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=Path('exps'),
        help='dir to host all your experiments'
    )
    parser.add_argument(
        '-e', '--experiment-name',
        type=str,
        default='000',
        help='experiment name'
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='activate debug mode'
    )
    # parser.add_argument(
    #     '--world-size',
    #     type=int, default=1,
    #     help='number of GPUs/processes for distributed training'
    # )
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='to skip some steps'
    )
    return parser


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
