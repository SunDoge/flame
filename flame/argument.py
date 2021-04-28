"""
我先用标准库，后面再考虑要不要用typed-args
"""

import logging
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import typed_args as ta

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
