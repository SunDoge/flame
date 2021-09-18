from logging import debug
from typing import List, Optional
import typed_args as ta
from dataclasses import dataclass
from pathlib import Path
from .config import from_snippet, parse_config
import functools


def parse_gpu_list(gpu_str: str) -> List[int]:
    """
    0-7 => [0,1,2,3,4,5,6,7]
    1,3,5,7 => [1,3,5,7]
    0-3,6-7 => [0,1,2,3,6,7]
    """
    segments = gpu_str.split(',')
    ranges = [s.split('-') for s in segments]
    ranges = [list(map(int, ran)) for ran in ranges]
    res = []
    for ran in ranges:
        if len(ran) == 1:
            res.append(ran[0])
        elif len(ran) == 2:
            res.extend(
                list(range(ran[0], ran[1] + 1))
            )
        else:
            raise Exception

    return res


@dataclass
class BaseArgs(ta.TypedArgs):

    """
    一般训练必须的参数
    """
    config: str = ta.add_argument(
        '-c', '--config',
        type=str,
        help='config file path'
    )
    output_dir: Path = ta.add_argument(
        '-o', '--output-dir',
        default=Path('./exps'),
        type=Path,
        help='folder to save all experiments'
    )
    experiment_name: str = ta.add_argument(
        '-e', '--experiment-name',
        default='000',
        type=str,
        help=''
    )
    debug: bool = ta.add_argument(
        '-d', '--debug',
        action='store_true',
        help='debug mode'
    )
    add: List[str] = ta.add_argument(
        '--add', type=str, action='append',
        default=[]
    )
    """
    """
    rank: int = ta.add_argument(
        '--rank',
        type=int,
        default=0,
    )
    world_size: int = ta.add_argument(
        '--world-size',
        type=int,
        default=1,
    )
    dist_url: str = ta.add_argument(
        '--dist-url',
        type=str,
        default='tcp://127.0.0.1:12345'
    )
    dist_backend: str = ta.add_argument(
        '--dist-backend',
        type=str,
        default='nccl'
    )
    seed: int = ta.add_argument(
        '--seed',
        type=int,
        default=None
    )
    gpu: List[int] = ta.add_argument(
        '--gpu',
        type=parse_gpu_list,
        default=[],
    )
    rank_start: int = ta.add_argument(
        '--rank-start',
        type=int,
        default=0,
    )
    device_id: int = ta.add_argument(
        '--device-id',
        type=int,
        default=0,
    )

    @property
    def experiment_dir(self) -> Path:
        return self.output_dir / ('debug' if self.debug else 'release') / self.experiment_name

    def parse_config(self) -> dict:
        snippet = parse_config(
            self.config,
            self.add
        )
        config = from_snippet(snippet)
        return config
