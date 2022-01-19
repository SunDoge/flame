from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typed_args as ta


@dataclass
class BaseArgs(ta.TypedArgs):
    config_path: Path = ta.add_argument(
        '-c', '--config', type=Path, required=True,
        help='config 文件'
    )

    experiment_dir: Path = ta.add_argument(
        '-e', '--experiment-dir', type=Path,
        default=Path('exps/000'),
        help='实验目录'
    )

    print_freq: int = ta.add_argument(
        '--print-freq', '--pf', type=int, default=-1,
        help='显示 log 的频率，一般为10'
    )

    debug_dir: Path = ta.add_argument(
        '--debug-dir', type=Path,  default=Path('debug'),
        help='debug 专用目录，记得定期删'
    )
    debug: bool = ta.add_argument(
        '-d', '--debug', action='store_true',
        help='debug 模式'
    )
    no_tqdm: bool = ta.add_argument(
        '--no-tqdm', action='store_true',
        help='关闭 tqdm'
    )

    resume: Optional[Path] = ta.add_argument(
        '--resume', type=Path,
        help='resume checkpoint path'
    )


if __name__ == '__main__':
    args = BaseArgs.from_args()
    print(args)
