from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import typed_args as ta
from datetime import datetime
import logging
from .config import parse_config, from_snippet


@dataclass
class BaseArgs(ta.TypedArgs):
    config_file: Optional[Path] = ta.add_argument(
        '-c', '--config', type=Path,
        help='config 文件'
    )

    experiment_dir: Path = ta.add_argument(
        '-e', '--experiment-dir', type=Path,
        default=Path('exps/000'),
        help='实验目录'
    )

    apply: List[str] = ta.add_argument(
        '-a', '--apply', type=str,
        action='append', default=[],
        help='额外 config，可 merge 到 main config'
    )

    print_freq: int = ta.add_argument(
        '--print-freq', '--pf', type=int, default=-1,
        help='显示 log 的频率，一般为10'
    )

    temp_dir: Path = ta.add_argument(
        '--temp-dir', type=Path,  default=Path('temp'),
        help='临时目录，记得定期删'
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

    force: bool = ta.add_argument(
        '-f', '--force', action='store_true',
        help='移除旧实验目录到 debug dir，强制创建新实验目录'
    )

    def try_make_experiment_dir(self):
        if self.experiment_dir.exists():
            # force 和 debug 都会导致覆盖实验目录
            if self.force or self.debug:
                timestamp = datetime.now().strftime('%Y_%m_%d-%H.%M.%S')
                new_experiment_name = self.experiment_dir.name + '-' + timestamp
                new_experiment_dir = self.temp_dir / new_experiment_name
                print(
                    f'move old experiment dir from {self.experiment_dir} to {new_experiment_dir}'
                )
                self.temp_dir.mkdir(parents=True, exist_ok=True)
                self.experiment_dir.rename(new_experiment_dir)
            else:
                print('实验目录已存在，可使用 -f/--force 参数覆盖实验目录')
                exit(0)

        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> dict:
        assert self.config_file, "请指定 config file"
        snippet = parse_config(self.config_file, self.apply)
        config = from_snippet(snippet)
        return config


if __name__ == '__main__':
    args = BaseArgs.from_args()
    args.try_make_experiment_dir()
    print(args)
    print(args.config)
