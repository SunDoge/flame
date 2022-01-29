from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import typed_args as ta
from datetime import datetime
from .config import parse_config, from_snippet, dump_to_json
from shlex import quote
import sys
import os
import logging

_logger = logging.getLogger(__name__)


def parse_gpu_list(gpu_str: str) -> List[int]:
    """
    0-7 => [0,1,2,3,4,5,6,7]
    1,3,5,7 => [1,3,5,7]
    0-3,6-7 => [0,1,2,3,6,7]
    """
    segments = gpu_str.split(",")
    ranges = [s.split("-") for s in segments]
    ranges = [list(map(int, ran)) for ran in ranges]
    res = []
    for ran in ranges:
        if len(ran) == 1:
            res.append(ran[0])
        elif len(ran) == 2:
            res.extend(list(range(ran[0], ran[1] + 1)))
        else:
            raise Exception

    return res


@dataclass
class BaseArgs(ta.TypedArgs):
    config_file: Optional[Path] = ta.add_argument(
        "-c", "--config", type=Path, help="config 文件"
    )

    experiment_dir: Path = ta.add_argument(
        "-e", "--experiment-dir", type=Path, default=Path("exps/000"), help="实验目录"
    )

    apply: List[str] = ta.add_argument(
        "-a",
        "--apply",
        type=str,
        action="append",
        default=[],
        help="额外 config，可 merge 到 main config",
    )

    print_freq: int = ta.add_argument(
        "--print-freq", "--pf", type=int, default=1000, help="显示 log 的频率，一般为10"
    )

    temp_dir: Path = ta.add_argument(
        "--temp-dir", type=Path, default=Path("temp"), help="临时目录，记得定期删"
    )
    debug: bool = ta.add_argument(
        "-d", "--debug", action="store_true", help="debug 模式")
    no_tqdm: bool = ta.add_argument(
        "--no-tqdm", action="store_true", help="关闭 tqdm")

    resume: Optional[Path] = ta.add_argument(
        "--resume", type=Path, help="resume checkpoint path"
    )

    force: bool = ta.add_argument(
        "-f", "--force", action="store_true", help="移除旧实验目录到 temp dir，强制创建新实验目录"
    )

    # 默认使用 cpu，后续可能加入 xla 支持
    gpu: List[int] = ta.add_argument(
        "--gpu", type=parse_gpu_list, default=[], help="指定gpu，`1,2,5-7 -> [1,2,5,6,7]`"
    )

    def try_make_experiment_dir(self):
        if self.experiment_dir.exists():
            if self.force:
                timestamp = datetime.now().strftime('%Y_%m_%d-%H.%M.%S')
                new_experiment_name = self.experiment_dir.name + '-' + timestamp
                new_experiment_dir = self.temp_dir / new_experiment_name
                print(
                    f"move old experiment dir from {self.experiment_dir} to {new_experiment_dir}"
                )
                # 确保 temp dir 存在
                self.temp_dir.mkdir(parents=True, exist_ok=True)
                self.experiment_dir.rename(new_experiment_dir)
            else:
                print(
                    f'实验目录 {self.experiment_dir} 已存在，可使用 -f/--force 参数覆盖实验目录')
                exit(0)

        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> dict:
        assert self.config_file, "请指定 config file"
        snippet = parse_config(self.config_file, self.apply)
        config = from_snippet(snippet)
        return config

    def save_config(self, name: str = "config.json"):
        config = self.config
        dump_to_json(config, self.experiment_dir / name)

    def save_command(self, name: str = 'run.sh'):
        with open(self.experiment_dir / name, 'w') as f:
            f.write(f"cd {quote(os.getcwd())}\n")
            envs = ['CUDA_VISIBLE_DEVICES']
            for env in envs:
                value = os.environ.get(env, None)
                if value is not None:
                    f.write(f'export {env}={quote(value)}\n')

            args_str = ' '.join(quote(arg)for arg in sys.argv)
            f.write(f'alias python={sys.executable}\n')
            f.write(f'python {args_str}\n')

        _logger.info('save command to %s', self.experiment_dir / name)


if __name__ == "__main__":
    args = BaseArgs.from_args()
    args.try_make_experiment_dir()
    print(args)
    print(args.config)
