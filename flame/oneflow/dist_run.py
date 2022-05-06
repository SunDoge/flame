"""
基本按照 oneflow.distributed.launch 实现
未实现 logdir
"""

from typing import List, Any, Optional, IO
import typed_args as ta
from dataclasses import dataclass
from flame.core.operating_system import find_free_port
from argparse import REMAINDER
import flame
import os
from flame.core.arguments import parse_gpu_list
import logging
from flame.core.logging import init_logging
import sys
import subprocess
import signal
import time
from icecream import ic

_logger = logging.getLogger(__name__)


@dataclass
class Args(ta.TypedArgs):

    rank_start: int = ta.add_argument(
        '--rank-start', type=int,
        default=0,
    )
    world_size: int = ta.add_argument(
        '--world-size', type=int,
        default=1,
    )
    master_addr: str = ta.add_argument(
        '--master-addr', type=str,
        default='127.0.0.1'
    )
    master_port: int = ta.add_argument(
        '--master-port', type=int,
        default=29500
    )
    redirect_stdout_and_stderr: bool = ta.add_argument(
        '--redirect-stdout-and-stderr', action='store_true',
    )
    # 默认使用 cpu，后续可能加入 xla 支持
    gpu: List[int] = ta.add_argument(
        "--gpu", type=parse_gpu_list, default=[], help="指定gpu，`1,2,5-7 -> [1,2,5,6,7]`"
    )
    debug: bool = ta.add_argument(
        '-d', '--debug', action='store_true',
    )
    no_python: bool = ta.add_argument(
        '--no-python', action='store_true'
    )
    module: bool = ta.add_argument(
        '-m', '--module',
        action='store_true'
    )
    training_script: str = ta.add_argument(
        type=str,
    )
    training_script_args: List[str] = ta.add_argument(
        nargs=REMAINDER
    )


@flame.main_fn
def main():
    args = Args.from_args()
    init_logging(debug=args.debug)
    _logger.info(args)

    # Infer world size
    if len(args.gpu) > args.world_size:
        world_size = len(args.gpu)
        _logger.info('set args.world_size=%d', world_size)
        args.world_size = world_size

    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(args.world_size)
    current_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu))

    processes: List[Any] = []
    subprocess_file_handles = []

    for local_rank in range(args.world_size):
        dist_rank = args.rank_start + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)
        if args.gpu:
            current_env["DEVICE_ID"] = str(local_rank)
        else:
            current_env["DEVICE_ID"] = '-1'

        with_python = not args.no_python
        cmd = []
        if with_python:
            cmd = [sys.executable, "-u"]
            if args.module:
                cmd.append("-m")
        elif args.module:
            raise ValueError(
                "Don't use both the '--no_python' flag and the '--module' flag at the same time."
            )

        cmd.append(args.training_script)
        cmd.extend(args.training_script_args)

        sig_names = {2: "SIGINT", 15: "SIGTERM"}
        last_return_code = None

        # set killing flag to make sure killing signal only executed once
        kill_flag = True

        def sigkill_handler(signum, frame):
            nonlocal kill_flag
            if not kill_flag:
                return
            for process in processes:
                _logger.info(f"Killing subprocess {process.pid}")
            kill_flag = False
            try:
                # Note: use os.kill or process.kill() may only kill current process
                # use killpg will kill(use signal) this process and all sub-processes
                # if orphan sub-processes still exist, use signal.SIGKILL instead.
                os.killpg(os.getpid(), signal.SIGTERM)
            except Exception:
                pass
            if last_return_code is not None:
                raise subprocess.CalledProcessError(
                    returncode=last_return_code, cmd=cmd
                )
            if signum in sig_names:
                _logger.info(
                    f"Main process received {sig_names[signum]}, exiting")
            sys.exit(1)

        signal.signal(signal.SIGINT, sigkill_handler)
        signal.signal(signal.SIGTERM, sigkill_handler)

        # ic(current_env)
        process = subprocess.Popen(
            cmd, env=current_env,
        )
        processes.append(process)

    try:
        alive_processes = set(processes)
        while len(alive_processes):
            finished_processes = []
            for process in alive_processes:
                if process.poll() is None:
                    continue
                elif process.returncode != 0:
                    last_return_code = process.returncode
                    sigkill_handler(signal.SIGTERM, None)
                else:
                    finished_processes.append(process)
            alive_processes = set(alive_processes) - set(finished_processes)
            time.sleep(1)
    finally:
        pass
