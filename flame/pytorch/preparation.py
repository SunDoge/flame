"""
初始化主进程：

1. parse args
2. 根据 args 生成 experiment dir
3. 如果当前是主进程，创建 experiment dir
4. 如果当前是主进程，init logging and set logfile
5. 构建 root_container with Args, ExperimentDir

初始化多个进程：

1. 用户可以指定 nprocs，如果没有指定，用 device_count
2. 创建 n processes, 调用主进程初始化帮助函数，传入 device_id (process id)
3. 帮助函数init_process_group，设置 current_device，再调用 初始化主进程 方法

"""

from typing import Callable
from flame.argument import BasicArgs
import torch.distributed as dist
import flame
from .utils.distributed import is_dist_available_and_initialized
from injector import Injector, Binder
import functools
from dataclasses import dataclass
import logging
import torch.multiprocessing as mp
import torch



_logger = logging.getLogger(__name__)


@dataclass
class DistOptions:
    rank_start: int = 0
    world_size: int = 1
    dist_backend: str = 'NCCL'
    dist_url: str = 'tcp://127.0.0.1:12345'
    nprocs: int = 1

    def get_rank(self, proc_id: int) -> int:
        return self.rank_start + proc_id


def main_worker_helper(args: BasicArgs, main_worker: Callable):

    # 2. get experiment dir
    experiment_dir = flame.utils.experiment.get_experiment_dir(
        args.output_dir, args.experiment_name, debug=args.debug
    )

    # 3. make experiment dir if rank 0
    rank = 0
    if is_dist_available_and_initialized():
        rank = dist.get_rank()
        flame.utils.experiment.make_experiment_dir(
            experiment_dir, yes=args.yes
        )

    # 4. init logger depending on rank
    flame.logging.init_logger(
        rank=rank, filename=experiment_dir / 'experiment.log'
    )

    def _configure(binder: Binder):
        binder.bind(BasicArgs, to=args)


def distributed_helper(user_func: Callable, proc_id: int, dist_options: DistOptions):
    """
    传入一个user_func，user_func和单进程没有任何区别，函数功能和launch是一样的
    """

    rank = dist_options.get_rank(proc_id)

    dist.init_process_group(
        dist_options.dist_backend,
        init_method=dist_options.dist_url,
        world_size=dist_options.world_size,
        rank=rank,
    )

    if torch.cuda.is_available():
        _logger.info('set cuda device: %d', proc_id)
        torch.cuda.set_device(proc_id)

    user_func()


def spawn_process(func: Callable, dist_options: DistOptions):

    # get experiment dir

    if dist_options.rank_start == 0:
        # make dir
        pass

    mp.spawn(func, nprocs=dist_options.nprocs)
