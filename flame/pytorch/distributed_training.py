import logging
from typing import Any, Callable, Optional

import torch.cuda as torch_cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from dataclasses import dataclass
import flame
import argparse
import functools
from flame.utils.experiment import get_experiment_dir, make_experiment_dir

_logger = logging.getLogger(__name__)


@dataclass
class DistOptions:
    rank_start: int = 0
    world_size: int = 1
    dist_backend: str = 'nccl'
    dist_host: str = '127.0.0.1'
    dist_port: int = 12345

    @property
    def dist_url(self) -> str:
        return 'tcp://{host}:{port}'.format(host=self.dist_host, port=self.dist_port)

    def get_rank(self, proc_id: int) -> int:
        return self.rank_start + proc_id


def get_dist_options() -> DistOptions:
    parser = argparse.ArgumentParser(prog='distributed launcher')
    parser.add_argument('--rank-start', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--dist-backend', type=str, default='nccl')
    parser.add_argument('--dist-host', type=str, default='127.0.0.1')
    parser.add_argument('--dist-port', type=int, default=12345)

    args, _ = parser.parse_known_args()
    dist_options = DistOptions(args.__dict__)
    return dist_options


def _init_process_group_fn(proc_id: int, worker_fn: Callable, dist_options: DistOptions, *args):
    """wrapper function for worker_fn

    必须定义成可以被pickle的函数。

    1. compute rank
    2. init_process_group
    3. set cuda device if cuda is available
    4. call worker_fn

    """

    rank = dist_options.get_rank(proc_id)
    print(f'=> rank: {rank}')

    dist.init_process_group(
        backend=dist_options.dist_backend,
        init_method=dist_options.dist_url,
        world_size=dist_options.world_size,
        rank=rank
    )

    if torch_cuda.is_available():
        _logger.info('set cuda_device=%d', proc_id)
        torch_cuda.set_device(proc_id)

    worker_fn(*args)


def start_distributed_training(
    worker_fn: Callable,
    dist_options: DistOptions,
    args: tuple = (),
    nprocs: Optional[int] = None,
):
    """helper function for distributed training

    1. get number of GPUs
    2. start N process, N = number of GPUs
    3. init_process_group
    4. call worker_fn with args

    """

    # CPU + GLOO, nprocs = 1
    # nprocs = num_gpus if num_gpus > 0 else 1

    if nprocs is None:
        _logger.info('nprocs is None, start inferring nprocs')
        if dist_options.dist_backend.lower() == 'nccl':
            nprocs = torch_cuda.device_count()
            if nprocs == 0:
                _logger.error('no gpu for distributed training, stop')
                return
        else:
            nprocs = 1

        _logger.info('nprocs = %d', nprocs)

    mp.spawn(
        _init_process_group_fn,
        args=(worker_fn, dist_options, *args),
        nprocs=nprocs
    )


def get_available_local_dist_url() -> str:
    """helper function for single node distributed training

    Get a local dist url like::

        tcp://127.0.0.1:12345


    """

    port = flame.utils.operating_system.find_free_port()
    return f'tcp://127.0.0.1:{port}'


def init_cpu_process_group(
    rank: int = 0,
    world_size: int = 1,
    dist_url: str = 'tcp://127.0.0.1:12345',
):
    """helper function for testing distributed training

    测试函数，方便在CPU环境下测试分布式操作，比如测试MoCo的shuffle bn。
    不要在生产环境中使用。

    """
    _logger.warning(
        'this function is for testing only, do not use it for production'
    )
    dist.init_process_group(
        backend='GLOO',
        init_method=dist_url,
        rank=rank,
        world_size=world_size,
    )
