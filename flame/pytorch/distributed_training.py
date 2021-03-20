import logging
from typing import Any, Callable

import torch.cuda as torch_cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from dataclasses import dataclass

_logger = logging.getLogger(__name__)


@dataclass
class DistOptions:
    rank_start: int = 0
    world_size: int = 1
    dist_backend: str = 'NCCL'
    dist_url: str = 'tcp://127.0.0.1:12345'


def _init_process_group_fn(device_id: int, worker_fn: Callable, dist_options: DistOptions, *args):
    rank = dist_options.rank_start + device_id
    print(f'=> rank: {rank}')

    dist.init_process_group(
        backend=dist_options.dist_backend,
        init_method=dist_options.dist_url,
        world_size=dist_options.world_size,
        rank=rank
    )

    if torch_cuda.is_available():
        _logger.info('set cuda_device=%d', device_id)
        torch_cuda.set_device(device_id)

    worker_fn(*args)


def start_distributed_training(
    worker_fn: Callable,
    args: tuple = (),
    rank_start: int = 0,
    world_size: int = 1,
    dist_backend: str = 'NCCL',
    dist_url: str = 'tcp://127.0.0.1:12345',
):
    """helper function for distributed training

    1. get number of GPUs
    2. start N process, N = number of GPUs
    3. init_process_group
    4. call worker_fn with *args

    """

    num_gpus = torch_cuda.device_count()

    if num_gpus == 0 and dist_backend.upper() == 'NCCL':
        _logger.error('no GPUs, stop training')
        return

    # CPU + GLOO, nprocs = 1
    nprocs = num_gpus if num_gpus > 0 else 1

    dist_options = DistOptions(
        rank_start=rank_start,
        world_size=world_size,
        dist_backend=dist_backend,
        dist_url=dist_url,
    )

    mp.spawn(
        _init_process_group_fn,
        args=(worker_fn, dist_options, *args),
        nprocs=nprocs
    )
