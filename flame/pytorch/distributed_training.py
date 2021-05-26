import logging
from dataclasses import dataclass
from typing import Callable, Optional

import flame
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import typed_args as ta
from flame.utils.operating_system import find_free_port
from flame.argument import BasicArgs

_logger = logging.getLogger(__name__)


@dataclass
class DistOptions(BasicArgs):
    # dist: bool = ta.add_argument(
    #     '--dist', action='store_true',
    #     help='activate distributed mode'
    # )
    rank_start: int = ta.add_argument(
        '--rank-start', type=int, default=0,
        help='当前 node 的 rank 起始值'
    )
    world_size: Optional[int] = ta.add_argument(
        '-w', '--world-size', type=int,
        help='如果不指定，自动计算'
    )
    dist_backend: str = ta.add_argument(
        '--dist-backend', type=str, default='nccl',
        choices=['nccl', 'gloo'],
    )
    dist_host: str = ta.add_argument(
        '--dist-host', type=str, default='127.0.0.1',
    )
    dist_port: Optional[int] = ta.add_argument(
        '--dist-port', type=int,
    )
    nprocs: Optional[int] = ta.add_argument(
        '--nprocs', type=int,
        help='number of processes'
    )

    @property
    def dist_url(self) -> str:
        assert self.dist_port
        return 'tcp://{host}:{port}'.format(host=self.dist_host, port=self.dist_port)

    def get_rank(self, local_rank: int) -> int:
        return self.rank_start + local_rank

    def find_free_port(self):
        if self.dist_port is None:
            self.dist_port = find_free_port()
            _logger.info('Find a free port: %s', self.dist_port)


def get_dist_options() -> DistOptions:

    dist_options, _ = DistOptions.from_known_args()
    dist_options: DistOptions

    if dist_options.dist_port is None:
        dist_options.dist_port = str(find_free_port())
        _logger.info('Find a free port: %s', dist_options.dist_port)

    return dist_options


def _init_process_group_fn(local_rank: int, worker_fn: Callable, dist_options: DistOptions, *args):
    """wrapper function for worker_fn

    必须定义成可以被pickle的函数。

    1. compute rank
    2. init_process_group
    3. set cuda device if cuda is available
    4. call worker_fn

    """

    print('start distributed training')
    rank = dist_options.get_rank(local_rank)
    print(f'=> rank: {rank}')

    dist.init_process_group(
        backend=dist_options.dist_backend,
        init_method=dist_options.dist_url,
        world_size=dist_options.world_size,
        rank=rank
    )

    print('init process group')

    if torch.cuda.is_available():
        # _logger.info('set cuda_device=%d', local_rank)
        print('set cuda device=', local_rank)
        torch.cuda.set_device(local_rank)

    worker_fn(local_rank, *args)


def start_distributed_training(
    worker_fn: Callable,
    dist_options: DistOptions,
    args: tuple = (),
):
    """helper function for distributed training

    1. get number of GPUs
    2. start N process, N = number of GPUs
    3. init_process_group
    4. call worker_fn with args

    """

    # CPU + GLOO, nprocs = 1
    # nprocs = num_gpus if num_gpus > 0 else 1
    if dist_options.nprocs is None:
        _logger.info('nprocs is None, start inferring nprocs')
        if dist_options.dist_backend.lower() == 'nccl':
            dist_options.nprocs = torch.cuda.device_count()

            if dist_options.nprocs == 0:
                _logger.error('no gpu for distributed training, stop')
                return
        else:
            dist_options.nprocs = 1

        _logger.info('change world_size: %s => %d',
                     dist_options.world_size, dist_options.nprocs)
        dist_options.world_size = dist_options.nprocs
        _logger.info('nprocs = %d', dist_options.nprocs)

    # 根据nprocs决定是否启动multiprocess
    if dist_options.nprocs == 1:
        _logger.debug('nprocs==1, disable multiprocessing')
        _init_process_group_fn(0, worker_fn, dist_options, *args)
    else:
        _logger.debug('nprocs==%d, enable multiprocessing',
                      dist_options.nprocs)
        mp.spawn(
            _init_process_group_fn,
            args=(worker_fn, dist_options, *args),
            nprocs=dist_options.nprocs
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


def start_training(worker_fn: Callable, args: tuple = ()):
    flame.logging.init_logger()
    dist_options = get_dist_options()
    # if dist_options.dist:
    #     start_distributed_training(
    #         worker_fn, dist_options, args=args)
    # else:
    #     worker_fn(0, *args)
    start_distributed_training(
        worker_fn, dist_options, args=args
    )
