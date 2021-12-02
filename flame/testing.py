import torch.distributed as dist
import logging

_logger = logging.getLogger(__name__)

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