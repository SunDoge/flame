from typing import List, Union
import torch.distributed as dist
import torch
import functools
from torch.distributed import ReduceOp
from numbers import Number
from pathlib import Path
import logging
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

_logger = logging.getLogger(__name__)


def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank_safe() -> int:
    if is_dist_available_and_initialized():
        return dist.get_rank()
    else:
        return 0


def get_world_size_safe() -> int:
    if is_dist_available_and_initialized():
        return dist.get_world_size()
    else:
        return 1


@functools.lru_cache(maxsize=1)
def get_device_by_backend() -> torch.device:
    return torch.device('cuda' if dist.get_backend() == 'nccl' else 'cpu')


def reduce_numbers(nums: List[Union[int, float]], op=ReduceOp.SUM) -> List[Union[int, float]]:
    types = [type(n) for n in nums]
    floats = [float(n) for n in nums]

    device = get_device_by_backend()
    tensor = torch.tensor(floats, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=op)
    output = tensor.tolist()
    output = [t(o) for t, o in zip(types, output)]
    return output


def init_process_group_from_file(
    backend: str, filename: Union[Path, str], world_size: int = 1, rank: int = 0,
):
    """
    https://pytorch.org/docs/stable/distributed.html

    Args:
        backend: nccl or gloo
        filename: 用于init的file
        world_size: 或总节点数
        rank: 当前节点id
    """
    file_path = Path(filename)

    if file_path.exists():
        # file_path.unlink(missing_ok=True)
        file_path.unlink()  # Unfortunately, py37 does not support missing_ok

    uri = file_path.resolve().as_uri()

    _logger.info('init_method=%s', uri)

    dist.init_process_group(
        backend=backend,
        init_method=uri,
        world_size=world_size,
        rank=rank
    )


def num_valid_samples(num_samples: int, rank: int, num_replicas: int) -> int:
    '''
    Note: depends on the implementation detail of `DistributedSampler`
    Written by @huww98 
    '''
    return (num_samples - rank - 1) // num_replicas + 1


def num_valid_samples_from_data_loader(loader: DataLoader) -> int:
    sampler: DistributedSampler = loader.sampler
    assert sampler.shuffle == False, "DistributedSampler must not be shuffled"

    num_total = len(loader.dataset)

    num_valid = num_valid_samples(
        num_total,
        sampler.rank,
        sampler.num_replicas
    )
    _logger.info(
        f'{num_valid} valid samples of {len(sampler)} samples in rank {sampler.rank}, total samples: {num_total}'
    )
    return num_valid
