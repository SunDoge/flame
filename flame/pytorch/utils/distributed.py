from typing import List
import torch.distributed as dist
import torch
import functools
from torch.distributed import ReduceOp
from numbers import Number


def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


@functools.lru_cache(maxsize=1)
def get_device_by_backend() -> torch.device:
    return torch.device('cuda' if dist.get_backend() == 'nccl' else 'cpu')


def reduce_numbers(nums: List[Number], op=ReduceOp.SUM) -> List[Number]:
    types = [type(n) for n in nums]
    floats = [float(n) for n in nums]

    device = get_device_by_backend()
    tensor = torch.tensor(floats, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=op)
    output = tensor.tolist()
    output = [t(o) for t, o in zip(types, output)]
    return output
