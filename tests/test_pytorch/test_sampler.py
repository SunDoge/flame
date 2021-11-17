from typing import List

import torch
from flame.pytorch.sampler import UniformDistributedSampler, remove_padding
import math


def get_dataset(n: int = 11) -> List[int]:
    return list(range(n))


def test_uniform_distributed_sampler():
    N = 11
    ds = get_dataset(n=N)
    sampler0 = UniformDistributedSampler(
        ds, num_replicas=2, rank=0, shuffle=False)
    sampler1 = UniformDistributedSampler(
        ds, num_replicas=2, rank=1, shuffle=False)

    assert len(sampler0) == math.ceil(N / 2)
    assert len(sampler1) == math.ceil(N / 2)

    assert list(sampler0) == list(range(N))[::2]
    assert list(sampler1) == [1, 3, 5, 7, 9, -1]


def test_remove_padding():
    indices = torch.tensor([0, 1, -1, 3, 4])
    x = torch.rand(5, 2)
    y = remove_padding(x, indices)
    assert y.size(0) == 4

    # 应该要和去掉中间的一样
    y1 = torch.cat([x[:2], x[3:]], dim=0)
    assert torch.allclose(y, y1)
