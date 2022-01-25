from flame.pytorch.sampler import DistributedNoPaddingSampler
from torch.utils.data.distributed import DistributedSampler
import pytest


@pytest.mark.parametrize(
    "total_size,world_size",
    [(13, 2), (13, 4)]
)
def test_distributed_no_padding_sampler(
    total_size: int,
    world_size: int,
):

    dataset = list(range(total_size))

    dist_samplers = [
        DistributedSampler(dataset, num_replicas=world_size,
                           rank=i, shuffle=False)
        for i in range(world_size)
    ]
    dist_no_padding_samplers = [
        DistributedNoPaddingSampler(dataset, num_replicas=world_size, rank=i)
        for i in range(world_size)
    ]
    dist_indices = [list(s) for s in dist_samplers]
    dist_no_padding_indices = [list(s) for s in dist_no_padding_samplers]

    for xi, yi in zip(dist_indices, dist_no_padding_indices):
        assert len(yi) <= len(xi)

        if len(yi) == len(xi):
            assert yi == xi
        elif len(yi) < len(xi):
            """
            dist: [1, 3, 5, 7, 9, 11, 0]
            dist_no_padding: [1, 3, 5, 7, 9, 11]
            """

            last_index = len(yi) - 1
            assert yi == xi[:len(yi)]
            assert xi[last_index + 1] < xi[last_index]
