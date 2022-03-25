from typing import Optional

import oneflow.env
from oneflow.utils.data.dataset import Dataset
from oneflow.utils.data.sampler import Sampler


class DistributedNoPaddingSampler(Sampler):

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None
    ) -> None:
        if num_replicas is None:
            # if not dist.is_available():
            #     raise RuntimeError(
            #         "Requires distributed package to be available")
            num_replicas = oneflow.env.get_world_size()
        if rank is None:
            # if not dist.is_available():
            #     raise RuntimeError(
            #         "Requires distributed package to be available")
            rank = oneflow.env.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        total_size = len(dataset)
        num_samples = num_valid_samples(total_size, rank, num_replicas)

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = num_samples
        self.total_size = total_size

    def __iter__(self):
        indices = list(range(self.total_size))
        sub_indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(sub_indices) == self.num_samples
        return iter(sub_indices)

    def __len__(self) -> int:
        return self.num_samples


def num_valid_samples(num_samples: int, rank: int, num_replicas: int) -> int:
    '''
    Note: depends on the implementation detail of `DistributedSampler`
    Written by @huww98 
    '''
    return (num_samples - rank - 1) // num_replicas + 1
