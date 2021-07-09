from torch.utils.data.distributed import DistributedSampler, Dataset

from typing import Iterable
import torch


class UniformDistributedSampler(DistributedSampler):

    def __iter__(self) -> Iterable[int]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # type: ignore[arg-type]
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        """
        return -1 if keep last
        """
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            indices += [-1] * padding_size
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def remove_padding(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return x[indices != -1]
