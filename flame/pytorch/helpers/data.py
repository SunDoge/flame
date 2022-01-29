import logging
from typing import Callable, Optional

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from flame.pytorch.sampler import DistributedNoPaddingSampler

_logger = logging.getLogger(__name__)


def create_data_loader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn=None,
    pin_memory: bool = True,
    multiprocessing_context=None,
    persistent_workers: bool = True,
    drop_last: bool = False,
    worker_init_fn: Optional[Callable[[int], None]] = None,
) -> DataLoader:
    if dist.is_available() and dist.is_initialized():
        if shuffle:
            sampler = DistributedSampler(dataset)
        else:
            _logger.info('Using DistributedNoPaddingSampler')
            sampler = DistributedNoPaddingSampler(dataset)
    else:
        sampler = None

    if multiprocessing_context is None:
        # 在linux下更快更好，不会在每次启动时重新读取文件
        multiprocessing_context = mp.get_context('fork')
        _logger.debug('set mp context: %s', multiprocessing_context)

    if num_workers == 0:
        _logger.warning(
            'num_workers=0, disable persistent workers and multiprocess context'
        )
        persistent_workers = False
        multiprocessing_context = None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        multiprocessing_context=multiprocessing_context,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
    )

    return loader
