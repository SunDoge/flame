from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import logging
import torch

_logger = logging.getLogger(__name__)


def create_data_loader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    collate_fn=None,
    pin_memory: bool = True,
    multiprocessing_context=None,
    persistent_workers: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset)

        if shuffle == True:
            _logger.debug('set shuffle=False when using DistributedSampler')
            shuffle = False
    else:
        sampler = None

    if multiprocessing_context is None:
        # 在linux下更快更好，不会在每次启动时重新读取文件
        multiprocessing_context = mp.get_context('fork')
        _logger.debug('set mp context: %s', multiprocessing_context)

    if pin_memory and persistent_workers:
        torch_version: str = torch.__version__
        major_version, minor_version, patch_version = map(
            int, torch_version.split('+')[0].split('.')
        )
        assert major_version >= 1 and minor_version >= 8, "pytorch version lower than 1.8 has bug for pin_menory and persistent_workers"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        multiprocessing_context=multiprocessing_context,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )

    return loader
