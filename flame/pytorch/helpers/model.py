from typing import Optional
from torch import nn, Tensor
from torch.nn.parallel import DistributedDataParallel, DataParallel

import torch
import torch.distributed as dist
import logging

_logger = logging.getLogger(__name__)


def create_model(
    base_model: nn.Module,
    device: torch.device,
    # local_rank: Optional[int] = None,
    use_sync_bn: bool = False,
    find_unused_parameters: bool = False,
) -> nn.Module:
    base_model.to(device)

    if dist.is_available() and dist.is_initialized():
        # assert local_rank is not None, 'must pass in local_rank in distributed mode'
        if device.type == 'cuda':
            model = DistributedDataParallel(
                base_model,
                device_ids=[device],
                find_unused_parameters=find_unused_parameters
            )
        else:
            model = DistributedDataParallel(
                base_model,
                find_unused_parameters=find_unused_parameters
            )

        if use_sync_bn:
            _logger.info('convert bn of %s to sync bn', model.module.__class__)
            model=nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        assert use_sync_bn == False, 'sync bn can only be used in distributed mode'
        model=DataParallel(base_model)

    return model
