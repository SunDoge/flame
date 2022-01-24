import logging

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

_logger = logging.getLogger(__name__)


def create_ddp_model(
    base_model: nn.Module,
    device: torch.device,
    use_sync_bn: bool = False,
    find_unused_parameters: bool = False,
) -> nn.Module:
    base_model.to(device)

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
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    return model
