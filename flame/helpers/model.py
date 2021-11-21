from typing import Optional
from torch import nn, Tensor
from torch.nn.parallel import DistributedDataParallel, DataParallel

import torch
import torch.distributed as dist
import logging
from pygtrie import CharTrie
from flame.config_parser import ConfigParser


_logger = logging.getLogger(__name__)


class Model(nn.Module):

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def create_model(
    base_model: nn.Module,
    device: torch.device,
    # local_rank: Optional[int] = None,
    # use_sync_bn: bool = False,
    find_unused_parameters: bool = False,
) -> Model:
    """
    TODO
    """
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

    #     if use_sync_bn:
    #         _logger.info('convert bn of %s to sync bn', model.module.__class__)
    #         model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        #     assert use_sync_bn == False, 'sync bn can only be used in distributed mode'
        model = DataParallel(base_model)

    return model


def create_model_from_config(config: dict, find_unused_parameters: bool = False) -> Model:
    """
    TODO: 应该拆分成两个函数
    """
    config_parser = ConfigParser()
    base_model = config_parser.parse(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(base_model, device,
                         find_unused_parameters=find_unused_parameters)
    return model
