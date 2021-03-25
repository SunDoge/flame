from torch import nn
from torch.optim import Optimizer
import torch
from torch import Tensor


def get_first_parameter_or_buffer(module: nn.Module) -> Tensor:
    """Get first parameter or buffer from module


    """

    try:
        first = next(module.parameters())
        return first
    except StopIteration:
        try:
            first = next(module.buffers())
            return first
        except StopIteration:
            raise Exception(f'no parameter or buffer in module: {module}')


def get_device_from_module(module: nn.Module) -> torch.device:
    return get_first_parameter_or_buffer(module).device


def get_dtype_from_module(module: nn.Module) -> torch.dtype:
    return get_first_parameter_or_buffer(module).dtype


def get_learning_rate_from_optimizer(optimizer: Optimizer) -> float:
    """Get learning rate from optimizer

    """
    return optimizer.param_groups[0]['lr']
