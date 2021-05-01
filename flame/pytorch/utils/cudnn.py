from torch.backends import cudnn
import torch
import logging

_logger = logging.getLogger(__name__)


def set_cudnn_benchmark():
    if torch.cuda.is_available():
        cudnn.benchmark = True
        _logger.info('cudnn.benchmark = %s', cudnn.benchmark)
    else:
        _logger.warning('CUDA is not available, fail to set cudnn.benchmark')
