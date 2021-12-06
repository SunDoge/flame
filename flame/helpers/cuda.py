from torch.backends import cudnn
import torch
import logging

_logger = logging.getLogger(__name__)


def cudnn_benchmark_if_possible():
    if torch.cuda.is_available() and cudnn.is_available():
        cudnn.benchmark = True
        _logger.info('cudnn.benchmark=%s', cudnn.benchmark)
