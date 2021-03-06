from pathlib import Path
from typing import Dict, List, NewType

import torch
from flame.argument import BasicArgs
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LrScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


TrainDataset = NewType('TrainDataset', Dataset)
ValDataset = NewType('ValDataset', Dataset)
TestDataset = NewType('TestDataset', Dataset)

TrainSampler = NewType('TrainSampler', DistributedSampler)
ValSampler = NewType('ValSampler', DistributedSampler)
TestSampler = NewType('TestSampler', DistributedSampler)

TrainLoader = NewType('TrainLoader', DataLoader)
ValLoader = NewType('ValLoader', DataLoader)
TestLoader = NewType('TestLoader', DataLoader)

BasicArgs = BasicArgs
RootConfig = NewType('RootConfig', dict)
ExperimentDir = NewType('ExperimentDir', Path)

Criterion = NewType('Criterion', nn.Module)
Optimizer = Optimizer
LrScheduler = LrScheduler

Device = NewType('Device', torch.device)
Dtype = NewType('Dtype', torch.dtype)
Rank = NewType('Rank', int)
LocalRank = NewType('LocalRank', int)

# Trainer = NewType('Trainer', Engine)
# Evaluator = NewType('Evaluator', Engine)


class Model(nn.Module):
    """
    模拟 DataParallel 和 DistributedDataParallel
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


TensorDict = Dict[str, Tensor]
TensorList = List[Tensor]
