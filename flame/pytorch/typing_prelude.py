from pathlib import Path
from typing import Dict, List, NewType
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LrScheduler
# from .engine import Engine
# from .experimental.engine import Engine
import torch
from .engine import EpochState, IterationState
from flame.argument import BasicArgs
from torch.utils.data.sampler import Sampler

TrainDataset = NewType('TrainDataset', Dataset)
ValDataset = NewType('ValDataset', Dataset)
TestDataset = NewType('TestDataset', Dataset)

TrainSampler = NewType('TrainSampler', Sampler)
ValSampler = NewType('ValSampler', Sampler)
TestSampler = NewType('TestSampler', Sampler)

TrainLoader = NewType('TrainLoader', DataLoader)
ValLoader = NewType('ValLoader', DataLoader)
TestLoader = NewType('TestLoader', DataLoader)

RootConfig = NewType('RootConfig', dict)
ExperimentDir = NewType('ExperimentDir', Path)

Criterion = NewType('Criterion', nn.Module)
Optimizer = Optimizer
LrScheduler = LrScheduler

Device = NewType('Device', torch.device)

# Trainer = NewType('Trainer', Engine)
# Evaluator = NewType('Evaluator', Engine)


TrainState = NewType('TrainState', IterationState)
ValState = NewType('ValState', IterationState)
TestState = NewType('TestState', IterationState)


class Model(nn.Module):

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


TensorDict = Dict[str, Tensor]
TensorList = List[Tensor]
