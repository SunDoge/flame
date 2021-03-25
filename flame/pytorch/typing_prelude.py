from typing import Dict, List, NewType
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LrScheduler


TrainDataset = NewType('TrainDataset', Dataset)
ValDataset = NewType('ValDataset', Dataset)
TestDataset = NewType('TestDataset', Dataset)
TrainDataLoader = NewType('TrainDataLoader', DataLoader)
ValDataLoader = NewType('ValDataLoader', DataLoader)
TestDataLoader = NewType('TestDataLoader', DataLoader)

Criterion = NewType('Criterion', nn.Module)
Optimizer = Optimizer
LrScheduler = LrScheduler


class Model(nn.Module):

    module: nn.Module


TensorDict = Dict[str, Tensor]
TensorList = List[Tensor]

