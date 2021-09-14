from typing import NewType
from flame.next_version.arguments import BaseArgs
from torch import nn

IConfig = NewType('IConfig', dict)
IArgs = NewType('IArgs', BaseArgs)
IModel = NewType('IModel', nn.parallel.DistributedDataParallel)
