import logging
from typing import Tuple, Any
from torch import Tensor
import torch
from flame.pytorch.utils.attributes import get_dtype_from_module, get_device_from_module
from torch import nn

_logger = logging.getLogger(__name__)


class Process:

    def forward(self, batch: Any) -> dict:
        """forward wrapper

        处理数据并计算loss，决定返回哪些数据。

        Return:
            output: 必须包含loss和batch_size

        example::

            def forward(self, batch):
                image, label = batch
                pred = self.model(image)
                loss = self.criterion(pred, label)
                return self.output(loss=loss, batch_size=labe.size(0), pred=pred, label=label)

        """
        pass

    def training_step(self, batch: Any) -> dict:
        output = self.forward(batch)
        loss: Tensor = output['loss']
        loss.backward()
        return output

    def validation_step(self, batch: Any) -> dict:
        with torch.no_grad():
            output = self.forward(batch)
        return output

    def update(self):
        pass

    def train(self, mode: bool = True):
        pass

    def eval(self):
        self.train(mode=False)

    @staticmethod
    def output(loss: Tensor = None, batch_size: int = None, **kwargs) -> dict:
        output = {
            'loss': loss,
            'batch_size': batch_size,
            **kwargs
        }
        return output

    @staticmethod
    def infer_dtype(module: nn.Module) -> torch.dtype:
        """
        通过model推断dtype
        """
        return get_dtype_from_module(module)

    @staticmethod
    def infer_device(module: nn.Module) -> torch.device:
        """
        通过model推断device
        """
        return get_device_from_module(module)
