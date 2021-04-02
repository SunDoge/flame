

from typing import Any, Dict, Optional, Tuple
from torch import Tensor
from injector import inject, singleton
from .utils.attributes import get_device_from_module, get_dtype_from_module
from .typing_prelude import Model, Optimizer, Criterion, TensorDict
from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler
import torch
import logging

_logger = logging.getLogger(__name__)


class BaseProcess:

    def forward(self, batch: Any) -> Tuple[Tensor, Any]:
        """forward wrapper

        处理数据并计算loss，决定返回哪些数据。

        Return:
            loss: 第一个返回值必为loss
            output: 其他不需要backward的内容

        example::

            def forward(self, batch):
                image, label = batch
                pred = self.model(image)
                loss = self.criterion(pred, label)
                return loss, {'loss': loss, 'pred': pred, 'label': label}

        """
        pass

    def training_step(self, batch: Any) -> Any:
        pass

    def validation_step(self, batch: Any) -> Any:
        pass

    def update(self):
        pass

    def train(self, mode: bool = True):
        pass

    def eval(self):
        self.train(mode=False)


@singleton
class SupervisedProcess(BaseProcess):

    @inject
    def __init__(
        self,
        model: Model,
        optimizer: Optimizer,
        criterion: Criterion,
        grad_scaler: Optional[GradScaler] = None
    ) -> None:
        super().__init__()

        if grad_scaler is None:
            _logger.debug('Disable AMP')
            grad_scaler = GradScaler(enabled=False)

        use_amp = grad_scaler._enabled

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.use_amp = use_amp
        self.grad_scaler = grad_scaler

        self.device = get_device_from_module(model)
        self.dtype = get_dtype_from_module(model)

        _logger.info('AMP mode: %s', self.use_amp)

    def forward(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, TensorDict]:
        data, label = batch
        data = data.to(self.device, non_blocking=True)
        label = label.to(self.device, non_blocking=True)

        with autocast(enabled=self.use_amp):
            pred = self.model(data)
            loss = self.criterion(pred, label)

        output = {
            'pred': pred,
            'label': label,
            'loss': loss,
        }

        return loss, output

    def training_step(self, batch: Any) -> Any:
        loss, output = self.forward(batch)

        # self.optimizer.zero_grad()
        # self.grad_scaler.scale(loss).backward()
        # self.grad_scaler.step(self.optimizer)
        # self.grad_scaler.update()

        self.grad_scaler.scale(loss).backward()

        return output

    def validation_step(self, batch: Any) -> Any:
        with torch.no_grad():
            _loss, output = self.forward(batch)

        return output

    def update(self):
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.optimizer.zero_grad()

    def train(self, mode: bool = True):
        self.model.train(mode=mode)
