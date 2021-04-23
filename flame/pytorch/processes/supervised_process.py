from torch.cuda.amp import autocast
from torch.tensor import Tensor
from .process import Process
from .amp_process import AmpProcess
from flame.pytorch.typing_prelude import Model, Optimizer, Criterion
from injector import inject, singleton
from typing import Any, Tuple
from torch.cuda.amp.grad_scaler import GradScaler


@singleton
@inject
class SupervisedProcess(Process):

    def __init__(
        self,
        model: Model,
        optimizer: Optimizer,
        criterion: Criterion,
    ) -> None:
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.device = self.infer_device(model)
        self.dtype = self.infer_dtype(model)

    def forward(self, batch: Tuple[Tensor, Tensor]) -> dict:
        data, label = batch

        data = data.to(self.device, non_blocking=True)
        label = label.to(self.device, non_blocking=True)

        pred = self.model(data)
        loss = self.criterion(pred, label)
        batch_size = data.size(0)

        return self.output(
            loss=loss,
            batch_size=batch_size,
            pred=pred,
            label=label
        )

    def train(self, mode: bool = True):
        return self.model.train(mode=mode)

    def update(self):
        """
        为了兼容gradient accumulation，将zero_grad放最后
        """
        self.optimizer.step()
        self.optimizer.zero_grad()


@singleton
@inject
class SupervisedAmpProcess(AmpProcess):

    def __init__(
        self,
        model: Model,
        optimizer: Optimizer,
        criterion: Criterion,
        scaler: GradScaler
    ) -> None:
        super().__init__(scaler)

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.device = self.infer_device(model)
        self.dtype = self.infer_dtype(model)

    def forward(self, batch: Tuple[Tensor, Tensor]) -> dict:
        data, label = batch

        data = data.to(self.device, non_blocking=True)
        label = label.to(self.device, non_blocking=True)

        with autocast(enabled=self.is_amp_enabled()):
            pred = self.model(data)
            loss = self.criterion(pred, label)

        batch_size = data.size(0)

        return self.output(
            loss=loss,
            batch_size=batch_size,
            pred=pred,
            label=label
        )

    def train(self, mode: bool = True):
        return self.model.train(mode=mode)

    def update(self):
        """
        为了兼容gradient accumulation，将zero_grad放最后
        """
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
