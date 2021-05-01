from .process import Process
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast
from injector import inject, singleton
import logging
from typing import Any
from torch import Tensor

_logger = logging.getLogger(__name__)


@singleton
@inject
class AmpProcess(Process):

    def __init__(
        self,
        scaler: GradScaler
    ) -> None:
        super().__init__()
        self.scaler = scaler
        _logger.info('Amp mode: %s', self.scaler.is_enabled())

    def training_step(self, batch: Any) -> Any:
        output = self.forward(batch)
        loss: Tensor = output['loss']
        self.scaler.scale(loss).backward()

        return output

    def is_amp_enabled(self) -> bool:
        return self.scaler.is_enabled()

    def autocast(self):
        return autocast(self.is_amp_enabled)
