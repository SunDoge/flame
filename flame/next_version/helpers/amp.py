from typing import Iterable, Optional, Union
import torch
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.functional import Tensor
from torch.optim import Optimizer

TensorOrIterableTensors = Union[Tensor, Iterable[Tensor]]


class Amp:

    def __init__(
        self,
        enabled: bool = False,
        max_norm: Optional[float] = None,
    ) -> None:
        self.grad_scaler = GradScaler(enabled=enabled)
        self.enabled = enabled
        self.max_norm = max_norm

    def autocast(self):
        return autocast(enabled=self.enabled)

    def scale(self, outputs: TensorOrIterableTensors) -> TensorOrIterableTensors:
        return self.grad_scaler.scale(outputs)

    def unscale_(self, optimizer: Optimizer):
        return self.grad_scaler.unscale_(optimizer)

    def step(self, optimizer: Optimizer, *args, **kwargs):
        return self.grad_scaler.step(optimizer, *args, **kwargs)

    def update(self, new_scale: Union[float, Tensor, None] = None):
        return self.grad_scaler.update(new_scale=new_scale)

    def clip_grad_norm_(self, params: TensorOrIterableTensors):
        if self.max_norm:
            torch.nn.utils.clip_grad_norm_(
                params, self.max_norm
            )