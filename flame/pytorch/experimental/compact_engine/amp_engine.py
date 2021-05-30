from .engine import BaseEngine
from torch.cuda.amp.grad_scaler import GradScaler
from torch import nn, Tensor


class AmpEngine(BaseEngine):

    scaler: GradScaler

    def training_step(self, batch, batch_idx: int) -> dict:
        output = self.forward(batch, batch_idx)
        loss: Tensor = output['loss']
        self.scaler.scale(loss).backward()

    def clip_grad_norm_if_needed(self):
        if self.cfg.max_norm > 0.0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.max_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def update(self):
        self.scaler.step(self.optimizer)
        self.scaler.update()
