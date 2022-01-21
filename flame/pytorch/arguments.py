from flame.core.arguments import BaseArgs as Base
import torch

class BaseArgs(Base):

    @property
    def device(self) -> torch.device:
        if len(self.gpu) == 0:
            pass
