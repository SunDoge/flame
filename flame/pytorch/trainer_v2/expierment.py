from pathlib import Path
from .base import Dataclass
import torch


class Experiment(Dataclass):

    def initialize(self, device: torch.device, work_dir: Path) -> int:
        pass
