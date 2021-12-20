from typing import Any, Callable

import icecream
from flame.arguments import BaseArgs
from flame.distributed_training import start_distributed_training
from icecream import ic
from torchvision import transforms as T
from torch.utils.data.dataloader import DataLoader
import flame
import torch
from flame.experimental.trainer import BaseTrainer
import torch.nn.functional as F

Args = BaseArgs


@flame.main_fn
def main():
    args = Args.from_args(["-c", "example/mnist/pytorch_example_v3.jsonnet", "-d"])
    start_distributed_training(args)


class Trainer(BaseTrainer):
    def __init__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        max_epochs: int,
        **kwargs
    ) -> None:
        super().__init__()

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.max_epochs = max_epochs

        self.run()

    def run(self):

        while self.state.epoch < self.max_epochs:
            self.train(self.train_loader)
            self.test(self.test_loader)
            print(self.state.epoch)

    def forward(self, batch, batch_idx: int, prefix: str):
        # ic(batch)
        # ic(prefix, batch_idx)
        yield
        # ic(prefix, batch_idx)
