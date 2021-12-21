from typing import Any, Callable, Sequence

import icecream
from torch.functional import Tensor
from flame.arguments import BaseArgs
from flame.distributed_training import start_distributed_training
from icecream import ic
from torchvision import transforms as T
from torch.utils.data.dataloader import DataLoader
import flame
import torch
from flame.experimental.trainer import BaseTrainer
import torch.nn.functional as F
from torch import nn
import functools
from flame import helpers
from flame.experimental.trainer.data_module import DataModule
from flame.pytorch.meters.average_meter import DynamicAverageMeterGroup
import logging

_logger = logging.getLogger(__name__)

Args = BaseArgs


@flame.main_fn
def main():
    args = Args.from_args(["-c", "example/mnist/pytorch_example_v3.jsonnet", "-d"])
    start_distributed_training(args)


@torch.no_grad()
def accuracy(output: Tensor, target: Tensor) -> Tensor:
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum()
    batch_size = target.size(0)
    return correct / batch_size * 100.0


class Trainer(BaseTrainer):
    def __init__(
        self,
        data_module: DataModule,
        max_epochs: int,
        model: nn.Module,
        optimizer_fn: functools.partial,
        scheduler_fn: functools.partial,
        args: Args,
        log_interval: int,
        **kwargs,
    ) -> None:
        super().__init__()

        model = helpers.model.create_model(model, args.device)

        helpers.scale_lr_for_partial(optimizer_fn)
        optimizer = optimizer_fn(model.parameters())
        scheduler = scheduler_fn(optimizer)

        self.checkpoint_manager.register_model(model)
        self.checkpoint_manager.register_optimizer(optimizer)
        self.checkpoint_manager.register_lr_scheduler(scheduler)

        self.max_epochs = max_epochs
        self.device = args.device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_interval = log_interval
        self.experiment_dir = args.experiment_dir

        self.run(data_module, max_epochs)

    def forward(self, batch: Sequence[Tensor], batch_idx: int, prefix: str):
        data, target = map(lambda x: x.to(self.device, non_blocking=True), batch)

        output = self.model(data)
        loss = F.nll_loss(output, target)

        if self.state.training:
            self.backward(loss, self.optimizer)

        acc = accuracy(output, target)

        batch_size = target.size(0)

        yield batch_size
        # ic(prefix, batch_idx)

        self.meters.update(prefix, "acc", acc.item(), batch_size)
        self.meters.update(prefix, "loss", loss.item(), batch_size)

        if self.every_n_iters(batch_idx, n=self.log_interval):
            _logger.info(f"{prefix} {self.iter_eta} {self.meters.with_prefix(prefix)}")

    def epoch_middleware(self, next_fn: Callable):
        next_fn()
        self.scheduler.step()
        helpers.checkpoint_saver.save_checkpoint(
            self.checkpoint_manager.state_dict(),
            self.experiment_dir,
            is_best=self.meters["test"]["acc"].is_best(),
        )
