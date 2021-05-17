
from flame.pytorch.meters.time_meter import EstimatedTimeOfArrival
import logging
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from example.mnist.config import Config
from flame.pytorch.container import CallableAssistedBuilder
from flame.pytorch.experimental.compact_engine.amp_engine import AmpEngine
# from flame.pytorch.experimental.compact_engine.engine import (BaseEngine,
#                                                               BaseEngineConfig)

from flame.pytorch.experimental.compact_engine.engine_v2 import BaseEngine, State
from flame.pytorch.meters.average_meter import AverageMeter, AverageMeterGroup
from flame.pytorch.metrics.functional import topk_accuracy
from flame.pytorch.typing_prelude import (Criterion, Device, LrScheduler,
                                          Model, Optimizer)
from injector import ClassAssistedBuilder, inject
from torch import Tensor, nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data.dataloader import DataLoader

_logger = logging.getLogger(__name__)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


@inject
class NetEngine(BaseEngine):

    device: Device
    criterion: Callable[[Tensor, Tensor], Tensor]

    def __init__(
        self,
        state: State,
        model: Model,
        optimizer: Optimizer,
        criterion: Criterion,
        # scaler: GradScaler,
        device: Device,
        data_loader_builder: CallableAssistedBuilder[DataLoader],
        cfg: dict,
    ):
        super().__init__(
            # model=model,
            # optimizer=optimizer,
            # criterion=criterion,
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            state=state
        )

        # self.scaler = scaler
        self.device = device
        self.data_loader_builder = data_loader_builder
        self.criterion = criterion

        self.meters = AverageMeterGroup({
            'loss': AverageMeter('loss'),
            'acc1': AverageMeter('acc1'),
            'acc5': AverageMeter('acc5')
        })

    def forward(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> dict:
        data, target = map(
            lambda x: x.to(self.device, non_blocking=True),
            batch
        )
        pred = self.model(data)
        loss = self.criterion(pred, target)

        batch_size = data.size(0)

        acc1, acc5 = topk_accuracy(
            pred, target, topk=(1, 5)
        )
        self.meters.update({
            'loss': loss.item(),
            'acc1': acc1.item(),
            'acc5': acc5.item(),
        }, n=batch_size)

        self.step_eta.update(batch_size)
        if self.every_n_steps(self.cfg.print_freq):
            _logger.info(
                f'{self.step_eta}\t{self.meters}'
            )

        return self.output(
            loss=loss,
            batch_size=batch_size,
        )

    def run(self):
        train_loader = self.data_loader_builder.build(split='train')
        val_loader = self.data_loader_builder.build(split='val')

        self.epoch_eta = EstimatedTimeOfArrival(self.cfg.max_epochs)
        while self.unfinished():
            self.train(train_loader)
            self.validate(val_loader)
            self.epoch_eta.update()

            _logger.info(f'epoch complete {self.epoch_eta}\t')
