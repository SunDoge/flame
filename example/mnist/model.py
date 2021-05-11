
from flame.pytorch.meters.average_meter import AverageMeter, AverageMeterGroup
from flame.pytorch.experimental.compact_engine.engine import BaseEngine, BaseEngineConfig
from typing import Callable, Tuple
from torch import nn, Tensor
from torch.cuda.amp.grad_scaler import GradScaler
import torch.nn.functional as F
import torch
from flame.pytorch.experimental.compact_engine.amp_engine import AmpEngine
from injector import inject
from flame.pytorch.typing_prelude import Device, Model, Optimizer, LrScheduler, Criterion
import logging
from flame.pytorch.metrics.functional import topk_accuracy

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
    cfg: BaseEngineConfig

    def __init__(
        self,
        model: Model,
        optimizer: Optimizer,
        criterion: Criterion,
        scaler: GradScaler,
        device: Device,
        cfg: BaseEngineConfig,
    ):
        kwargs = locals()
        kwargs.pop('self')
        super().__init__(**kwargs)

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

        if self.every_n_steps(self.cfg.log_interval):
            _logger.info(
                f'{self.state.mode}\t{self.meters}'
            )

        return self.output(
            loss=loss,
            batch_size=batch_size,
        )
