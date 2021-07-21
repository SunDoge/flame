from flame.pytorch.experimental.metrics.functional import topk_accuracy
import functools
from torch.functional import Tensor
from flame.pytorch.experimental.compact_engine.engine_v3 import BaseEngine, Stage, BaseEngineConfig, BaseState
from injector import inject
from torch.utils.tensorboard import SummaryWriter
from flame.pytorch.typing_prelude import Criterion, Device, Model, Optimizer
from pfun import Effect, success
from typing import Tuple, Any
import logging

_logger = logging.getLogger(__name__)


class ClassificationState(BaseState):
    model: Model
    optimizer: Optimizer

    # def train(self, mode: bool):
    #     return self.model.train(mode=mode)


@inject
class ClassificationEngine(BaseEngine):

    def __init__(
        self,
        summary_writer: SummaryWriter,
        config: BaseEngineConfig,
        device: Device,
        criterion: Criterion,

    ) -> None:
        super().__init__(config)

        self.device = device
        self.criterion = criterion
        self.summary_writer = summary_writer

    def forward(self, state: ClassificationState, batch: Any) -> Tuple[dict, Effect]:
        _logger.debug('state: ')

        image, label = batch
        image = image.to(self.device, non_blocking=True)
        label = label.to(self.device, non_blocking=True)

        pred = state.model(image)

        loss = self.criterion(pred, label)

        acc1, acc5 = topk_accuracy(pred, label, topk=(1, 5))
        batch_size = len(label)

        def update_meters(state: ClassificationState):
            state.meters.update('loss', loss.item(), batch_size)
            state.meters.update('acc1', acc1.item(), batch_size)
            state.meters.update('acc5', acc5.item(), batch_size)
            return success(state)

        eff = success(state).and_then(update_meters)

        if state.stage == Stage.Train:
            eff = eff.and_then(functools.partial(self.backward, loss))

        if self.every(state.batch_idx, self.config.print_freq):
            eff = eff.and_then(self.log_metrics)

        return self.output(loss=loss), eff

    def backward(self, loss: Tensor, state: ClassificationState) -> Effect:
        loss.backward()
        state.optimizer.step()
        state.optimizer.zero_grad()
        return success(state)

    def log_metrics(self, state: ClassificationState) -> Effect:
        _logger.info(f'{state.meters}')
        return success(state)
