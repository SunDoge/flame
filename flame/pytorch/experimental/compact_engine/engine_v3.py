"""
设计composable的Engine
"""
from typing import Any, Iterable, Optional, Tuple
from pfun.effect import success
from pydantic import BaseModel
import logging
from pfun import Effect
import torch
from dataclasses import dataclass

_logger = logging.getLogger(__name__)


class BaseState(BaseModel):
    step: int = 0
    epoch: int = 0
    batch_idx: int = 0
    epoch_length: int = 0
    mode: str = 'train'

    metrics: dict = {}

    def train(self, mode: bool = True):
        raise NotImplementedError()

    def eval(self):
        self.train(mode=False)

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict: dict):
        pass

    class Config:
        arbitrary_types_allowed = True


class BaseEngineConfig(BaseModel):
    max_epochs: int
    print_freq: int = 1
    update_freq: int = 1  # For gradient accumulation


class DataModule:

    pass


class BaseEngine:

    """
    如果需要覆盖config，必须在这里声明config的类型
    """
    config: BaseEngineConfig

    def __init__(self, dict_config: dict, state: BaseState) -> None:
        config_factory = self.__class__.__annotations__['config']
        _logger.debug('config factory: %s', config_factory)
        config = config_factory(**dict_config)

        self.config = config
        self.state = state

    def forward(self, batch: Any) -> Tuple[dict, Effect]:
        """
        Args:
            state: 训练需要用到的状态，不能修改
            batch: 训练数据
        """
        raise NotImplementedError()

    def switch_training_mode(self, mode: bool):
        raise NotImplementedError()

    def training_step(self, batch: Any) -> dict:
        self.state.step += 1
        output, effect = self.forward(batch)
        effect.run(None)
        return output

    def validation_step(self, batch: Any) -> dict:
        output, effect = self.forward(batch)
        effect.run(None)
        return output

    def train(self, loader: Iterable, epoch_length: Optional[int] = None, mode: str = 'train'):
        self.state.epoch += 1
        self.state.mode = mode
        self.state.train()

        if epoch_length is None:
            epoch_length = self._try_infer_epoch_length(loader)

        self.state.epoch_length = epoch_length

        for batch_idx, batch in enumerate(loader, start=1):
            self.state.batch_idx = batch_idx
            output = self.training_step(batch)

    def validate(self, loader: Iterable, epoch_length: Optional[int] = None, mode: str = 'val'):
        self.state.mode = mode
        self.state.eval()

        if epoch_length is None:
            epoch_length = self._try_infer_epoch_length(loader)

        self.state.epoch_length = epoch_length

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                self.state.batch_idx = batch_idx
                output = self.validation_step(batch)

    def test(self, loader: Iterable, epoch_length: Optional[int] = None, mode: str = 'test'):
        self.validate(mode=mode)

    @staticmethod
    def _try_infer_epoch_length(loader: Iterable):
        return len(loader)

    @staticmethod
    def every(i: int, n: int) -> bool:
        return i > 0 and i % n == 0

    def every_n_steps(self, n: int = 1) -> bool:
        return self.every(self.state.step, n)

    def run(self,):

        while self.state.epoch < self.config.max_epochs:
            self.train(range(10))
            self.validate(range(5))


class ExampleState(BaseState):
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer

    def train(self, mode: bool = True):
        self.model.train(mode=mode)


class ExampleEngine(BaseEngine):

    config: BaseEngineConfig
    state: ExampleState

    def forward(self, batch):
        data = torch.full((4, 2), batch, dtype=torch.float)
        pred = self.state.model(data)
        loss = pred.sum()

        eff = success(self.state)
        if self.state.mode == 'train':
            def update_model(state: ExampleState):
                loss.backward()
                state.optimizer.step()
                state.optimizer.zero_grad()
                return success(state)

            eff = eff.and_then(update_model)

        if self.every(self.state.batch_idx, self.config.print_freq):
            def start_logging(state: ExampleEngine):
                _logger.info(
                    f'{self.state.epoch}/{self.config.max_epochs} [{self.state.batch_idx}/{self.state.epoch_length}]\t'
                    f'{loss}'
                )
                return success(state)

            eff = eff.and_then(start_logging)

        return {'loss': loss}, eff


if __name__ == '__main__':
    import torch
    from icecream import ic
    logging.basicConfig(level=logging.INFO)
    dict_config = {'max_epochs': 100, 'print_freq': 5}

    model = torch.nn.Linear(2, 4)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    state = ExampleState(
        model=model,
        optimizer=optimizer
    )
    engine = ExampleEngine(dict_config, state)
    print(engine.config)
    engine.run()
    ic(engine.state)
