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
from enum import Enum
from torch.utils.data.distributed import DistributedSampler

_logger = logging.getLogger(__name__)


class Stage(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class BaseState(BaseModel):
    step: int = 0
    epoch: int = 0
    batch_idx: int = 0
    epoch_length: int = 0
    stage: Stage = Stage.TRAIN

    metrics: dict = {}

    def train(self, mode: bool = True):
        raise NotImplementedError()

    def eval(self):
        self.train(mode=False)

    def state_dict(self):
        state_dict = {}
        for key, value in self.__dict__.items():
            # if isinstance(value, torch.nn.Module):
            #     state_dict[key] = value.state_dict()
            # elif isinstance(value, torch.optim.Optimizer):
            #     state_dict[key] = value.state_dict()
            # elif isinstance(value, torch.optim.lr_scheduler._LRScheduler):
            #     state_dict[key] = value.state_dict()
            # elif isinstance(value, torch.cuda.amp.grad_scaler.GradScaler):
            #     state_dict[key] = value.state_dict()
            if hasattr(value, 'state_dict'):
                state_dict[key] = value.state_dict()
            else:
                state_dict[key] = value

        return state_dict

    def load_state_dict(self, state_dict: dict):
        for key, value in state_dict.items():
            attribute = getattr(self, key)
            if hasattr(attribute, 'load_state_dict'):
                attribute.load_state_dict(value)
            else:
                setattr(self, key, value)

    def is_last_batch(self) -> bool:
        return self.batch_idx == self.epoch_length

    class Config:
        arbitrary_types_allowed = True


class BaseEngineConfig(BaseModel):
    max_epochs: int
    print_freq: int = 1
    update_freq: int = 1  # For gradient accumulation


class DataModule:
    def __init__(self) -> None:
        pass

    def get_data(self, stage: Stage) -> Tuple[Iterable, int]:
        loader = self.get_loader(stage)
        epoch_length = self.infer_epoch_length(loader, stage)
        return loader, epoch_length

    def get_train_data(self, stage: Stage = Stage.TRAIN) -> Tuple[Iterable, int]:
        return self.get_data(stage)

    def get_val_data(self, stage: Stage = Stage.VAL) -> Tuple[Iterable, int]:
        return self.get_data(stage)

    def get_test_data(self, stage: Stage = Stage.TEST) -> Tuple[Iterable, int]:
        return self.get_data(stage)

    def get_loader(self, stage: Stage) -> Iterable:
        raise NotImplementedError()

    def infer_epoch_length(self, loader: Iterable, stage: Stage) -> int:
        return len(loader)


class BaseEngine:

    """
    如果需要覆盖config，必须在这里声明config的类型
    """
    config: BaseEngineConfig

    def __init__(self, dict_config: dict) -> None:
        config_factory = self.__class__.__annotations__['config']
        _logger.debug('config factory: %s', config_factory)
        config = config_factory(**dict_config)

        self.config = config

    def forward(self, state: BaseState, batch: Any) -> Tuple[dict, Effect]:
        """
        Args:
            state: 训练需要用到的状态，不能修改
            batch: 训练数据
        """
        raise NotImplementedError()

    def training_step(self, state: BaseState, batch: Any) -> dict:
        state.step += 1
        output, effect = self.forward(state, batch)
        effect.run(None)
        return output

    def validation_step(self, state: BaseState, batch: Any) -> dict:
        output, effect = self.forward(state, batch)
        effect.run(None)
        return output

    def train(self, state: BaseState, loader: Iterable, epoch_length: Optional[int] = None, stage: Stage = Stage.TRAIN):
        if epoch_length is None:
            epoch_length = self._try_infer_epoch_length(loader)
        elif epoch_length <= 0:
            _logger.info('skip stage %s', stage.value)
            return

        state.epoch += 1
        state.stage = stage
        state.train()

        state.epoch_length = epoch_length

        self._try_set_epoch(loader, state.epoch)

        for batch_idx, batch in enumerate(loader, start=1):
            state.batch_idx = batch_idx
            _output = self.training_step(state, batch)

    def validate(self, state: BaseState, loader: Iterable, epoch_length: Optional[int] = None, stage: Stage = Stage.VAL):
        if epoch_length is None:
            epoch_length = self._try_infer_epoch_length(loader)
        elif epoch_length <= 0:
            _logger.info('skip stage %s', stage.value)
            return

        state.stage = stage
        state.eval()

        state.epoch_length = epoch_length

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                state.batch_idx = batch_idx
                _output = self.validation_step(state, batch)

    def test(self, state: BaseState, loader: Iterable, epoch_length: Optional[int] = None, stage: Stage = Stage.TEST):
        self.validate(state, loader, epoch_length=epoch_length, stage=stage)

    @staticmethod
    def _try_infer_epoch_length(loader: Iterable):
        return len(loader)

    @staticmethod
    def _try_set_epoch(loader: Iterable, epoch: int):
        if hasattr(loader, 'sampler'):
            sampler = getattr(loader, 'sampler')
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
                _logger.info('train_loader.sampler.set_epoch(%s)', epoch)

    @staticmethod
    def every(i: int, n: int) -> bool:
        return i > 0 and i % n == 0

    # def every_n_steps(self, n: int = 1) -> bool:
    #     return self.every(self.state.step, n)

    def run(self, state: BaseState, data_module: DataModule):
        train_loader, train_epoch_length = data_module.get_train_data()
        val_loader, val_epoch_length = data_module.get_val_data()
        test_loader, test_epoch_length = data_module.get_test_data()

        while state.epoch < self.config.max_epochs:
            self.train(state, train_loader, train_epoch_length)
            self.validate(state, val_loader, val_epoch_length)
            self.test(state, test_loader, test_epoch_length)


class ExampleState(BaseState):
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer

    def train(self, mode: bool = True):
        self.model.train(mode=mode)


class ExampleEngine(BaseEngine):

    config: BaseEngineConfig

    def forward(self, state: ExampleState, batch):
        data = torch.full((4, 2), batch, dtype=torch.float)
        pred = state.model(data)
        loss = pred.sum()

        eff = success(state)
        if state.stage == 'train':
            def update_model(state: ExampleState):
                loss.backward()
                state.optimizer.step()
                state.optimizer.zero_grad()
                return success(state)

            eff = eff.and_then(update_model)

        if self.every(state.batch_idx, self.config.print_freq):
            def start_logging(state: ExampleState):
                _logger.info(
                    f'{state.stage.value} {state.epoch}/{self.config.max_epochs} [{state.batch_idx}/{state.epoch_length}]\t'
                    f'{loss}'
                )
                return success(state)

            eff = eff.and_then(start_logging)

        return {'loss': loss}, eff


class ExampleDataModule(DataModule):

    def get_loader(self, stage: Stage) -> Iterable:
        if stage == Stage.TEST:
            return range(0)

        if stage == Stage.TRAIN:
            return range(10)

        if stage == Stage.VAL:
            return range(5)


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

    data_module = ExampleDataModule()

    engine = ExampleEngine(dict_config)
    print(engine.config)
    engine.run(state, data_module)
    ic(state)

    state_dict = state.state_dict()
    ic(state_dict)

    new_model = torch.nn.Linear(2, 4)
    new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)
    new_state = ExampleState(
        model=new_model,
        optimizer=new_optimizer
    )
    ic(new_state)
    new_state.load_state_dict(state_dict)

    ic(new_state.state_dict())
