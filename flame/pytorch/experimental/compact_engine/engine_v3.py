"""
设计composable的Engine
"""
import logging
from typing import Any, Callable, Iterable, Optional, Tuple
from injector import inject

import torch
from pfun import Effect
from pfun.effect import success
from pydantic import BaseModel as PydanticModel
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from flame.pytorch.meters.time_meter import EstimatedTimeOfArrival
from flame.pytorch.meters.average_meter import DynamicAverageMeterGroup

_logger = logging.getLogger(__name__)


class Stage:
    """
    python的enum并不好用，所以这里还是用constant
    """
    Train: str = 'train'
    Val: str = 'val'
    Test: str = 'test'


class BaseState(PydanticModel):
    step: int = 0
    epoch: int = 0
    batch_idx: int = 0
    epoch_length: int = 0
    stage: str = Stage.Train

    metrics: dict = {}
    meters: DynamicAverageMeterGroup = DynamicAverageMeterGroup()

    def train(self, mode: bool = True):
        raise NotImplementedError()

    def eval(self):
        self.train(mode=False)

    def state_dict(self):
        state_dict = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict'):
                value: nn.Module
                state_dict[key] = value.state_dict()
            else:
                state_dict[key] = value

        return state_dict

    def load_state_dict(self, state_dict: dict):
        # FIXME: 这里没有检查state_dict是否有缺失state
        for key, value in state_dict.items():
            attribute = getattr(self, key)
            if hasattr(attribute, 'load_state_dict'):
                attribute: nn.Module
                attribute.load_state_dict(value)
            else:
                setattr(self, key, value)

    def is_last_batch(self) -> bool:
        return self.batch_idx == self.epoch_length

    class Config:
        arbitrary_types_allowed = True


class BaseEngineConfig(PydanticModel):
    max_epochs: int
    print_freq: int = 1
    update_freq: int = 1  # For gradient accumulation


def default_infer_length(x: Iterable) -> int:
    return len(x)


class BaseDataLoaderDesc():

    def __init__(
        self,
        loader: Iterable,
        length: Optional[int] = None,
        infer_length: Callable[[Iterable], int] = default_infer_length,
    ) -> None:
        if length is None:
            length = infer_length(loader)

        self.loader = loader
        self.length = length


class BaseDataModule:
    def __init__(
        self,
        train: Optional[BaseDataLoaderDesc] = None,
        val: Optional[BaseDataLoaderDesc] = None,
        test: Optional[BaseDataLoaderDesc] = None,
    ) -> None:
        self.train = train
        self.val = val
        self.test = test

    # def get_data(self, stage: str) -> Tuple[Iterable, int]:
    #     loader = self.get_loader(stage)
    #     epoch_length = self.infer_epoch_length(loader, stage)
    #     return loader, epoch_length

    # def get_train_data(self, stage: str = Stage.Train) -> Tuple[Iterable, int]:
    #     return self.get_data(stage)

    # def get_val_data(self, stage: str = Stage.Val) -> Tuple[Iterable, int]:
    #     return self.get_data(stage)

    # def get_test_data(self, stage: str = Stage.Test) -> Tuple[Iterable, int]:
    #     return self.get_data(stage)

    # def get_loader(self, stage: str) -> Iterable:
    #     raise NotImplementedError()

    # def infer_epoch_length(self, loader: Iterable, stage: str) -> int:
    #     return len(loader)


class BaseEngine:
    """
    如果需要覆盖config，必须在这里声明config的类型
    """

    # config: BaseEngineConfig

    def __init__(self, config: BaseEngineConfig) -> None:
        # config_factory = self.__class__.__annotations__['config']
        # _logger.debug('config factory: %s', config_factory)
        # config = config_factory(**dict_config)

        self.config = config

        # placeholders
        self.epoch_eta = EstimatedTimeOfArrival(0)
        self.iteration_eta = EstimatedTimeOfArrival(0)
        # self.meter_group = DynamicAverageMeterGroup()

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

    @staticmethod
    def output(batch_size: int = 1, **kwargs) -> dict:
        kwargs['batch_size'] = batch_size
        return kwargs

    def train(self,
              state: BaseState,
              loader: Iterable,
              epoch_length: Optional[int] = None,
              stage: str = Stage.Train):
        if epoch_length is None:
            epoch_length = self._try_infer_epoch_length(loader)
        elif epoch_length <= 0:
            _logger.info('skip stage %s', stage)
            return

        state.epoch += 1
        state.stage = stage
        state.train(True)

        state.epoch_length = epoch_length

        self._try_set_epoch(loader, state.epoch)

        self.iteration_eta = EstimatedTimeOfArrival(state.epoch_length, )
        # self.meter_group.reset()
        state.meters.reset()

        for batch_idx, batch in enumerate(loader, start=1):
            state.batch_idx = batch_idx
            output = self.training_step(state, batch)
            self.iteration_eta.update(n=output.get('batch_size', 1))

        state.meters.record()

    def validate(self,
                 state: BaseState,
                 loader: Iterable,
                 epoch_length: Optional[int] = None,
                 stage: str = Stage.Val):
        if epoch_length is None:
            epoch_length = self._try_infer_epoch_length(loader)
        elif epoch_length <= 0:
            _logger.info('skip stage %s', stage)
            return

        state.stage = stage
        state.eval()

        state.epoch_length = epoch_length

        self.iteration_eta = EstimatedTimeOfArrival(state.epoch_length, )
        # self.meter_group.reset()
        state.meters.reset()

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                state.batch_idx = batch_idx
                output = self.validation_step(state, batch)
                self.iteration_eta.update(n=output.get('batch_size', 1))

            state.meters.record()

    def test(self,
             state: BaseState,
             loader: Iterable,
             epoch_length: Optional[int] = None,
             stage: str = Stage.Test):
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

    @inject
    def run(self, state: BaseState, data_module: BaseDataModule):

        self.epoch_eta = EstimatedTimeOfArrival(
            self.config.max_epochs,
            state.epoch,
        )

        while state.epoch < self.config.max_epochs:
            if data_module.train:
                self.train(
                    state,
                    data_module.train.loader,
                    epoch_length=data_module.train.length
                )
            if data_module.val:
                self.validate(
                    state,
                    data_module.val.loader,
                    epoch_length=data_module.val.length
                )
            if data_module.test:
                self.test(
                    state,
                    data_module.test.loader,
                    epoch_length=data_module.test.length
                )

            self.epoch_eta.update()


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
                    f'{state.stage} {state.epoch}/{self.config.max_epochs} [{state.batch_idx}/{state.epoch_length}]\t'
                    f'{loss}')
                return success(state)

            eff = eff.and_then(start_logging)

        return {'loss': loss}, eff


class ExampleDataModule(BaseDataModule):
    def get_loader(self, stage: Stage) -> Iterable:
        if stage == Stage.Test:
            return range(0)

        if stage == Stage.Train:
            return range(10)

        if stage == Stage.Val:
            return range(5)


if __name__ == '__main__':
    import torch
    from icecream import ic
    logging.basicConfig(level=logging.INFO)
    engine_config = BaseEngineConfig(
        max_epochs=10,
        print_freq=5,
    )

    model = torch.nn.Linear(2, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    state = ExampleState(model=model, optimizer=optimizer)

    data_module = ExampleDataModule()

    engine = ExampleEngine(engine_config)
    print(engine.config)
    engine.run(state, data_module)
    ic(state)

    state_dict = state.state_dict()
    ic(state_dict)

    new_model = torch.nn.Linear(2, 4)
    new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)
    new_state = ExampleState(model=new_model, optimizer=new_optimizer)
    ic(new_state)
    new_state.load_state_dict(state_dict)

    ic(new_state.state_dict())
