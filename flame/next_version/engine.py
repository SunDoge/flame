from flame.pytorch.helpers.data import create_data_loader
from flame.next_version.config_parser import ConfigParser
from typing import Callable, Iterable, Optional
from torch.utils.data import Dataset, dataset
from torch.utils.data.dataloader import DataLoader
from flame.next_version.arguments import BaseArgs
from injector import Module, provider, singleton, inject
from .symbols import IConfig, IArgs, IModel
import logging
from typing import Any, List, Tuple

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


class DataModule:

    def __init__(
        self,
        train_loader: Optional[DataLoader],
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader] = None,
        length_fn: Callable[[Iterable], int] = len,
    ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.length_fn = length_fn


class BaseModule(Module):

    def __init__(self, args: BaseArgs, config: dict) -> None:
        super().__init__()
        self.args = args
        self.config = config

    @singleton
    @provider
    def provide_config(self) -> IConfig:
        return self.config

    @singleton
    @provider
    def provide_args(self) -> IArgs:
        return self.args

    @singleton
    @provider
    def provide_data_module(self, config: IConfig) -> DataModule:
        train_loader = self.get_loader(config['train'])
        val_loader = self.get_loader(config['val'])
        return DataModule(train_loader, val_loader)

    def get_loader(self, config: dict):
        config_parser = ConfigParser()
        transform_config = config['transform']
        _logger.info('transform config: %s', transform_config)
        transform = config_parser.parse(transform_config)
        _logger.info(f'transform: {transform}')
        ds_config = config['dataset']
        ds_config['transform'] = transform
        ds: Dataset = config_parser.parse(ds_config)
        loader_config = config['loader']
        loader_config['dataset'] = ds
        loader = config_parser.parse(loader_config)
        return loader

    @singleton
    @provider
    def provide_model(self, config: IConfig) -> IModel:
        model_config = config['model']
        # TODO


class BaseEngine:

    ProviderModule = BaseModule

    @inject
    def __init__(
        self,
        config: IConfig
    ) -> None:
        self.config = config
        self.epoch_eta = EstimatedTimeOfArrival(0)
        self.iteration_eta = EstimatedTimeOfArrival(0)

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

    def run(self, state: BaseState, data_module: DataModule):

        max_epochs = self.config['max_epochs']
        self.epoch_eta = EstimatedTimeOfArrival(
            max_epochs,
            state.epoch,
        )

        while state.epoch < max_epochs:
            if data_module.train_loader:
                self.train(
                    state,
                    data_module.train_loader,
                    epoch_length=data_module.length_fn(
                        data_module.train_loader
                    )
                )
            if data_module.val_loader:
                self.validate(
                    state,
                    data_module.val_loader,
                    epoch_length=data_module.length_fn(data_module.val_loader)
                )

            self.epoch_eta.update()

        if data_module.test_loader:
            self.test(
                state,
                data_module.test_loader,
                epoch_length=data_module.length_fn(data_module.test_loader)
            )
