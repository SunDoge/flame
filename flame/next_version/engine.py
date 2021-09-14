from flame.pytorch.helpers.data import create_data_loader
from flame.next_version.config_parser import ConfigParser
from typing import Callable, Iterable, Optional
from torch.utils.data import Dataset, dataset
from torch.utils.data.dataloader import DataLoader
from flame.next_version.arguments import BaseArgs
from injector import Module, provider, singleton, inject
from .symbols import IConfig, IArgs, IModel
import logging

_logger = logging.getLogger(__name__)


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

    def __init__(self) -> None:
        pass

    def run(self, data_module: DataModule):
        pass
