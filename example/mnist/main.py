"""
Debug模式测试
python -m example.mnist.main -c example/mnist/pytorch_example.jsonnet -dy
"""


from flame.pytorch.experimental.compact_engine.engine import BaseEngine, BaseEngineConfig
from torch.utils.data.dataloader import DataLoader

import flame
from flame.pytorch.typing_prelude import Criterion, Device, LocalRank, Model, RootConfig, Optimizer, LrScheduler
from flame.pytorch.container import RootModule
from injector import Injector, provider, singleton
from .config import Config, Stage
from flame.pytorch import helpers
from flame.pytorch.container import CallableAssistedBuilder
from flame.pytorch.container import build_from_config_with_container
from flame.pytorch.distributed_training import start_training
import logging

_logger = logging.getLogger(__name__)


class MnistModule(RootModule):
    @singleton
    @provider
    def create_typed_config(self, cfg: RootConfig) -> Config:
        return Config(**cfg)

    @provider
    def create_data_loader(self, cfg: Config, split: str = 'train') -> DataLoader:
        stage: Stage = getattr(cfg, split)

        transform = flame.auto_builder.build_from_config(
            stage.transform
        )
        dataset = flame.auto_builder.build_from_config(
            stage.dataset, transform=transform
        )
        loader = helpers.create_data_loader(
            dataset,
            batch_size=stage.batch_size,
            num_workers=stage.num_workers
        )
        return loader

    @singleton
    @provider
    def create_criterion(self, cfg: Config) -> Criterion:
        # TODO
        return flame.auto_builder.build_from_config(
            cfg.criterion
        )

    @singleton
    @provider
    def create_optimizer(self, cfg: Config, model: Model) -> Optimizer:
        # TODO
        return flame.auto_builder.build_from_config(
            cfg.optimizer, model.parameters()
        )

    @singleton
    @provider
    def create_model(self, cfg: Config, device: Device, local_rank: LocalRank) -> Model:
        # TODO
        base_model = flame.auto_builder.build_from_config(
            cfg.model
        )
        model = helpers.create_model(
            base_model, device, local_rank=local_rank,
        )
        return model

    @singleton
    @provider
    def create_scheduler(self, cfg: Config, optimizer: Optimizer) -> LrScheduler:
        return flame.auto_builder.build_from_config(
            cfg.scheduler, optimizer
        )

    @singleton
    @provider
    def create_engine_config(self, cfg: RootConfig) -> BaseEngineConfig:
        return cfg['engine']


def main():
    start_training(main_worker)


def main_worker(local_rank: int):
    container = Injector(MnistModule(local_rank))
    loader_builder = container.get(CallableAssistedBuilder[DataLoader])

    train_loader = loader_builder.build(split='train')
    val_loader = loader_builder.build(split='val')
    cfg = container.get(Config)

    engine: BaseEngine = container.get(
        flame.auto_builder.import_from_path(cfg.engine.type)
    )

    while engine.unfinished(cfg.max_epochs):
        engine.train(train_loader)
        engine.validate(val_loader)


if __name__ == '__main__':
    main()
