"""
Debug模式测试
python -m example.mnist.main -c example/mnist/pytorch_example.jsonnet -dy
"""


from flame.pytorch.experimental.compact_engine.engine_v2 import BaseEngine, State
from torch.utils.data.dataloader import DataLoader

import flame
from flame.pytorch.typing_prelude import Criterion, Device, LocalRank, Model, DictConfig, Optimizer, LrScheduler
from flame.pytorch.container import BaseModule
from injector import Injector, provider, singleton
from .config import TypedConfig, Stage
from flame.pytorch import helpers
from flame.pytorch.container import build_from_config_with_container
from flame.pytorch.distributed_training import start_training
import logging

_logger = logging.getLogger(__name__)


class MnistModule(BaseModule):
    
    @singleton
    @provider
    def create_typed_config(self, cfg: DictConfig) -> TypedConfig:
        return TypedConfig(**cfg)

    @provider
    def create_data_loader(self, cfg: TypedConfig, split: str = 'train') -> DataLoader:
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
    def create_criterion(self, cfg: TypedConfig) -> Criterion:
        # TODO
        return flame.auto_builder.build_from_config(
            cfg.criterion
        )

    @singleton
    @provider
    def create_optimizer(self, cfg: TypedConfig, model: Model) -> Optimizer:
        # TODO
        return flame.auto_builder.build_from_config(
            cfg.optimizer, model.parameters()
        )

    @singleton
    @provider
    def create_model(self, cfg: TypedConfig, device: Device, local_rank: LocalRank) -> Model:
        # TODO
        base_model = flame.auto_builder.build_from_config(
            cfg.model
        )
        model = helpers.create_model(
            base_model, device,
        )
        return model

    @singleton
    @provider
    def create_scheduler(self, cfg: TypedConfig, optimizer: Optimizer) -> LrScheduler:
        return flame.auto_builder.build_from_config(
            cfg.scheduler, optimizer
        )

    # @singleton
    # @provider
    # def create_engine_config(self, cfg: RootConfig) -> BaseEngineConfig:
    #     return cfg['engine']

    @singleton
    @provider
    def create_engine_state(self,) -> State:
        return State()


def main_worker(local_rank: int):
    container = Injector(MnistModule(local_rank))

    # train_loader = loader_builder.build(split='train')
    # val_loader = loader_builder.build(split='val')
    cfg = container.get(TypedConfig)

    # engine: BaseEngine = container.get(
    #     flame.auto_builder.import_from_path(cfg.engine.type)
    # )
    engine: BaseEngine = build_from_config_with_container(
        container, cfg.engine
    )

    # eta = EstimatedTimeOfArrival('epoch', cfg.max_epochs)
    # while engine.unfinished(cfg.max_epochs):
    #     engine.train(train_loader)
    #     engine.validate(val_loader)
    engine.run()


@flame.utils.main_fn
def main():
    start_training(main_worker)
