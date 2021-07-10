from .config import StageConfig, TypedConfig
from flame.pytorch.experimental.compact_engine.engine_v3 import BaseDataModule, Stage
from injector import inject
import flame
from flame.pytorch import helpers

from typing import Iterable


@inject
class MnistDataModule(BaseDataModule):

    def __init__(self, cfg: TypedConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def get_loader(self, stage: str) -> Iterable:
        stage: StageConfig = getattr(self.cfg, stage)

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
