from pydantic import BaseModel
from flame.pytorch.experimental.compact_engine.engine import BaseEngineConfig


class Stage(BaseModel):
    batch_size: int
    num_workers: int
    transform: dict
    dataset: dict


class Config(BaseModel):
    train: Stage
    val: Stage
    engine: BaseEngineConfig
    max_epochs: int
