from pydantic import BaseModel
from flame.pytorch.experimental.compact_engine.engine import BaseEngineConfig


class StageConfig(BaseModel):
    batch_size: int
    num_workers: int
    transform: dict
    dataset: dict



class TypedConfig(BaseModel):
    train: StageConfig
    val: StageConfig
    engine: str
    engine_cfg: dict
    # max_epochs: int
    optimizer: dict
    model: dict
    criterion: dict
    scheduler: dict