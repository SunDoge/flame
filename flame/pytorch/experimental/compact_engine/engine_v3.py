"""
设计composable的Engine
"""
from typing import Any, Iterable, Tuple
from pydantic import BaseModel
import logging
from pfun import Effect

_logger = logging.getLogger(__name__)


class BaseState(BaseModel):
    step: int = 0
    epoch: int = 0
    batch_idx: int = 0
    epoch_length: int = 0

    metrics: dict = {}


class BaseEngineConfig(BaseModel):
    max_epochs: int
    print_freq: int = 1
    update_freq: int = 1  # For gradient accumulation


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

    def switch_training_mode(self, mode: bool):
        raise NotImplementedError()

    def training_step(self, state: BaseState, batch: Any) -> dict:
        output, effect = self.forward(state, batch)
        effect.run(None)
        return output

    def validation_step(self, state: BaseState, batch: Any) -> dict:
        output, effect = self.forward(state, batch)
        effect.run(None)
        return output

    def train(self, state: BaseState):
        pass

    def validate(self, state: BaseState):
        pass

    def test(self, state: BaseState):
        pass

    def run(self, state: BaseState):
        pass


class ExampleState(BaseState):
    model: dict
    optimizer: dict


if __name__ == '__main__':
    dict_config = {'max_epochs': 100}
    engine = BaseEngine(dict_config)
    print(engine.config)
