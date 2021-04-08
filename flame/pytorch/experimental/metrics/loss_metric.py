from typing import Any, Dict
from flame.pytorch.experimental.engine.events import State
from .metric import Metric
from torch import Tensor


def _default_output_transform(output: Dict[str, Any]):
    return (output['loss'],)


class LossMetric(Metric):
    """
    Loss可能和一些typing命名冲突
    """

    def __init__(
        self,
        name: str,
        output_transform: _default_output_transform,
    ) -> None:
        super().__init__(name, output_transform=output_transform)

    def compute(self, loss: Tensor) -> Tensor:
        return loss.detach()
