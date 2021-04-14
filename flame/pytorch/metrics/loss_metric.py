from .base_metric import Metric
from typing import Callable, Sequence, Union


class LossMetric(Metric):

    def __init__(
        self,
        output_transform: Callable,
        name: Union[str, Sequence[str]] = 'loss',
    ) -> None:
        super().__init__(name, output_transform)
