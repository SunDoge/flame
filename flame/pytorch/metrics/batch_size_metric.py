from .base_metric import Metric
from typing import Callable


class BatchSizeMetric(Metric):

    def __init__(
        self,
        output_transform: Callable,
        name: str = 'batch_size',
    ) -> None:
        super().__init__(name, output_transform)
