from .base_metric import Metric
from typing import Union, Callable, Sequence


class CopyMetric(Metric):

    def __init__(
        self,
        name: Union[str, Sequence[str]],
    ) -> None:

        if isinstance(name, str):
            def output_transform(output: dict) -> tuple:
                return output[name],
        else:
            def output_transform(output: dict) -> tuple:
                return [output[n] for n in name],

        super().__init__(name, output_transform)
