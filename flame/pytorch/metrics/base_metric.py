

from typing import Callable, Any, Dict, List, Sequence, Tuple, Union
from numbers import Number
from torch import Tensor
from torch._C import R


def _to_number(x) -> Number:
    if isinstance(x, Tensor):
        return x.tolist()
    else:
        return x


def _default_compute_fn(*args) -> Any:
    return args


class Metric:

    def __init__(
        self,
        name: Union[str, Sequence[str]],
        output_transform: Callable,
        compute_fn: Callable = _default_compute_fn,
    ) -> None:
        self.name = name
        self.output_transform = output_transform
        self.compute_fn = compute_fn

    def __call__(self, output: dict) -> dict:
        output: tuple = self.output_transform(output)
        result = self.compute_fn(*output)

        if isinstance(self.name, str):
            result = (result,)
            name = (self.name,)
        else:
            name = self.name

        metric = {n: _to_number(r) for n, r in zip(name, result)}
        return metric


class MetricList:

    def __init__(self, metrics: List[Metric]) -> None:
        self.metrics = metrics

    def __call__(self, output: dict) -> dict:
        metrics = {}
        for m in self.metrics:
            metric = m(output)
            metrics.update(metric)
        return metrics
