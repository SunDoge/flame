import torch
from .base_metric import Metric
from .functional import topk_accuracy
from typing import Sequence, Union, Tuple, Callable
import functools


class TopkAccuracy(Metric):

    def __init__(
        self,
        name: Sequence[str],
        output_transform: Callable,
        topk: Sequence[str] = (1,),
        use_jit: bool = False,
    ) -> None:

        if use_jit:
            topk_accuracy = torch.jit.script(topk_accuracy)

        compute_fn = functools.partial(topk_accuracy, topk=topk)

        super().__init__(name, output_transform, compute_fn)
