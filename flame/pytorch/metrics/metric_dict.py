from flame.pytorch.meters import AverageMeter
from .base_metric import Metric
from dataclasses import dataclass


@dataclass
class Item:
    metric: Metric
    meter: AverageMeter


class MetricDict:

    def __init__(
        self,
        **metrics
    ) -> None:
        metric_dict = {}
        for name, metric in metrics.items():
            item = Item(
                metric,
                AverageMeter(name)
            )
