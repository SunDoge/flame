from typing import Any, Optional, TypeVar
from .base_checkpoint_saver import CheckpointSaver
import operator
import os
import shutil
import logging

_logger = logging.getLogger(__name__)


class BestCheckpointSaver(CheckpointSaver):

    def __init__(
        self,
        name: str = 'checkpoint.pth.tar',
        best_name: str = 'model_best.pth.tar',
        higher_is_better: bool = True,
        entries=None
    ) -> None:
        super().__init__(entries=entries)
        self.name = name
        self.best_name = best_name
        self.best_metric = None
        self.higher_is_better = higher_is_better

    def save(self, output_dir: str, is_best: Optional[bool] = None, metric: Any = None) -> bool:
        # 至少一个不为None
        assert not all(x is None for x in [is_best, metric])

        if is_best is None:
            comparator = operator.ge if self.higher_is_better else operator.le
            if self.best_metric is None:
                self.best_metric = metric
            is_best = comparator(metric, self.best_metric)

            if is_best:
                self.best_metric = metric

        filename = os.path.join(output_dir, self.name)
        super().save(filename)

        if is_best:
            filename_best = os.path.join(output_dir, self.best_name)
            _logger.info('save best checkpoint to: %s', filename_best)
            shutil.copyfile(filename, filename_best)

        return is_best
