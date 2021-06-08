from typing import Any, Optional, TypeVar, Tuple
from .base_checkpoint_saver import CheckpointSaver
import operator
import os
import shutil
import logging
from flame.pytorch.utils.ranking import rank0

_logger = logging.getLogger(__name__)

T = TypeVar('T')


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
        self.higher_is_better = higher_is_better
        self.comparator = operator.ge if self.higher_is_better else operator.le

    def is_best(self, metric: T, best_metric: Optional[T]) -> bool:
        if best_metric is None:
            return True
        else:
            return self.comparator(metric, best_metric)

    @rank0
    def save(self, output_dir: str, is_best: bool = False):
        """
        在save checkpoint之前，best metric就应该被更新。所以没法在这里计算is_best
        """

        filename = os.path.join(output_dir, self.name)
        super().save(filename)

        if is_best:
            filename_best = os.path.join(output_dir, self.best_name)
            _logger.info('save best checkpoint to: %s', filename_best)
            shutil.copyfile(filename, filename_best)

        return is_best
