import logging
from contextlib import contextmanager
from pathlib import Path

import oneflow as flow
from .rank0 import rank0
import shutil

_logger = logging.getLogger(__name__)


@contextmanager
def safe_delete(file_path: Path):

    new_name = file_path.name + ".tmp"
    new_file_path = file_path.parent / new_name
    if file_path.exists():
        file_path.rename(new_file_path)
    yield new_file_path
    # 如果成功完成保存，再删掉旧的
    new_file_path.rmdir()


@rank0
def save_checkpoint(
    state: dict,
    experiment_dir: Path,
    is_best: bool = False,
    filename="checkpoint",
):
    file_path: Path = experiment_dir / filename
    with safe_delete(file_path):
        flow.save(state, str(file_path))
        _logger.info("save checkpoint: %s", file_path)
    if is_best:
        best_file_path = experiment_dir / "model_best"
        with safe_delete(best_file_path):
            shutil.copytree(file_path, best_file_path)
            _logger.info("save best checkpoint: %s", best_file_path)
