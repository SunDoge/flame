from os import name
import torch
from flame.next_version.state import BaseState
from pathlib import Path
from flame.pytorch.utils.ranking import rank0
from contextlib import contextmanager
import logging

_logger = logging.getLogger(__name__)


@contextmanager
def safe_delete(file_path: Path):

    new_name = file_path.name + '.tmp'
    new_file_path = file_path.parent / new_name
    if file_path.exists():
        file_path.rename(new_file_path)
    yield new_file_path
    # 如果成功完成保存，再删掉旧的
    new_file_path.unlink(missing_ok=True)


@rank0
def save_checkpoint(
    state: dict,
    experiment_dir: Path,
    is_best: bool = False,
    filename='checkpoint.pth.tar'
):
    file_path: Path = experiment_dir / filename
    with safe_delete(file_path):
        torch.save(state, str(file_path))
    if is_best:
        best_file_path = experiment_dir / 'model_best.pth.tar'
        with safe_delete(best_file_path):
            file_path.link_to(best_file_path)


# @rank0
# class CheckpointSaver:

#     def __init__(
#         self,
#         experiment_dir: Path,
#         state: BaseState,
#     ) -> None:
#         self.experiment_dir = experiment_dir
#         self.state = state

#     def save(self, state_dict: dict, name: str) -> Path:
#         file_path = self.experiment_dir / name.format(**self.state.__dict__)
#         with safe_delete(file_path):
#             torch.save(state_dict, str(file_path))

#         _logger.info('save checkpoint: %s', file_path)

#         return file_path

#     def save_checkpoint(self, name: str = 'checkpoint.pth.tar') -> Path:
#         self.save(self.state.state_dict(), name)

#     def save_weights(self, name: str = 'checkpoint.pth') -> Path:
#         self.save(self.state.get_weights(), name)


# @rank0
# class LatestCheckpointSaver(CheckpointSaver):

#     def save_checkpoint(self, name: str = 'checkpoint.pth.tar'):
#         return super().save_checkpoint(name=name)

#     def save_weights(self, name: str = 'checkpoint.pth'):
#         return super().save_weights(name=name)


# @rank0
# class BestCheckpointSaver(CheckpointSaver):

#     def save_checkpoint(self, name: str = 'best_model.pth.tar'):
#         return super().save_checkpoint(name=name)

#     def save_weights(self, name: str = 'best_model.pth'):
#         return super().save_weights(name=name)


# @rank0
# class PeriodicCheckpointSaver(CheckpointSaver):

#     def save_checkpoint(self, name: str = 'checkpoint_{epoch:03d}.pth.tar'):
#         return super().save_checkpoint(name=name.format(self.state.epoch))

#     def save_weights(self, name: str = 'checkpoint_{epoch:03d}.pth'):
#         return super().save_weights(name=name)
