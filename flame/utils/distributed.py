from typing import Union
import torch.distributed as dist
from pathlib import Path
import logging

_logger = logging.getLogger(__name__)


def init_process_group_from_file(
    backend: str, filename: Union[Path, str], world_size: int = 1, rank: int = 0,
):
    """
    https://pytorch.org/docs/stable/distributed.html

    Args:
        backend: nccl or gloo
        filename: 用于init的file
        world_size: 或总节点数
        rank: 当前节点id
    """
    file_path = Path(filename)

    file_path.unlink(missing_ok=True)

    uri = file_path.resolve().as_uri()

    _logger.info('init_method=%s', uri)

    dist.init_process_group(
        backend=backend,
        init_method=uri,
        world_size=world_size,
        rank=rank
    )
