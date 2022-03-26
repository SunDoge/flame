from typing import Optional
from flame.core.arguments import BaseArgs as Base
import torch
import typed_args as ta
from dataclasses import dataclass
import torch.distributed as dist
import logging
from .distributed import init_process_group_from_file
from flame.core.operating_system import find_free_port

_logger = logging.getLogger(__name__)


@dataclass
class BaseArgs(Base):

    rank_start: int = ta.add_argument(
        '--rank-start', type=int, default=0,
        help=''
    )

    _dist_backend: str = ta.add_argument(
        "--dist-backend", type=str, choices=["nccl", "gloo"], default=None
    )
    world_size: int = ta.add_argument(
        '--world-size', type=int, default=1,
    )
    dist_url: Optional[str] = ta.add_argument(
        "--dist-url", type=str,
    )

    @property
    def device(self) -> torch.device:
        if self.gpu and torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    @property
    def dist_backend(self) -> str:
        backend = self._dist_backend
        if backend is None:
            if self.gpu and dist.is_nccl_available():
                backend = "nccl"
            else:
                backend = "gloo"
            _logger.info("infer dist_backend: %s", backend)

        return backend

    def init_process_group_from_file(self, local_rank: int) -> int:
        rank = self.rank_start + local_rank

        init_process_group_from_file(
            self.dist_backend,
            self.experiment_dir / 'dist_init',
            world_size=self.world_size,
            rank=rank,
        )

        return rank

    def init_process_group_from_tcp(self, local_rank: int) -> int:
        assert self.dist_url

        rank = self.rank_start + local_rank

        dist.init_process_group(
            self.dist_backend,
            init_method=self.dist_url,
            world_size=self.world_size,
            rank=rank,
        )

        return rank

    def try_cuda_set_device(self, local_rank: int):
        if self.gpu:
            device_id = self.gpu[local_rank]
            torch.cuda.set_device(device_id)
