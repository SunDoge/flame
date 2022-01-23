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
    dist_url: str = ta.add_argument(
        "--dist-url", type=str, default=f"tcp://127.0.0.1:{find_free_port()}"
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

    def init_process_group_from_file(self, local_rank: int):
        rank = self.rank_start + local_rank

        init_process_group_from_file(
            self.dist_backend,
            self.experiment_dir / 'dist_init',
            world_size=self.world_size,
            rank=rank,
        )
