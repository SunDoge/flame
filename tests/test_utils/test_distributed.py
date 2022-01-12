from flame.utils.distributed import init_process_group_from_file
import tempfile
import torch.distributed as dist


def test_init():
    fd, path = tempfile.mkstemp()
    init_process_group_from_file(
        'gloo', path, world_size=1, rank=0
    )
    assert dist.is_initialized()
