from flame.pytorch.distributed import init_process_group_from_file
import tempfile
import torch.distributed as dist
import os


def test_init():
    fd, path = tempfile.mkstemp()
    print('tempfile:', path)
    init_process_group_from_file(
        'gloo', path, world_size=1, rank=0
    )
    assert dist.is_initialized()
