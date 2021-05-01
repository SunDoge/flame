from flame.pytorch import distributed_training
import flame
import torch.distributed as dist


def _assert_x_eq_1(x: int, local_rank=0):
    assert x == 1


def test_start_distributed_training_on_cpu():
    port = flame.utils.operating_system.find_free_port()
    # dist_url = f'tcp://127.0.0.1:{port}'

    distributed_training.start_distributed_training(
        _assert_x_eq_1,
        args=(1,),
        dist_options=distributed_training.DistOptions(
            dist=True,
            dist_backend='gloo',
            dist_port=port,
            rank_start=0,
            world_size=1,
            dist_host='127.0.0.1'
        ),
    )

    dist.destroy_process_group()


def test_local_dist_url():
    assert distributed_training.get_available_local_dist_url()


def test_init_cpu_process_group():
    distributed_training.init_cpu_process_group(
        dist_url=distributed_training.get_available_local_dist_url()
    )

    assert dist.get_rank() == 0

    dist.destroy_process_group()