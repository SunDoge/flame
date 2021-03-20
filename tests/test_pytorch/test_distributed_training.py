from flame.pytorch import distributed_training
import flame


def _assert_x_eq_1(x: int):
    assert x == 1


def test_start_distributed_training_on_cpu():
    port = flame.utils.operating_system.find_free_port()
    dist_url = f'tcp://127.0.0.1:{port}'

    distributed_training.start_distributed_training(
        _assert_x_eq_1,
        args=(1,),
        dist_backend='GLOO',
        dist_url=dist_url
    )
