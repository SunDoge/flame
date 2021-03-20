from flame.utils import operating_system
import resource


def test_find_free_port():
    port = operating_system.find_free_port()
    assert port > 0


def test_ulimit_n_max():
    operating_system.ulimit_n_max()
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    assert soft_limit == hard_limit
