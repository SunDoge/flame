from flame.core import operating_system
import resource
import sys
import logging

_logger = logging.getLogger(__name__)

def test_find_free_port():
    port = operating_system.find_free_port()
    assert port > 0


def test_ulimit_n_max():
    if not 'linux' in sys.platform:
        _logger.warning('this is only supported on linux')
        return

    operating_system.ulimit_n_max()
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    assert soft_limit == hard_limit
