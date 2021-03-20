import socket
import resource
import logging

_logger = logging.getLogger(__name__)


def find_free_port() -> int:
    """find a free port for distributed training automatically
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        _host, port = s.getsockname()

    return port


def ulimit_n_max():
    """Raise ulimit to its max value

    在某些情况下，dataloader会把文件描述符用完

    """
    _soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)

    _logger.warning('setting ulimit -n %d', hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))
