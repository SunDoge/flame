from typing import Optional
from rich.logging import RichHandler
import logging
import sys

_logger = logging.getLogger(__name__)


FILE_FORMAT = '%(asctime)s:%(levelname)s:%(name)s:%(message)s'
CONSOLE_FORMAT = "%(message)s"
# CONSOLE_FORMAT = "%(asctime)s %(levelname)-8s: %(message)s"


def log_hook(exc_type, exc_value, traceback):
    """
    log the exception when it raise
    """
    _logger.exception("Uncaught exception", exc_info=(
        exc_type, exc_value, traceback))


def set_excepthook():
    _logger.debug('Setting except hook')
    sys.excepthook = log_hook


def get_file_handler(filename: str, fmt: str = FILE_FORMAT):
    file_handler = logging.FileHandler(filename)
    formatter = logging.Formatter(fmt)
    file_handler.setFormatter(formatter)
    return file_handler


def get_console_handler():
    formatter = logging.Formatter(CONSOLE_FORMAT)
    console_handler = RichHandler()
    # console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    return console_handler


def init_logger(rank: int = 0, filename: Optional[str] = None, debug: bool = False):
    """

    Args:
        rank: 目前只有rank0会输出到console和log file
    """

    # 如果不是main process，设置level后退出
    if rank != 0:
        logging.basicConfig(level=logging.CRITICAL)
        return

    handlers = []
    console_handler = get_console_handler()
    handlers.append(console_handler)

    if filename is not None:
        file_handler = get_file_handler(filename)
        handlers.append(file_handler)

    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        handlers=handlers
    )

    set_excepthook()
