from rich.logging import RichHandler
import logging


def init_logger(rank: int = 0, debug: bool = False):
    """

    Args:
        rank: 目前只有rank0会输出到console和log file
    """

    if rank != 0:
        return

    
