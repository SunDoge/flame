import logging
from typing import Optional
from rich import console
from tqdm import tqdm as _tqdm
from rich.console import Console

FILE_FORMAT = "%(asctime)s|%(levelname)-8s|%(message)s"
CONSOLE_FORMAT = '%(asctime)s|%(levelname)-8s|%(message)s'
CONSOLE_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'  # 不需要精确到毫秒


class TqdmHandler(logging.Handler):

    def __init__(self, level=logging.NOTSET) -> None:
        super().__init__(level=level)
        self.console = Console()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)

            # Colorize
            with self.console.capture() as capture:
                self.console.print(msg, end='')
            msg = capture.get()

            _tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def remove_all_handlers(name: Optional[str] = None):
    logger = logging.getLogger(name=name)
    for h in logger.handlers:
        logger.removeHandler(h)
        h.close()


def create_console_handler(fmt: str = CONSOLE_FORMAT, datefmt: str = CONSOLE_DATE_FORMAT):
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    console_handler = TqdmHandler()
    console_handler.setFormatter(formatter)
    return console_handler


def create_file_handler(filename: str, fmt: str = FILE_FORMAT):
    formatter = logging.Formatter(fmt)
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    return file_handler


def init_logging(
    filename: Optional[str] = None,
    debug: bool = False,
    force: bool = True,
):
    if force:
        remove_all_handlers()

    handlers = []
    console_handler = create_console_handler()
    handlers.append(console_handler)

    if filename is not None:
        file_handler = create_file_handler(filename)
        handlers.append(file_handler)

    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        handlers=handlers
    )
