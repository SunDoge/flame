import logging
from tqdm import tqdm as _tqdm

FILE_FORMAT = "%(asctime)s|%(levelname)-8s|%(message)s"
CONSOLE_FORMAT = '%(asctime)s|%(levelname)-8s|%(message)s'
CONSOLE_DATE_FORMAT = '%Y-%m-%d %H:%M:%s'  # 不需要精确到毫秒


class TqdmHandler(logging.Handler):

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            _tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)



def init_logger():
    pass
