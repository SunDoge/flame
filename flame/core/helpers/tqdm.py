from tqdm import tqdm
import logging
from datetime import timedelta

_logger = logging.getLogger(__name__)


def tqdm_get_rate(pbar: tqdm) -> float:
    # _logger.debug('pbar.format_dict: %s', pbar.format_dict)

    rate = pbar.format_dict['rate']
    elapsed = pbar.format_dict['elapsed']
    initial = pbar.format_dict['initial']
    n = pbar.format_dict['n']

    if rate is None and elapsed:
        rate = (n - initial) / elapsed

    return rate


def tqdm_get_total_time(pbar: tqdm) -> timedelta:
    elapsed = pbar.format_dict['elapsed']
    return timedelta(seconds=elapsed)
