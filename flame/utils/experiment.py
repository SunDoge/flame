from pathlib import Path
import logging
import time
from datetime import datetime

_logger = logging.getLogger(__name__)


def get_experiment_dir(output_dir: Path, experiment_name: str, debug: bool = False) -> Path:
    """
    实验默认是release模式，只有指定debug模式的时候，才会生成debug目录

    experiment_dir = output_dir / release or debug / experiment_name
    e.g. exps/release/000
    """
    if debug:
        # FIXME: 这里的操作不利于测试
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = experiment_name + '.' + timestamp

    return output_dir / ('debug' if debug else 'release') / experiment_name


def make_experiment_dir(output_dir: Path, experiment_name: str, debug: bool = False, yes: bool = False) -> Path:
    experiment_dir = get_experiment_dir(
        output_dir, experiment_name, debug=debug)

    if experiment_dir.exists():
        _logger.warning('experiment dir %s exists', experiment_dir)
        _logger.info('Do you want to move %s to debug folder?', experiment_dir)
        _logger.info('(Y)es/(N)o')

        if yes:
            choice = 'yes'
        else:
            choice = input()

        if choice.lower() in {'y', 'yes'}:
            target_debug_dir = get_experiment_dir(
                output_dir, experiment_name, debug=True
            )

            target_debug_dir.parent.mkdir(parents=True, exist_ok=True)

            _logger.info('move %s to %s', experiment_dir, target_debug_dir)
            experiment_dir.rename(target_debug_dir)
        else:
            raise FileExistsError(experiment_dir)

    _logger.info('make experiment_dir: %s', experiment_dir)
    experiment_dir.mkdir(parents=True)

    return experiment_dir
