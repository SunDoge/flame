from pathlib import Path


def get_experiment_dir(output_dir: Path, experiment_name: str, debug: bool = False) -> Path:
    """
    实验默认是release模式，只有指定debug模式的时候，才会生成debug目录

    experiment_dir = output_dir / release or debug / experiment_name
    e.g. exps/release/000
    """
    return output_dir / ('debug' if debug else 'release') / experiment_name


