from flame import argument
from flame.argument import BasicArgs
from pathlib import Path


def test_basic_args():
    args = BasicArgs.from_args(
        '-c config1 -c config2 -o exps -e 001 -dy'.split()
    )
    assert args == BasicArgs(
        config=['config1', 'config2'],
        local=[],
        output_dir=Path('exps'),
        experiment_name='001',
        debug=True, yes=True
    )
