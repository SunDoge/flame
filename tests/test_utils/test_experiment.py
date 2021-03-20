from flame.utils import experiment
from pathlib import Path


def test_experiment_dir():
    output_dir = Path('output')
    experiment_name = '000'

    debug_exp_dir = experiment.get_experiment_dir(
        output_dir, experiment_name, debug=True)

    debug_exp_dir_str, _timestamp = str(debug_exp_dir).split('.')

    assert debug_exp_dir_str == str(output_dir / 'debug' / experiment_name)

    release_exp_dir = experiment.get_experiment_dir(
        output_dir, experiment_name, debug=False)

    assert release_exp_dir == output_dir / 'release' / experiment_name
