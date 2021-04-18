from flame.utils import experiment
from pathlib import Path


def test_experiment_dir():
    output_dir = Path('output')
    experiment_name = '000'

    debug_exp_dir = experiment.get_experiment_dir(
        output_dir, experiment_name, debug=True, with_timestamp=True)

    debug_exp_dir_str, _timestamp = str(debug_exp_dir).split('.')

    assert debug_exp_dir_str == str(output_dir / 'debug' / experiment_name)

    release_exp_dir = experiment.get_experiment_dir(
        output_dir, experiment_name, debug=False)

    assert release_exp_dir == output_dir / 'release' / experiment_name


def test_get_output_dir_and_experiment_name():
    output_dir = Path('output')
    experiment_name = '000'
    experiment_dir = experiment.get_experiment_dir(output_dir, experiment_name)

    output_dir1, experiment_name1 = experiment.get_output_dir_and_experiment_name(experiment_dir)

    assert output_dir == output_dir1
    assert experiment_name == experiment_name1

