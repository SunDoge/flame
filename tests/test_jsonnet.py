from flame.utils import jsonnet
import json


def test_eval_snippet():
    expr = '{batch_size: 16}'
    json_str = jsonnet.evaluate_snippet('snippet', expr)
    json_obj = json.loads(json_str)
    assert json_obj['batch_size'] == 16


def test_eval_file():
    json_str = jsonnet.evaluate_file('tests/fake_config/batch_size_16.jsonnet')
    json_obj = json.loads(json_str)
    assert json_obj['batch_size'] == 16
