from flame import config
from flame.utils import jsonnet
from pathlib import Path
import rich

def test_from_snippet():
    snippet = "{a: 1}"
    cfg = config.from_snippet(snippet)
    assert cfg == {'a': 1}


# def test_config_snippet():
#     snippet = config.config_snippet(
#         [
#             'a=1', 'b=tests/fake_config/batch_size_16.jsonnet'
#         ],
#         [
#             "{c:3, a:a}", "tests/fake_config/batch_size_16.jsonnet"
#         ]
#     )

#     cfg = config.from_snippet(snippet)

#     assert cfg == {'batch_size': 16, 'c': 3, 'a': 1}


# def test_parse_config():
#     cfg_file_path = "tests/fake_config/batch_size_16.jsonnet"
#     cfg, diff = config.parse_config([], [cfg_file_path])
#     assert cfg == {'batch_size': 16}

#     rich.print(diff)
