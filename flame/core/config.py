"""
暂时使用dict作为config
"""


import json
import logging
from pathlib import Path
from typing import List, Union

from . import jsonnet

_logger = logging.getLogger(__name__)

PATCH_FILE = 'patch.libsonnet'


def from_file(filename: str) -> dict:
    """
    读取jsonnet格式的config
    """
    json_str = jsonnet.evaluate_file(filename)
    json_obj = json.loads(json_str)
    return json_obj


def from_snippet(expr: str, filename: str = 'snippet.jsonnet') -> dict:
    """
    helper function for evaluating snippet to python dict

    Args:
        expr: jsonnet expression
        filename: fake file name for snippet
    """
    json_str = jsonnet.evaluate_snippet(filename, expr)
    json_obj = json.loads(json_str)
    return json_obj


def dump_to_json(cfg: dict, filename: str):
    """
    保存config为json文件。jsonnet可以直接读取json文件。
    """
    _logger.info('dumping config to %s', filename)
    with open(filename, 'w') as f:
        json.dump(cfg, f, indent=2)


def merge_jsonnet(main_snippet: str, added: str, patch_file: Union[str, Path]) -> str:

    template = f"""{main_snippet}
+(import '{patch_file}').{added}
    """
    return template


def parse_config(config_file: str, patches: List[str]) -> str:
    parent_dir = Path(config_file).parent
    main = f"""(import '{config_file}')
    """
    for patch in patches:
        main = merge_jsonnet(main, patch, parent_dir / PATCH_FILE)

    return main
