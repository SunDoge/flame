"""
暂时使用dict作为config
"""


from typing import List, Tuple
from .utils import jsonnet
import json
import os
import logging
import difflib

_logger = logging.getLogger(__name__)


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


def config_snippet(local_variables: List[str], files_or_snippets: List[str]) -> str:
    """config snippet

    The snippet looks like::

        {{local_variables}}

        {{(file_or_snippet) + (file_or_snippet) + (file_or_snippet)}}

    Convert strategy::

        local_variables: "a=a.jsonnet" -> local a = import "a.jsonnet"; "a=1" -> local a = 1;
        files_or_snippets: "{a:1}" -> ({a:1}), "a.jsonnet" -> (import "a.jsonnet")


    """
    snippet = ''
    for local_variable in local_variables:
        snippet += _process_local_variable(local_variable)
        snippet += '\n'

    contents = []
    for file_or_snippet in files_or_snippets:
        content = _process_file_or_snippet(file_or_snippet)
        contents.append(content)

    snippet += '+'.join(contents)

    return snippet


def _process_local_variable(local_variable: str) -> str:
    key, value = local_variable.split('=')
    if os.path.isfile(value):
        return f"local {key} = import '{value}';"
    else:
        return f"local {key} = {value};"


def _process_file_or_snippet(file_or_snippet: str) -> str:
    if os.path.isfile(file_or_snippet):
        return f"(import '{file_or_snippet}')"
    else:
        return f"{file_or_snippet}"


def dump_as_json(cfg: dict, filename: str):
    """
    保存config为json文件。jsonnet可以直接读取json文件。
    """
    _logger.info('dumping config to %s', filename)
    with open(filename, 'w') as f:
        json.dump(cfg, f, indent=2)


def parse_config(local_variables: List[str], files_or_snippets: List[str]) -> Tuple[dict, str]:
    snippet_before = config_snippet(local_variables, files_or_snippets[:1])
    snippet_after = config_snippet(local_variables, files_or_snippets)

    json_before = jsonnet.evaluate_snippet('snippet', snippet_before)
    json_after = jsonnet.evaluate_snippet('snippet', snippet_after)

    diff = difflib.unified_diff(
        json_before.splitlines(keepends=True),
        json_after.splitlines(keepends=True),
        fromfile='before.json',
        tofile='after.json'
    )

    diff_str = '\n'.join(diff)

    cfg = json.loads(json_after)

    return cfg, diff_str
