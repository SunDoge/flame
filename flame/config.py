from typing import List
from .utils import jsonnet
import json
import os


def from_file(filename: str) -> dict:
    """
    读取jsonnet格式的config
    """
    json_str = jsonnet.evaluate_file(filename)
    json_obj = json.loads(json_str)
    return json_obj


def from_snippet(expr: str, filename='snippet.jsonnet') -> dict:
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
        snippet += process_local_variable(local_variable)
        snippet += '\n'

    contents = []
    for file_or_snippet in files_or_snippets:
        content =  process_file_or_snippet(file_or_snippet)
        contents.append(content)

    snippet += '+'.join(contents)
        

    return snippet


def process_local_variable(local_variable: str) -> str:
    key, value = local_variable.split('=')
    if os.path.isfile(value):
        return f"local {key} = import '{value}';"
    else:
        return f"local {key} = {value};"


def process_file_or_snippet(file_or_snippet: str) -> str:
    if os.path.isfile(file_or_snippet):
        return f"(import '{file_or_snippet}')"
    else:
        return f"{file_or_snippet}"
