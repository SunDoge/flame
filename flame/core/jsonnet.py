import os

import rjsonnet


#  Returns content if worked, None if file not found, or throws an exception
def try_path(dir: str, rel: str):
    if not rel:
        raise RuntimeError('Got invalid filename (empty string).')
    if rel[0] == '/':
        full_path = rel
    else:
        full_path = dir + '/' + rel
    if full_path[-1] == '/':
        raise RuntimeError('Attempted to import a directory')

    if not os.path.isfile(full_path):
        return full_path, None
    with open(full_path) as f:
        return full_path, f.read()


def import_callback(dir: str, rel: str):
    full_path, content = try_path(dir, rel)
    if content:
        return full_path, content
    raise RuntimeError('File not found')


def evaluate_file(filename: str, import_callback=import_callback, **kwargs) -> str:
    """eval file

    Args:
        filename: jsonnet file
    """
    return rjsonnet.evaluate_file(filename, import_callback=import_callback, **kwargs)


def evaluate_snippet(filename: str, expr: str, import_callback=import_callback, **kwargs) -> str:
    """eval snippet

    Args:
        filename: fake name for snippet
        expr: the snippet
    """
    return rjsonnet.evaluate_snippet(filename, expr, import_callback=import_callback, **kwargs)
