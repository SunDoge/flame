import _jsonnet


def evaluate_file(filename: str, **kwargs) -> str:
    return _jsonnet.evaluate_file(filename, **kwargs)


def evaluate_snippet(filename: str, expr: str, **kwargs) -> str:
    return _jsonnet.evaluate_snippet(filename, expr, **kwargs)
