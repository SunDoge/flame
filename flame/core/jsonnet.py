import _jsonnet


def evaluate_file(filename: str, **kwargs) -> str:
    """eval file

    Args:
        filename: jsonnet file
    """
    return _jsonnet.evaluate_file(filename, **kwargs)


def evaluate_snippet(filename: str, expr: str, **kwargs) -> str:
    """eval snippet

    Args:
        filename: fake name for snippet
        expr: the snippet
    """
    return _jsonnet.evaluate_snippet(filename, expr, **kwargs)
