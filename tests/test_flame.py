from flame import __version__
import re


def get_version_from_pyproject_toml() -> str:
    with open('pyproject.toml', 'r') as f:
        content = f.read()

    matches = re.findall(r"version = \"(.*?)\"", content)
    assert len(matches) == 1
    return matches[0]


def test_version():
    version = get_version_from_pyproject_toml()
    assert __version__ == version
