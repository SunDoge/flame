import sys
import importlib
from typing import Optional


def import_any(module_name: str, object_name: Optional[str] = None):
    module = importlib.import_module(module_name)
    if object_name is not None:
        ret = getattr(module, object_name)
    else:
        ret = module
    return ret


def import_from_zip(zip_file: str, module_name: str, object_name: Optional[int] = None):
    sys.path.insert(0, zip_file)
    ret = import_any(module_name, object_name=object_name)
    sys.path.pop(0)
    return ret
