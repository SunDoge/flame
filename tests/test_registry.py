from flame.registry import Registry
import pytest

def test_registry():
    """
    https://github.com/facebookresearch/fvcore/blob/f08e0c3764ed15dc7344a2a1722e58427c923369/tests/test_common.py#L201
    """

    OBJECT_REGISTRY = Registry("OBJECT")

    @OBJECT_REGISTRY.register()
    class Object1:
        def __init__(self, name: str) -> None:
            pass

    with pytest.raises(AssertionError):
        OBJECT_REGISTRY.register(Object1)

    assert OBJECT_REGISTRY.get('Object1') is Object1

    with pytest.raises(KeyError):
        OBJECT_REGISTRY.get('Object2')

    items = list(OBJECT_REGISTRY)

    assert items == [("Object1", Object1)]
    assert Object1 in OBJECT_REGISTRY
    assert 'Object1' in OBJECT_REGISTRY
    assert 'Object2' not in OBJECT_REGISTRY
