from flame import auto_builder


def test_signature():
    def f(x, y=1):
        pass

    assert auto_builder.signature_contains(f, 'x')
    assert not auto_builder.signature_contains(f, 'a')


def test_get_kwargs():
    """
    测试是否可以正确排除type_key
    """
    cfg = {
        '_type': 'anything',
    }
    kwargs = {
        'a': 1, 'b': 2
    }
    cfg.update(kwargs)

    output = auto_builder.get_kwargs_from_config(cfg, type_key='_type')
    assert output == kwargs


class A:
    def __init__(self, v: int) -> None:
        self.v = v


class B:
    def __init__(self, a: A) -> None:
        self.a = a


def test_build_combination():
    prefix = __name__
    v = 1
    cfg = {
        '_type': f'{prefix}.B',
        'a': {
            '_type': f'{prefix}.A',
            'v': v,
        }
    }
    b: B = auto_builder.build_from_config(cfg)
    assert b.a.v == v


def test_build_with_args():
    prefix = __name__
    cfg = {
        '_type': f'{prefix}.A',
    }
    a: A = auto_builder.build_from_config(cfg, 1)
    assert a.v == 1