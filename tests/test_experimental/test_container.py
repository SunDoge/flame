from flame.pytorch.experimental.engines.container import Container
from injector import inject


class A:

    def __init__(self) -> None:
        self.foo = 'foo'


@inject
def _func1(a: A, b=None):
    assert a.foo == 'foo'
    assert b == 'bar'


def test_container():
    container = Container()
    func1 = container.with_injection(_func1)
    func1(b='bar')
