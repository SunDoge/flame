"""
下次改掉原有container的unittest
"""

from flame.pytorch.experimental.container.container import Container, cached_inject
from injector import singleton


@singleton
class A:

    def __init__(self) -> None:
        self.foo = 'foo'


class C(Container):

    @cached_inject
    def call_with_a(a: A, input_a=None):
        assert a is input_a

    @cached_inject
    def call_with_kwargs(a: A, input_a=None, train=False):
        assert train
        assert a is input_a


def test_c():
    c = C()
    a1 = c._injector.get(A)
    c.call_with_a(input_a=a1)
    c.call_with_kwargs(input_a=a1, train=True)

    a2 = A()
    C.call_with_a(a2, input_a=a2)
    C.call_with_kwargs(a2, input_a=a2, train=True)
