from injector import Injector
from flame.pytorch.experimental.singleton_registry import configure_singletons


class A:
    def __init__(self, **kwargs) -> None:
        print(self, kwargs)


class B:
    def __init__(self, **kwargs) -> None:
        print(self, kwargs)


def test_build():

    class_a_path = 'tests.test_experimental.test_new_builder.A'
    class_b_path = 'tests.test_experimental.test_new_builder.B'

    config = {
        'A1': {
            '_builder': class_a_path,
            'x': '@B',
            'y': 1
        },
        'B': {
            '_builder': class_b_path,
            '_scope': 'singleton',
        },
        'A2': {
            '_builder': class_a_path,
            'a': '@A1'
        }
    }
    container = Injector(configure_singletons(config))

    b1 = container.get('B')
    b2 = container.get('B')
    assert b1 is b2

    a1 = container.get('A1')
    a2 = container.get('A2')
