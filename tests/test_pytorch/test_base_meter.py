from flame.pytorch.meters.base_meter import Meter


class MeterA(Meter):

    def __init__(self, total: int) -> None:
        super().__init__()

        self._total = total

        with self.resettable():
            self._count = 0
            self._value = None

class MeterB:

    def __init__(self) -> None:
        self.foo = 1
        self.reset()

    def reset(self):
        self.val = 0
        self.count = 0
        self.sum = 0

    def avg(self):
        self.foo
        self.val
        self.sum
        return self.sum / self.val


def test_resettable():
    pass
