from .base_meter import BaseMeter


class NaiveAverageMeter(BaseMeter):

    def __init__(self, name: str, fmt: str = ':f') -> None:
        super().__init__()

        self.name = name
        self.fmt = fmt

        self.val = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    @property
    def avg(self):
        return self.sum / self.count

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n

    def __str__(self) -> str:
        fmt_str: str = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        # _logger.debug('fmt_str: %s', fmt_str)
        return fmt_str.format(name=self.name, val=self.val, avg=self.avg)
