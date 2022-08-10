

class BaseTrainer:

    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def initialize(self, args) -> int:
        """
        Must return max_epochs
        """
        pass

    def run(self, max_epochs: int = 1):
        pass

    def run_once(self, epoch: int):
        pass
