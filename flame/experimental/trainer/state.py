from typing import Dict, Optional


class State:
    def __init__(self) -> None:
        self.epoch: int = 0
        self.step: int = 0
        self.training: bool = True
        self.epoch_length: Optional[int] = None

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict: Dict):
        for key, value in state_dict.items():
            setattr(self, key, value)

    def train(self, mode: bool = True):
        self.training = mode

    def eval(self):
        self.train(mode=False)
