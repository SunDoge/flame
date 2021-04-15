from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional
from torch import optim
import torch
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel, DataParallel
from flame.pytorch.typing_prelude import Optimizer, LrScheduler
import logging

_logger = logging.getLogger(__name__)


ToStateDict = Callable[[Any], dict]
LoadStateDict = Callable[[Any, dict], Any]


def module_to_state_dict(m: Module) -> dict:
    return m.state_dict()


def module_load_state_dict(m: Module, state: dict):
    m.load_state_dict(state)


def parallel_module_to_state_dict(m: DataParallel) -> dict:
    return m.module.state_dict()


def parallel_module_load_state_dict(m: DataParallel, state: dict):
    m.module.load_state_dict(state)

# def optimizer_to_state_dict(optimizer: Optimizer) -> dict:
#     return optimizer.state_dict()

# def optimizer_load_state_dict(optimizer: Optimizer, state: dict):
#     optimizer.load_state_dict(state)

# def scheduler_to_state_dict(scheduler: LrScheduler) ->dict:
#     return scheduler.state_dict()

# def scheduler_load_state_dict(scheduler: LrScheduler, state: dict):
#     scheduler.load_state_dict()


# @dataclass
# class Entry:
#     key: str
#     value: Any
#     to_state_dict: ToStateDict
#     load_state_dict: LoadStateDict

Entry = namedtuple(
    'Entry', ['key', 'to_state_dict', 'load_state_dict']
)


class CheckpointSaver:

    def __init__(self, entries=None) -> None:
        # self._to_state_dict: Dict[Any, ToStateDict] = {}
        # self._load_state_dict: Dict[Any, LoadStateDict] = {}
        self.entries: List[Entry] = entries if entries is not None else []

    # def register(self, type_: Any, to_state_dict: ToStateDict, load_state_dict: LoadStateDict):
    #     self._to_state_dict[type_] = to_state_dict
    #     self._load_state_dict[type_] = load_state_dict

    def register(self, key: str, to_state_dict: ToStateDict, load_state_dict: LoadStateDict):
        self.entries.append(Entry(key, to_state_dict, load_state_dict))

    def state_dict(self) -> dict:
        state = {}
        for key, to_state_dict, load_state_dict in self.entries:
            state[key] = to_state_dict()

        return state

    def load_state_dict(self, state: dict):
        for key, to_state_dict, load_state_dict in self.entries:
            load_state_dict(state[key])

    def save(self, filename: str):
        _logger.info('save checkpoint to: %s', filename)
        torch.save(self.state_dict(), filename)

    def load(self, filename: str):
        cp = torch.load(filename, map_location='cpu')
        self.load_state_dict(cp)
