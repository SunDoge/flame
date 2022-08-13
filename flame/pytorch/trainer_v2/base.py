
from ctypes import Union
from typing import Any, Dict, List, TypedDict
from torch import nn, Tensor


class InputsOutputs(TypedDict):
    inputs: Union[Dict[str, Any], List[str]]
    outputs: Union[Dict[str, Any], List[str]]


class Dataclass:

    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def as_dict(self):
        return self.__dict__


class WeightedLosses(nn.Module):

    def __init__(
        self,
        weights: Dict[str, nn.Module],
        losses: Dict[str, nn.Module],
        config: Dict[str, InputsOutputs],
    ) -> None:
        super().__init__()

        for key in losses.keys():
            if key not in weights:
                weights[key] = 1.0

            assert key in config

        self.weights = weights
        self.losses = losses

    def forward(self, outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        pass


class ModelWithLosses(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        losses
    ) -> None:
        super().__init__()
