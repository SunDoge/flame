
from typing import Mapping, OrderedDict


class Serializable:
    """
    作为Engine的基类
    """

    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Mapping):
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                f"Argument state_dict should be a dictionary, but given {type(state_dict)}"
            )

        
