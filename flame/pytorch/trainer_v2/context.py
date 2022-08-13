from ctypes import Union
from curses import keyname
from typing import Any, Callable, Dict, List, Sequence, Tuple, TypeVar, TypedDict


class MappingConfig(TypedDict):
    inputs: Union[Dict[str, Any], List[str]]
    outputs: Union[Dict[str, Any], List[str]]


R = TypeVar('R')


def remap_inputs(
    inputs: Union[Dict[str, Any], Tuple[Any, ...]],
    config: Union[Dict[str, Any], List[str]]
) -> Union[Dict, List]:
    """

    if inputs is dict, config can be list or dict
        if config is list, output is list
            we select value using keys in config
        if config is dict, output is dict
            we select value using key and remap it to new key
            the config is in [new_key, key] form

    if inputs is tuple, config must be list
        we name the items in inputs using keys in config

    we restrict inputs to tuple, so that list can be named correctly
    """

    if isinstance(inputs, tuple):
        assert isinstance(config, list)
        assert len(inputs) == len(config)

        outputs = {k: v for k, v in zip(config, inputs)}
        return outputs

    elif isinstance(inputs, dict):
        if isinstance(config, list):
            outputs = [inputs[k] for k in config]
            return outputs
        elif isinstance(config, dict):
            outputs = {new_key: inputs[key] for new_key, key in config.items()}
            return outputs

    else:
        raise NotImplementedError


def remap_outputs(
    outputs: Union[Dict[str, Any], Tuple[Any, ...]],
    config: Union[Dict[str, str], List[str]]
) -> Dict[str, Any]:
    """


    if output is tuple, config must be list
        we name every output

    if output is dict, config can be list or dict
        if config is list
            we select some output
        if config is dict
            config is in [old_key, new_key] so some keys are rewrite


    """

    if isinstance(outputs, tuple):
        assert isinstance(config, list)
        assert len(outputs) == len(config)
        res = {k: v for k, v in zip(config, outputs)}
        return res

    elif isinstance(outputs, dict):
        if isinstance(config, list):
            res = {k: outputs[k] for k in config}
            return res
        elif isinstance(config, dict):
            res = {}
            for key, value in outputs.items():
                new_key = config.get(key, key)
                res[new_key] = value
            return res

    else:
        raise NotImplementedError


def call_with_mapping(func: Callable[..., R], mapping_config: MappingConfig, inputs: Dict[str, Any]) -> R:
    pass
