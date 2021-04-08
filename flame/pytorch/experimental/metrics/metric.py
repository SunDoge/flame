from typing import Any, Callable, Dict, Iterable, List, Sequence, Union
from flame.pytorch.experimental.engine import Engine, Events, State


def _default_output_transform(x):
    return x


class Metric:

    def __init__(
        self,
        name: Union[str, Sequence[str]],
        output_transform: Callable = _default_output_transform
    ) -> None:
        self._name = name
        self._output_transform = output_transform

    def compute(self, *args, **kwargs):
        """计算metric"""
        pass

    def update(self, state: State):
        """写入metric"""
        args = self._output_transform(state.output)
        result = self.compute(*args)
        if isinstance(self._name, Iterable):
            for n, r in zip(self._name, result):
                state.metrics[n] = r
        else:
            state.metrics[self._name] = result

    def attach(self, engine: Engine, event_name: Any = Events.ITERATION_COMPLETED):
        engine.add_event_handler(event_name, self.update)
