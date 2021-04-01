from typing import Any, Callable, DefaultDict, Iterable, List, Optional, Tuple, Union
from .mixins import Serializable
from collections import defaultdict
from .events import Events, EventsList, State
from injector import Injector
import logging

_logger = logging.getLogger(__name__)


HandlerTuple = Tuple[Callable, tuple, dict]
EventHandlerDict = DefaultDict[Events, List[HandlerTuple]]


def _default_process_function(batch: Any) -> Any:
    return batch


class Engine(Serializable):

    def __init__(
        self,
        process_function: Callable = _default_process_function,
        container: Optional[Injector] = None
    ):

        if container is None:
            _logger.debug('Init empty DI container')
            container = Injector()

        state = State()

        container.binder.bind(Engine, to=self)
        container.binder.bind(State, to=state)

        self._event_handlers: EventHandlerDict = defaultdict(list)

        self.last_event_name: Optional[Events] = None
        self.should_terminate = False
        self.should_terminate_single_epoch = False
        self.state = state
        self.container = container

    def add_event_handler(
        self,
        event_name: Union[Events, EventsList],
        handler: Callable,
        *args: Any, **kwargs: Any
    ):
        if isinstance(event_name, EventsList):
            for e in event_name:
                self.add_event_handler(e, handler, *args, **kwargs)

            return

        self._event_handlers[event_name].append((handler, args, kwargs))

    def has_event_handler(self, handler: Callable, event_name: Optional[Any] = None) -> bool:
        """
        FIXME: 实现不是很正确
        """
        if event_name is not None:
            if event_name not in self._event_handlers:
                # 不存在的event name
                return False

            events = [event_name]
        else:
            # 查询所有event
            events = self._event_handlers.keys()

        for e in events:
            for h, _, _ in self._event_handlers[e]:
                if self.compare_handlers(handler, h):
                    return True
        return False

    @staticmethod
    def compare_handlers(user_handler: Callable, registered_handler: Callable) -> bool:
        """
        FIXME: 比较不是很准确
        """
        wrapped = getattr(
            registered_handler,
            '__wrapped__',
            default=registered_handler
        )
        return wrapped == user_handler

    def _fire_event(self, event_name: Any, *event_args, **event_kwargs):
        """
        最好不要用args，顺序很难确定
        """
        _logger.debug(f"firing handlers for event {event_name}")
        self.last_event_name = event_name
        for func, args, kwargs in self._event_handlers[event_name]:
            new_kwargs = {**kwargs, **event_kwargs}
            new_args = event_args + args
            self.container.call_with_injection(
                func, args=new_args, kwargs=new_kwargs
            )

    def fire_event(self, event_name: Any):
        self._fire_event(event_name)

    def step_epoch(self, epoch: int):
        pass

    def set_epoch(self, epoch: int):
        self.state.epoch = epoch

    def setup(
        self,
        data: Iterable,
        max_epochs: Optional[int] = None,
        max_iters: Optional[int] = None,
        epoch_length: Optional[int] = None,
    ):
        if not isinstance(data, Iterable):
            raise TypeError("Argument data should be iterable")

        if self.state.max_epochs is not None:
            # Check and apply overridden parameters
            if max_epochs is not None:
                if max_epochs < self.state.epoch:
                    raise ValueError(
                        "Argument max_epochs should be larger than the start epoch "
                        f"defined in the state: {max_epochs} vs {self.state.epoch}. "
                        "Please, set engine.state.max_epochs = None "
                        "before calling engine.run() in order to restart the training from the beginning."
                    )
                self.state.max_epochs = max_epochs

            if epoch_length is not None:
                if epoch_length != self.state.epoch_length:
                    raise ValueError(
                        "Argument epoch_length should be same as in the state, "
                        f"but given {epoch_length} vs {self.state.epoch_length}"
                    )

            self.state.dataloader = data
