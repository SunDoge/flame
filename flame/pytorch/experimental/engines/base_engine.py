from typing import Any, Callable, DefaultDict, Iterable, List, Optional, Tuple, Union
from .mixins import Serializable
from collections import defaultdict
from .events import Events, EventsList, State
from injector import Injector
import logging

_logger = logging.getLogger(__name__)


HandlerTuple = Tuple[Callable, tuple, dict]
EventHandlerDict = DefaultDict[Events, List[HandlerTuple]]


class BaseEngine(Serializable):

    def __init__(
        self,
        container: Optional[Injector] = None
    ):

        if container is None:
            _logger.debug('Init empty DI container')
            container = Injector()

        self._event_handlers: EventHandlerDict = defaultdict(list)

        self.last_event_name: Optional[Events] = None
        self.should_terminate = False
        self.should_terminate_single_epoch = False
        self.state = State()
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
            kwargs.update(event_kwargs)
            new_args = (self,) + event_args + args
            self.container.call_with_injection(
                func, args=new_args, kwargs=kwargs
            )

    def fire_event(self, event_name: Any):
        self._fire_event(event_name)
