from enum import Enum
from typing import Any, Dict, Iterable, Iterator, List, Optional, TYPE_CHECKING, Callable, Union
from dataclasses import dataclass, field
import logging
from types import DynamicClassAttribute
import numbers

if TYPE_CHECKING:
    from .engine import Engine


_logger = logging.getLogger(__name__)


class CallableEventWithFilter:
    """Single Event containing a filter, specifying whether the event should
    be run at the current event (if the event type is correct)

    Args:
        value: The actual enum value. Only needed for internal use. Do not touch!
        event_filter: A function taking the engine and the current event value as input and returning a
            boolean to indicate whether this event should be executed. Defaults to None, which will result to a
            function that always returns `True`
        name: The enum-name of the current object. Only needed for internal use. Do not touch!
    """

    def __init__(self, value: str, event_filter: Optional[Callable] = None, name: Optional[str] = None) -> None:
        if event_filter is None:
            event_filter = CallableEventWithFilter.default_event_filter
        self.filter = event_filter

        if not hasattr(self, "_value_"):
            self._value_ = value

        if not hasattr(self, "_name_") and name is not None:
            self._name_ = name

    # copied to be compatible to enum
    @DynamicClassAttribute
    def name(self) -> str:
        """The name of the Enum member."""
        return self._name_

    @DynamicClassAttribute
    def value(self) -> str:
        """The value of the Enum member."""
        return self._value_

    def __call__(
        self, event_filter: Optional[Callable] = None, every: Optional[int] = None, once: Optional[int] = None
    ) -> "CallableEventWithFilter":
        """
        Makes the event class callable and accepts either an arbitrary callable as filter
        (which must take in the engine and current event value and return a boolean) or an every or once value

        Args:
            event_filter: a filter function to check if the event should be executed when
                the event type was fired
            every: a value specifying how often the event should be fired
            once: a value specifying when the event should be fired (if only once)

        Returns:
            CallableEventWithFilter: A new event having the same value but a different filter function
        """

        if not ((event_filter is not None) ^ (every is not None) ^ (once is not None)):
            raise ValueError(
                "Only one of the input arguments should be specified")

        if (event_filter is not None) and not callable(event_filter):
            raise TypeError("Argument event_filter should be a callable")

        if (every is not None) and not (isinstance(every, numbers.Integral) and every > 0):
            raise ValueError(
                "Argument every should be integer and greater than zero")

        if (once is not None) and not (isinstance(once, numbers.Integral) and once > 0):
            raise ValueError("Argument once should be integer and positive")

        if every is not None:
            if every == 1:
                # Just return the event itself
                event_filter = None
            else:
                event_filter = self.every_event_filter(every)

        if once is not None:
            event_filter = self.once_event_filter(once)

        # check signature: FIXME: I remove the checking
        # if event_filter is not None:
        #     _check_signature(event_filter, "event_filter", "engine", "event")

        return CallableEventWithFilter(self.value, event_filter, self.name)

    @staticmethod
    def every_event_filter(every: int) -> Callable:
        """A wrapper for every event filter."""

        def wrapper(engine: "Engine", event: int) -> bool:
            if event % every == 0:
                return True
            return False

        return wrapper

    @staticmethod
    def once_event_filter(once: int) -> Callable:
        """A wrapper for once event filter."""

        def wrapper(engine: "Engine", event: int) -> bool:
            if event == once:
                return True
            return False

        return wrapper

    @staticmethod
    def default_event_filter(engine: "Engine", event: int) -> bool:
        """Default event filter."""
        return True

    def __str__(self) -> str:
        return "<event=%s, filter=%r>" % (self.name, self.filter)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, CallableEventWithFilter):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        else:
            return NotImplemented

    def __hash__(self) -> int:
        return hash(self._name_)

    def __or__(self, other: Any) -> "EventsList":
        return EventsList() | self | other


class EventEnum(CallableEventWithFilter, Enum):  # type: ignore[misc]
    """Base class for all :class:`~ignite.engine.events.Events`. User defined custom events should also inherit
    this class. For example, Custom events based on the loss calculation and backward pass can be created as follows:

        .. code-block:: python

            from ignite.engine import EventEnum

            class BackpropEvents(EventEnum):
                BACKWARD_STARTED = 'backward_started'
                BACKWARD_COMPLETED = 'backward_completed'
                OPTIM_STEP_COMPLETED = 'optim_step_completed'

            def update(engine, batch):
                # ...
                loss = criterion(y_pred, y)
                engine.fire_event(BackpropEvents.BACKWARD_STARTED)
                loss.backward()
                engine.fire_event(BackpropEvents.BACKWARD_COMPLETED)
                optimizer.step()
                engine.fire_event(BackpropEvents.OPTIM_STEP_COMPLETED)
                # ...

            trainer = Engine(update)
            trainer.register_events(*BackpropEvents)

            @trainer.on(BackpropEvents.BACKWARD_STARTED)
            def function_before_backprop(engine):
                # ...
    """

    pass


class Events(EventEnum):
    # For epoch
    EPOCH_STARTED = 'epoch_started'
    EPOCH_COMPLETED = 'epoch_completed'

    # For training
    STARTED = 'started'
    COMPLETED = 'completed'

    # For iteration
    ITERATION_STARTED = 'iteration_started'
    ITERATION_COMPLETED = 'iteration_completed'
    EXCEPTION_RAISED = 'exception_raised'  # FIXME: 不确定有什么用

    # For data
    GET_BATCH_STARTED = 'get_batch_started'
    GET_BATCH_COMPLETED = 'get_batch_completed'

    # 这几个理论上都用不到
    DATALOADER_STOP_ITERATION = 'dataloader_stop_iteration'
    TERMINATE = 'terminate'
    TERMINATE_SINGLE_EPOCH = 'terminate_single_epoch'

    def __or__(self, other: Any) -> "EventsList":
        return EventsList() | self | other


class EventsList:

    def __init__(self) -> None:
        self._events: List[Events] = []

    def _append(self, event: Events):
        self._events.append(event)

    def __getitem__(self, index: int) -> Events:
        return self._events[index]

    def __iter__(self) -> Iterator[Events]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def __or__(self, other: Events) -> 'EventsList':
        self._append(other)
        return self


def _get_default_event_to_attr():
    event_to_attr = {
        Events.GET_BATCH_STARTED: "global_iteration",
        Events.GET_BATCH_COMPLETED: "global_iteration",
        Events.ITERATION_STARTED: "global_iteration",
        Events.ITERATION_COMPLETED: "global_iteration",
        Events.EPOCH_STARTED: "epoch",
        Events.EPOCH_COMPLETED: "epoch",
        Events.STARTED: "epoch",
        Events.COMPLETED: "epoch",
    }  # type: Dict[Union[str, "Events", "CallableEventWithFilter"], str]
    return event_to_attr


@dataclass
class State:
    # For epoch engine
    epoch: int = 0
    max_epochs: Optional[int] = None

    # For iteration engine
    # 内循环的iter
    local_iteration: int = 0
    # 总的iter
    global_iteration: int = 0

    # dataloader的长度，和local_iteration相关
    epoch_length: Optional[int] = None

    # 最多跑多少iter，和global_iteration相关
    max_iterations: Optional[int] = None

    batch: Optional[Any] = None  # model input
    output: Optional[Any] = None  # model output
    dataloader: Optional[Iterable[Any]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    event_to_attr: Dict[Events, str] = field(
        default_factory=_get_default_event_to_attr
    )

    def update_max_iterations(self):
        # 自动计算max_iterations
        if self.max_epochs is not None and self.epoch_length is not None and self.max_iterations is None:
            old_max_iterations = self.max_iterations
            self.max_iterations = self.max_epochs * self.epoch_length
            _logger.info(
                'update max_iterations: %s => %s',
                old_max_iterations, self.max_iterations
            )
        else:
            _logger.warning('max_iterations is not updated')

    def update_local_iteration(self, iteration: int):
        self.local_iteration = iteration

    def update_global_iteration(self):
        self.global_iteration += 1

    def is_done_iterations(self) -> bool:
        return self.max_iterations is not None and self.global_iteration >= self.max_iterations

    def is_done_epochs(self) -> bool:
        return self.max_epochs is not None and self.epoch >= self.max_epochs

    def is_done_count(self) -> bool:
        return (
            self.epoch_length is not None
            and self.max_epochs is not None
            and self.global_iteration >= self.epoch_length * self.max_epochs
        )

    def is_done(self) -> bool:
        # 终止条件
        return self.is_done_iterations() or self.is_done_count() or self.is_done_epochs()

    def reset(self):
        self.epoch = 0
        self.max_epochs = None

        self.local_iteration = 0
        self.global_iteration = 0

        self.epoch_length = None
        self.max_iterations = None

        self.batch = None
        self.output = None
        self.dataloader = None
        self.metrics = {}

    def every_iterations(self, n: int) -> bool:
        return self.global_iteration > 0 and self.global_iteration % n == 0

    def every_epochs(self, n: int) -> bool:
        return self.epoch > 0 and self.epoch % n == 0

    def _update_attrs(self) -> None:
        for value in self.event_to_attr.values():
            if not hasattr(self, value):
                setattr(self, value, 0)

    def get_event_attrib_value(self, event_name: Union[str, Events, CallableEventWithFilter]) -> int:
        """Get the value of Event attribute with given `event_name`."""
        if event_name not in self.event_to_attr:
            raise RuntimeError(f"Unknown event name '{event_name}'")
        return getattr(self, self.event_to_attr[event_name])
