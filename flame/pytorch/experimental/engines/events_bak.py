from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Optional, Union, Iterator, TYPE_CHECKING
from dataclasses import dataclass, field
import weakref
from types import DynamicClassAttribute
import numbers

if TYPE_CHECKING:
    from .base_engine import BaseEngine


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

    def __init__(
        self,
        value: str,
        event_filter: Optional[Callable] = None,
        name: Optional[str] = None
    ) -> None:
        if event_filter is None:
            event_filter = CallableEventWithFilter.default_event_filter

        self.filter = event_filter

        """
        Compatible with Enum
        """
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
        self,
        event_filter: Optional[Callable] = None,
        every: Optional[int] = None,
        once: Optional[int] = None,
    ) -> 'CallableEventWithFilter':
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

        # FIXME: 不太明白这里为什么要这样实现
        if every is not None:
            if every == 1:
                # Just return the event itself
                event_filter = None
            else:
                event_filter = self.every_event_filter(every)

        if once is not None:
            event_filter = self.once_event_filter(once)

        return CallableEventWithFilter(self.value, event_filter, self.name)

    @staticmethod
    def default_event_filter(engine: 'BaseEngine', event: int) -> bool:
        return True

    @staticmethod
    def every_event_filter(every: int) -> Callable:
        """A wrapper for every event filter."""

        def wrapper(engine: "BaseEngine", event: int) -> bool:
            if event % every == 0:
                return True
            return False

        return wrapper

    @staticmethod
    def once_event_filter(once: int) -> Callable:
        """A wrapper for once event filter."""

        def wrapper(engine: "BaseEngine", event: int) -> bool:
            if event == once:
                return True
            return False

        return wrapper

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


class EventEnum(CallableEventWithFilter, Enum):

    pass


class Events(EventEnum):

    """Events that are fired by the :class:`~ignite.engine.engine.Engine` during execution. Built-in events:

    - STARTED : triggered when engine's run is started
    - EPOCH_STARTED : triggered when the epoch is started
    - GET_BATCH_STARTED : triggered before next batch is fetched
    - GET_BATCH_COMPLETED : triggered after the batch is fetched
    - ITERATION_STARTED : triggered when an iteration is started
    - ITERATION_COMPLETED : triggered when the iteration is ended

    - DATALOADER_STOP_ITERATION : engine's specific event triggered when dataloader has no more data to provide

    - EXCEPTION_RAISED : triggered when an exception is encountered
    - TERMINATE_SINGLE_EPOCH : triggered when the run is about to end the current epoch,
      after receiving a :meth:`~ignite.engine.engine.Engine.terminate_epoch()` or
      :meth:`~ignite.engine.engine.Engine.terminate()` call.

    - TERMINATE : triggered when the run is about to end completely,
      after receiving :meth:`~ignite.engine.engine.Engine.terminate()` call.

    - EPOCH_COMPLETED : triggered when the epoch is ended. Note that this is triggered even
      when :meth:`~ignite.engine.engine.Engine.terminate_epoch()` is called.
    - COMPLETED : triggered when engine's run is completed

    The table below illustrates which events are triggered when various termination methods are called.

    .. list-table::
       :widths: 24 25 33 18
       :header-rows: 1

       * - Method
         - EVENT_COMPLETED
         - TERMINATE_SINGLE_EPOCH
         - TERMINATE
       * - no termination
         - ✔
         - ✗
         - ✗
       * - :meth:`~ignite.engine.engine.Engine.terminate_epoch()`
         - ✔
         - ✔
         - ✗
       * - :meth:`~ignite.engine.engine.Engine.terminate()`
         - ✗
         - ✔
         - ✔

    Since v0.3.0, Events become more flexible and allow to pass an event filter to the Engine:

    .. code-block:: python

        engine = Engine()

        # a) custom event filter
        def custom_event_filter(engine, event):
            if event in [1, 2, 5, 10, 50, 100]:
                return True
            return False

        @engine.on(Events.ITERATION_STARTED(event_filter=custom_event_filter))
        def call_on_special_event(engine):
            # do something on 1, 2, 5, 10, 50, 100 iterations

        # b) "every" event filter
        @engine.on(Events.ITERATION_STARTED(every=10))
        def call_every(engine):
            # do something every 10th iteration

        # c) "once" event filter
        @engine.on(Events.ITERATION_STARTED(once=50))
        def call_once(engine):
            # do something on 50th iteration

    Event filter function `event_filter` accepts as input `engine` and `event` and should return True/False.
    Argument `event` is the value of iteration or epoch, depending on which type of Events the function is passed.

    Since v0.4.0, user can also combine events with `|`-operator:

    .. code-block:: python

        events = Events.STARTED | Events.COMPLETED | Events.ITERATION_STARTED(every=3)
        engine = ...

        @engine.on(events)
        def call_on_events(engine):
            # do something

    Since v0.4.0, custom events defined by user should inherit from :class:`~ignite.engine.events.EventEnum` :

    .. code-block:: python

        class CustomEvents(EventEnum):
            FOO_EVENT = "foo_event"
            BAR_EVENT = "bar_event"
    """

    EPOCH_STARTED = "epoch_started"
    """triggered when the epoch is started."""
    EPOCH_COMPLETED = "epoch_completed"
    """Event attribute indicating epoch is ended."""

    STARTED = "started"
    """triggered when engine’s run is started."""
    COMPLETED = "completed"
    """"triggered when engine’s run is completed"""

    ITERATION_STARTED = "iteration_started"
    """triggered when an iteration is started."""
    ITERATION_COMPLETED = "iteration_completed"
    """triggered when the iteration is ended."""
    EXCEPTION_RAISED = "exception_raised"
    """triggered when an exception is encountered."""

    GET_BATCH_STARTED = "get_batch_started"
    """triggered before next batch is fetched."""
    GET_BATCH_COMPLETED = "get_batch_completed"
    """triggered after the batch is fetched."""

    DATALOADER_STOP_ITERATION = "dataloader_stop_iteration"
    """"engine’s specific event triggered when dataloader has no more data to provide"""
    TERMINATE = "terminate"
    """triggered when the run is about to end completely, after receiving terminate() call."""
    TERMINATE_SINGLE_EPOCH = "terminate_single_epoch"
    """triggered when the run is about to end the current epoch,
    after receiving a terminate_epoch() or terminate() call."""

    def __or__(self, other: Any) -> "EventsList":
        return EventsList() | self | other


class EventsList:
    """Collection of events stacked by operator `__or__`.

    .. code-block:: python

        events = Events.STARTED | Events.COMPLETED
        events |= Events.ITERATION_STARTED(every=3)

        engine = ...

        @engine.on(events)
        def call_on_events(engine):
            # do something

    or

    .. code-block:: python

        @engine.on(Events.STARTED | Events.COMPLETED | Events.ITERATION_STARTED(every=3))
        def call_on_events(engine):
            # do something

    """

    def __init__(self) -> None:
        self._events = []  # type: List[Union[Events, CallableEventWithFilter]]

    def _append(self, event: Union[Events, CallableEventWithFilter]) -> None:
        if not isinstance(event, (Events, CallableEventWithFilter)):
            raise TypeError(
                f"Argument event should be Events or CallableEventWithFilter, got: {type(event)}")
        self._events.append(event)

    def __getitem__(self, item: int) -> Union[Events, CallableEventWithFilter]:
        return self._events[item]

    def __iter__(self) -> Iterator[Union[Events, CallableEventWithFilter]]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def __or__(self, other: Union[Events, CallableEventWithFilter]) -> "EventsList":
        self._append(event=other)
        return self


@dataclass
class EpochState:
    epoch: int = 0
    max_epochs: Optional[int] = None


@dataclass
class IterationState(EpochState):
    """
    local_iteration指当前循环的iter，global_iteration指总的循环次数
    """
    local_iteration: int = 0
    global_iteration: int = 0
    epoch_length: Optional[int] = None
    max_iters: Optional[int] = None  # 还没懂有什么用
    output: Optional[Any] = None  # 每个iter模型输出的内容
    batch: Optional[Any] = None  # 每个iter模型的输入
    metrics: Dict[str, Any] = field(default=dict())
    dataloader: Optional[Iterable[Any]] = None


class RemovableEventHandle:

    def __init__(
        self,
        event_name: Union[Events, EventsList],
        handler: Callable,
        engine: 'BaseEngine',
    ) -> None:
        self.event_name = event_name
        self.handler = weakref.ref(handler)
        self.engine = weakref.ref(engine)

    def remove(self):
        handler = self.handler()
        engine = self.engine()

        if handler is None or engine is None:
            return

        if isinstance(self.event_name, EventsList):
            for e in self.event_name:
                if engine:
                    pass

    def __enter__(self) -> 'RemovableEventHandle':
        """没搞懂用途"""
        return self

    def __exit__(self, *args, **kwargs):
        """没搞懂用途"""
        self.remove()



