import logging
from ignite.engine import Engine as BaseEngine, RemovableEventHandle, CallableEventWithFilter, EventsList
from ignite.engine.utils import _check_signature
from injector import Binder, Injector, inject

from typing import Any, Callable, Optional


class Engine(BaseEngine):

    def __init__(self, process_function: Callable, injector: Optional[Injector] = None):
        super().__init__(process_function)

        # self.logger = logging.getLogger(__name__)

        if injector is None:

            # def configure_self(binder: Binder):
            # binder.bind(self.__class__, to=self)

            self.logger.warning('Init an empty Dependency Injector')
            injector = Injector()

        # injector.binder.bind(self.__class__, to=self)

        self.injector = injector

    @staticmethod
    def with_engine(process_function_without_engine: Callable) -> Callable:
        """Helper function"""
        def process_function(_engine: 'Engine', batch: Any):
            return process_function_without_engine(batch)
        return process_function

    def add_event_handler(self, event_name: Any, handler: Callable, *args: Any, **kwargs: Any) -> RemovableEventHandle:
        """Add an event handler to be executed when the specified event is fired.

        Args:
            event_name: An event or a list of events to attach the handler. Valid events are
                from :class:`~ignite.engine.events.Events` or any ``event_name`` added by
                :meth:`~ignite.engine.engine.Engine.register_events`.
            handler: the callable event handler that should be invoked. No restrictions on its signature.
                The first argument can be optionally `engine`, the :class:`~ignite.engine.engine.Engine` object,
                handler is bound to.
            args: optional args to be passed to ``handler``.
            kwargs: optional keyword args to be passed to ``handler``.

        Returns:
            :class:`~ignite.engine.events.RemovableEventHandle`, which can be used to remove the handler.

        Note:
            Note that other arguments can be passed to the handler in addition to the `*args` and  `**kwargs`
            passed here, for example during :attr:`~ignite.engine.events.Events.EXCEPTION_RAISED`.

        Example usage:

        .. code-block:: python

            engine = Engine(process_function)

            def print_epoch(engine):
                print(f"Epoch: {engine.state.epoch}")

            engine.add_event_handler(Events.EPOCH_COMPLETED, print_epoch)

            events_list = Events.EPOCH_COMPLETED | Events.COMPLETED

            def execute_something():
                # do some thing not related to engine
                pass

            engine.add_event_handler(events_list, execute_something)

        Note:
            Since v0.3.0, Events become more flexible and allow to pass an event filter to the Engine.
            See :class:`~ignite.engine.events.Events` for more details.

        """
        if isinstance(event_name, EventsList):
            for e in event_name:
                self.add_event_handler(e, handler, *args, **kwargs)
            return RemovableEventHandle(event_name, handler, self)
        if (
            isinstance(event_name, CallableEventWithFilter)
            and event_name.filter != CallableEventWithFilter.default_event_filter
        ):
            event_filter = event_name.filter
            handler = self._handler_wrapper(handler, event_name, event_filter)

        self._assert_allowed_event(event_name)

        # event_args = (
        #     Exception(),) if event_name == Events.EXCEPTION_RAISED else ()
        # try:
        #     _check_signature(handler, "handler", self, *(event_args + args), **kwargs)
        #     self._event_handlers[event_name].append((handler, (self,) + args, kwargs))
        # except ValueError:
        #     _check_signature(handler, "handler", *(event_args + args), **kwargs)
        #     self._event_handlers[event_name].append((handler, args, kwargs))

        """
        Always add self
        """
        # self._event_handlers[event_name].append((handler, args, kwargs))
        self._event_handlers[event_name].append(
            (handler, (self,) + args, kwargs))

        self.logger.debug(f"added handler for event {event_name}")

        return RemovableEventHandle(event_name, handler, self)

    def _fire_event(self, event_name: Any, *event_args: Any, **event_kwargs: Any) -> None:
        """Execute all the handlers associated with given event.

        This method executes all handlers associated with the event
        `event_name`. Optional positional and keyword arguments can be used to
        pass arguments to **all** handlers added with this event. These
        arguments updates arguments passed using :meth:`~ignite.engine.engine.Engine.add_event_handler`.

        Args:
            event_name: event for which the handlers should be executed. Valid
                events are from :class:`~ignite.engine.events.Events` or any `event_name` added by
                :meth:`~ignite.engine.engine.Engine.register_events`.
            *event_args: optional args to be passed to all handlers.
            **event_kwargs: optional keyword args to be passed to all handlers.

        """
        self.logger.debug(f"firing handlers for event {event_name}")
        self.last_event_name = event_name
        for func, args, kwargs in self._event_handlers[event_name]:
            kwargs.update(event_kwargs)
            # first, others = ((args[0],), args[1:]) if (
            #     args and args[0] == self) else ((), args)
            # func(*first, *(event_args + others), **kwargs)
            self.injector.call_with_injection(func, args=args, kwargs=kwargs)

    @property
    def iter_counter(self):
        """Get current iter number, 1 base"""
        return (self.state.iteration - 1) % self.state.epoch_length + 1


if __name__ == '__main__':
    from ignite.engine import Events
    from injector import singleton

    @singleton
    class A:
        @inject
        def __init__(self) -> None:
            print(self.__class__)

    injector = Injector()
    engine = Engine(Engine.with_engine(lambda x: print(x)), injector=None)
    print(engine)

    @engine.on(Events.ITERATION_STARTED)
    @inject
    def print_something(engine: Engine, a: A):
        print('something')
        print('engine', engine)

    engine.run(range(10))
