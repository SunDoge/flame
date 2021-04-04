"""

Process::

    call setup(data, max_epochs, max_iterations, epoch_length)
    call start_epoch_loop() as
        while not done (epoch < max_epochs):
            call step_epoch() as
                update epoch
                call start_iteration_loop() as
                    update iteration
                    while True:
                        try:
                            call step_iteration()
                        except StopIteration:
                            break

                    reset iteraion
                
            reset epoch, data_iter
            
需要思考set local_iteration, reset local_iteration的时机，现在的设置可能有问题

"""

from typing import Any, Callable, DefaultDict, Iterable, List, Optional, Tuple, Union
from .mixins import Serializable
from collections import defaultdict
from .events import Events, EventsList, State
from injector import Injector
import logging
import math

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
        else:
            container = container.create_child_injector()

        state = State()

        container.binder.bind(Engine, to=self)
        container.binder.bind(State, to=state)

        self._event_handlers: EventHandlerDict = defaultdict(list)
        self._dataloader_iter: Optional[Iterable] = None
        self._process_function = process_function
        self._init_iter = []

        self.last_event_name: Optional[Events] = None
        self.should_terminate = False
        self.should_terminate_single_epoch = False
        self.state = state
        self.container = container

    def add_event_handler(
        self,
        event_name: Union[Events, EventsList],
        handler_factory: Callable,
        *args: Any, **kwargs: Any
    ):
        handler = self.container.call_with_injection(handler_factory)

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
            # self.container.call_with_injection(
            #     func, args=new_args, kwargs=new_kwargs
            # )
            func(*new_args, **new_kwargs)

    def fire_event(self, event_name: Any):
        self._fire_event(event_name)

    def step_epoch(self):
        # 更新epoch
        self.state.epoch += 1
        self.fire_event(Events.EPOCH_STARTED)

        # 设置dataloader iter
        # if self._dataloader_iter is None:
        #     self._dataloader_iter = iter(self.state.dataloader)
        self._dataloader_iter = iter(self.state.dataloader)

        self._start_iteration_loop()

        self.fire_event(Events.EPOCH_COMPLETED)

        # 结束的时候清理掉iter
        self._dataloader_iter = None

    def step_iteration(self):
        self.state.local_iteration += 1
        self.state.global_iteration += 1
        self.fire_event(Events.ITERATION_STARTED)
        self.state.output = self._process_function(self.state.batch)
        self.fire_event(Events.ITERATION_COMPLETED)

    def internal_run(self):
        # try:
        #     self.fire_event(Events.STARTED)
        #     self._start_epoch_loop()
        #     self.fire_event(Events.COMPLETED)
        # except Exception as e:
        #     pass
        self._start_epoch_loop()

    def set_epoch(self, epoch: int):
        self.state.epoch = epoch

    def set_data(self, data: Iterable):
        self.state.dataloader = data
        self._dataloader_iter = iter(self.state.dataloader)

    def setup(
        self,
        data: Iterable,
        max_epochs: Optional[int] = None,
        max_iterations: Optional[int] = None,
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

        if self.state.max_epochs is None or self.state.is_done():
            # Create new state
            if epoch_length is None:
                # 可能获取不到
                epoch_length = self.get_data_length(data)
                if epoch_length is not None and epoch_length < 1:
                    raise ValueError(
                        "Input data has zero size. Please provide non-empty data")

            if max_iterations is None:
                if max_epochs is None:
                    _logger.warning(
                        'This dataset may be infinity, set max_epoch=1')
                    max_epochs = 1
            else:
                if max_epochs is not None:
                    raise ValueError(
                        "Arguments max_iters and max_epochs are mutually exclusive."
                        "Please provide only max_epochs or max_iters."
                    )

                if epoch_length is not None:
                    max_epochs = math.ceil(max_iterations / epoch_length)
                    _logger.info('set max_epochs=%d', max_epochs)

            self.state.local_iteration = 0
            self.state.global_iteration = 0
            self.state.max_epochs = max_epochs
            self.state.max_iterations = max_iterations
            self.state.epoch_length = epoch_length

            self.state.update_max_iterations()

            _logger.info(f"Engine run starting with max_epochs={max_epochs}.")
        else:
            _logger.info(
                f"Engine run resuming from iteration {self.state.global_iteration}, "
                f"epoch {self.state.epoch} until {self.state.max_epochs} epochs"
            )

        self.state.dataloader = data

    def _start_epoch_loop(self):
        self.fire_event(Events.STARTED)
        while not self.state.is_done():
            self.step_epoch()
        self.fire_event(Events.COMPLETED)

    def _start_iteration_loop(self):
        self.state.local_iteration = 0
        # 开始iteration loop
        while True:
            # 尝试获取数据
            try:
                self.fire_event(Events.GET_BATCH_STARTED)
                self.state.batch = next(self._dataloader_iter)
                self.fire_event(Events.GET_BATCH_COMPLETED)

            # 数据结束
            except StopIteration:
                # if self.state.epoch_length is None:
                #     self.state.epoch_length = self.state.local_iteration

                #     if self.state.max_iterations is not None:
                #         self.state.max_epochs = math.ceil(
                #             self.state.max_iterations / self.state.epoch_length
                #         )
                #     break
                break

            self.step_iteration()

            # if self.should_terminate or self.should_terminate_single_epoch:
            #     self.fire_event(Events.TERMINATE_SINGLE_EPOCH)
            #     self.should_terminate_single_epoch = False
            #     self.set_data(self.state.dataloader)
            #     break

            if self.state.epoch_length is not None and self.state.local_iteration == self.state.epoch_length:
                break

            if self.state.max_iterations is not None and self.state.global_iteration == self.state.max_iterations:
                self.should_terminate = True
                break

    def _setup_engine(self):
        """Copy fron ignite engine

        不知道有什么用
        """
        if self.state.dataloader is None:
            raise RuntimeError(
                "Internal error, self.state.dataloader is None. Please, file an issue if you encounter this error."
            )

        iteration = self.state.global_iteration
        self._dataloader_iter = iter(self.state.dataloader)

        # Below we define initial counter value for _run_once_on_dataset to measure a single epoch
        if self.state.epoch_length is not None:
            iteration %= self.state.epoch_length
        self._init_iter.append(iteration)

    @staticmethod
    def get_data_length(data: Iterable) -> Optional[int]:
        if hasattr(data, '__len__'):
            return len(data)
        else:
            _logger.warning('This dataset may be infinity')
            return None

    def run(
        self,
        data: Iterable,
        max_epochs: Optional[int] = None,
        max_iterations: Optional[int] = None,
        epoch_length: Optional[int] = None,
    ):
        self.setup(
            data,
            max_epochs=max_epochs,
            max_iterations=max_iterations,
            epoch_length=epoch_length
        )

        self.internal_run()
