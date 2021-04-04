from flame.pytorch.experimental.engine import State, Events
from flame.pytorch.experimental.engine import Engine
from injector import inject


def _func():
    def f():
        print('func1')

    return f


@inject
def _func_with_engine(engine: Engine):
    def f(input_engine=None):
        assert engine is input_engine

    return f


@inject
def _func_with_state(state: State):
    def f(input_state=None):
        assert state is input_state

    return f


def test_engine_injection():
    engine = Engine()
    engine.add_event_handler(Events.STARTED, _func)
    engine.add_event_handler(
        Events.STARTED, _func_with_engine, input_engine=engine)
    engine.add_event_handler(
        Events.STARTED, _func_with_state, input_state=engine.state)

    engine.fire_event(Events.STARTED)


def test_engine_step_epoch():
    engine = Engine()
    engine.setup(range(10), max_epochs=2)

    assert engine.state.max_iterations == 20

    engine.step_epoch()

    assert engine.state.local_iteration == 10
    assert engine.state.global_iteration == 10

    engine.step_epoch()

    assert engine.state.local_iteration == 10
    assert engine.state.global_iteration == 20


def test_engine_run():
    engine = Engine()
    engine.run(range(10), max_epochs=2)

    assert engine.state.local_iteration == 10
    assert engine.state.global_iteration == 20
