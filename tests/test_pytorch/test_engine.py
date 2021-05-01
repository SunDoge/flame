from flame.pytorch.engine import EpochState, IterationState



def test_epoch_state():
    state = EpochState(
        max_epochs=100,
        train_state=IterationState(epoch_length=10)
    )

    state_dict = state.state_dict()

    assert state_dict['max_epochs'] == state.max_epochs
    assert state_dict['train_state']['epoch_length'] == state.train_state.epoch_length

    state_dict['train_state']['max_iterations'] = state.max_epochs * state.train_state.epoch_length

    state.load_state_dict(state_dict)

    assert state.train_state.max_iterations == state_dict['train_state']['max_iterations']