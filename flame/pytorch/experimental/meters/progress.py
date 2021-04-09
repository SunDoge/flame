from flame.pytorch.experimental.engine.events import Events
from tqdm.rich import tqdm
from flame.pytorch.experimental.engine import Engine


class ProgressBarHandler:

    def __init__(self) -> None:
        pbar = tqdm(total=1)
        pbar._prog.remove_task(pbar._task_id)
        self.progress = pbar._prog

    def attach_epoch(self, engine: Engine, desc: str = 'Epoch'):

        task_id = self.progress.add_task(desc, total=engine.state.max_epochs)
        self.progress.update(task_id, advance=engine.state.epoch)

        def update():
            self.progress.advance(task_id)

        engine.add_event_handler(Events.EPOCH_COMPLETED, update)

    def attach_iteration(self, engine: Engine, desc: str = 'Train'):

        task_id = self.progress.add_task(desc, total=engine.state.epoch_length)
        self.progress.update(task_id, advance=engine.state.local_iteration)

        def update():
            self.progress.advance(task_id)

        def reset():
            self.progress.reset(task_id)

        engine.add_event_handler(Events.EPOCH_STARTED, reset)
        engine.add_event_handler(Events.ITERATION_COMPLETED, update)
