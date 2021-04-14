from flame.pytorch.experimental.engine.events import Events
from tqdm.rich import tqdm
from flame.pytorch.experimental.engine import Engine
import logging

_logger = logging.getLogger(__name__)


class ProgressBarHandler:

    def __init__(self) -> None:
        pbar = tqdm(total=1)
        pbar._prog.remove_task(pbar._task_id)
        self._pbar = pbar
        self.progress = pbar._prog

    def attach_epoch(self, engine: Engine, desc: str = 'Epoch'):

        task_id = self.progress.add_task(desc, total=engine.state.max_epochs)
        self.progress.update(task_id, advance=engine.state.epoch)

        def update():
            self.progress.advance(task_id)

        engine.add_event_handler(Events.EPOCH_COMPLETED, update)

    def attach_iteration(self, engine: Engine, desc: str = 'Train'):

        task_id = self.progress.add_task(desc, total=engine.state.epoch_length, visible=False)
        self.progress.update(task_id, advance=engine.state.local_iteration)

        def update():
            # _logger.info('update %s', task_id)
            self.progress.advance(task_id)

        def reset():
            self.progress.update(task_id, visible=True)

        def complete():
            self.progress.reset(task_id, visible=False)


        engine.add_event_handler(Events.EPOCH_STARTED, reset)
        engine.add_event_handler(Events.ITERATION_COMPLETED, update)
        engine.add_event_handler(Events.EPOCH_COMPLETED, complete)


    # def __del__(self):
    #     _logger.error('deleted')