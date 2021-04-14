from .base_checkpoint_saver import CheckpointSaver
import os


class PeriodicCheckpointSaver(CheckpointSaver):

    def __init__(
        self,
        every: int,
        fmt: str = 'checkpoint_{:04d}.pth.tar',
        entries=None
    ) -> None:
        super().__init__(entries=entries)
        self.every = every
        self.fmt = fmt

    def save(self, output_dir: str, epoch: int):
        if epoch > 0 and epoch % self.every == 0:
            name = self.fmt.format(epoch)
            filename = os.path.join(output_dir, name)
            super().save(filename)
