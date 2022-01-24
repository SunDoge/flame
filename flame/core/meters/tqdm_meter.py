from tqdm import tqdm


class TqdmMeter:

    def __init__(self, pbar: tqdm) -> None:
        self._pbar = pbar
        self._batch_size = 0

    def update(self, batch_size: int):
        self._batch_size = batch_size
