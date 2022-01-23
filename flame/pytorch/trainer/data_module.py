from typing import Any, Callable, Optional, Tuple, TypeVar
from torch.utils.data.dataloader import DataLoader
import logging

_logger = logging.getLogger(__name__)

OptionalDataLoader = Optional[DataLoader]

T = TypeVar("T")


class DataModule:
    def __init__(
        self,
        train_loader: OptionalDataLoader = None,
        val_loader: OptionalDataLoader = None,
        test_loader: OptionalDataLoader = None,
        get_len: Callable[[Any], int] = len,
    ) -> None:
        self.train_loader = train_loader
        self.train_loader_len = self._infer_len(train_loader, get_len)
        self.val_loader = val_loader
        self.val_loader_len = self._infer_len(val_loader, get_len)
        self.test_loader = test_loader
        self.test_loader_len = self._infer_len(test_loader, get_len)
        self.get_len = get_len

    @staticmethod
    def _infer_len(loader, get_len) -> Optional[int]:
        if loader:
            return get_len(loader)
