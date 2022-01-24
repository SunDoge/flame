import cv2
import numpy as np
import logging

_logger = logging.getLogger(__name__)


def cv2_loader(filename: str) -> np.ndarray:
    img = cv2.imread(filename)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def cv2_worker_init_fn(worker_id: int):
    cv2.setNumThreads(0)
    _logger.info("cv2 num threads: %s", cv2.getNumThreads())
