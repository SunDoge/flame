from .base_engine import BaseEngine
from typing import Optional
from injector import Injector


class EpochEngine(BaseEngine):

    def __init__(self, ,container: Optional[Injector]):
        super().__init__(container=container)

    def step(self):
        pass