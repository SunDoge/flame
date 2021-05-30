from typing import Optional
from pydantic import BaseModel


class Stage(BaseModel):
    batch_size: int
    num_workers: int
    dataset: dict


    



