from flame.core.arguments import BaseArgs as Base
from dataclasses import dataclass
import typed_args as ta
import oneflow as flow
import os


DEFAULT_DEVICE_ID = int(os.environ.get("DEVICE_ID", "-1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))


@dataclass
class BaseArgs(Base):

    device_id: int = ta.add_argument(
        '--device-id', type=int,
        default=DEFAULT_DEVICE_ID,  # -1 for cpu
    )

    @property
    def device(self) -> flow.device:
        # if self.device_id < 0:
        #     return flow.device('cpu')
        # else:
        #     return flow.device('cuda:{}'.format(self.device_id))
        return flow.device('cuda:{}'.format(LOCAL_RANK))
