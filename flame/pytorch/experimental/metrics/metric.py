from typing import Any, Dict
from flame.pytorch.experimental.engine import Engine, Events

class Metric:

    def compute(self, *args, **kwargs):
        """计算metric"""
        pass

    def update(self, metrics: Dict[str, Any]):
        """写入metric"""
        pass

    def attach(self, engine: Engine, event_name: Any = Events.ITERATION_COMPLETED):
        engine.add_event_handler(event_name, self.update)