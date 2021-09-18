from .optimizer import create_optimizer_from_config, scale_lr_linearly
from .scheduler import create_scheduler_from_config
from .model import create_model_from_config
from .data import create_data_loader_from_config
from flame.next_version.config_parser import ConfigParser


def create_from_config(config: dict):
    return ConfigParser().parse(config)
