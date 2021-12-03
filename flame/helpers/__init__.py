from .optimizer import create_optimizer_from_config, scale_lr_linearly
from .scheduler import create_scheduler_from_config
from .model import create_model_from_config
from .data import create_data_loader_from_config, create_data_loader
from flame.config_parser import ConfigParser
from . import checkpoint_saver
from .tensorboard import Rank0SummaryWriter


def create_from_config(config: dict):
    return ConfigParser().parse(config)
