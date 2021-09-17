from flame.next_version.config_parser import ConfigParser


def create_scheduler_from_config(config: dict, optimizer):
    config_parser = ConfigParser()
    config['optimizer'] = optimizer
    scheduler = config_parser.parse(config)
    return scheduler
