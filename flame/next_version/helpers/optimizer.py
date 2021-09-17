from flame.next_version.config_parser import ConfigParser


def create_optimizer_from_config(config: dict, params):
    config_parser = ConfigParser()
    config['params'] = params
    optimizer = config_parser.parse(config)
    return optimizer
