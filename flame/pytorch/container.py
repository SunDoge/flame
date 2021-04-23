"""
方法全部以build为前缀，和detection一致
"""

from .utils.distributed import get_rank_safe
from injector import Module, provider, singleton
from flame.argument import BasicArgs
from .typing_prelude import Device, ExperimentDir, RootConfig
import flame
import logging
import torch

_logger = logging.getLogger(__name__)


class BaseModule(Module):

    @singleton
    @provider
    def build_args(self) -> BasicArgs:
        args, _ = BasicArgs.from_known_args()
        return args

    @singleton
    @provider
    def build_cfg(self, args: BasicArgs, experiment_dir: ExperimentDir) -> RootConfig:
        cfg, diff = flame.config.parse_config(args.local, args.config)
        _logger.info('Diff: \n%s', diff)

        _logger.debug('Config: \n%s', cfg)

        flame.config.dump_as_json(cfg, experiment_dir / 'config.json')

        return cfg

    @singleton
    @provider
    def build_experiment_dir(self, args: BasicArgs) -> ExperimentDir:
        experiment_dir = flame.utils.experiment.get_experiment_dir(
            args.output_dir, args.experiment_name, debug=args.debug
        )

        rank = get_rank_safe()
        if rank == 0:
            flame.utils.experiment.make_experiment_dir(
                experiment_dir, yes=args.yes
            )
        flame.logging.init_logger(
            rank=rank, filename=experiment_dir / 'experiment.log', force=True
        )
        flame.argument.save_command(
            experiment_dir / 'run.sh'
        )

        return experiment_dir

    @singleton
    @provider
    def build_device(self) -> Device:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
