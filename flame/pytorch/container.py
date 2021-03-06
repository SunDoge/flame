"""
方法全部以build为前缀，和detection一致
"""

import logging

import flame
import torch
from flame.argument import BasicArgs
from injector import Module, provider, singleton

from .typing_prelude import Device, ExperimentDir, LocalRank, RootConfig, TestDataset, TestSampler, TrainDataset, TrainSampler, ValDataset, ValSampler
from .utils.distributed import get_rank_safe
from torch.utils.data.distributed import DistributedSampler

_logger = logging.getLogger(__name__)


class RootModule(Module):

    def __init__(self, local_rank: int) -> None:
        super().__init__()
        self.local_rank = local_rank

    @singleton
    @provider
    def create_args(self) -> BasicArgs:
        args, _ = BasicArgs.from_known_args()
        return args

    @singleton
    @provider
    def create_config(self, args: BasicArgs, experiment_dir: ExperimentDir) -> RootConfig:
        cfg, diff = flame.config.parse_config(args.local, args.config)
        _logger.info('Diff: \n%s', diff)

        _logger.debug('Config: \n%s', cfg)

        flame.config.dump_as_json(cfg, experiment_dir / 'config.json')

        return cfg

    @singleton
    @provider
    def create_experiment_dir(self, args: BasicArgs) -> ExperimentDir:
        experiment_dir = flame.utils.experiment.get_experiment_dir(
            args.output_dir, args.experiment_name, debug=args.debug
        )

        rank = get_rank_safe()
        if rank == 0:
            flame.utils.experiment.make_experiment_dir(
                experiment_dir, 
                yes=args.yes
            )
            flame.archiver.make_archive(
                experiment_dir / 'code.zip'
            )
            flame.argument.save_command(
                experiment_dir / 'run.sh'
            )
        flame.logging.init_logger(
            rank=rank,
            filename=experiment_dir / 'experiment.log',
            force=True,
            debug=args.debug
        )
        

        return experiment_dir

    @singleton
    @provider
    def create_device(self) -> Device:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @singleton
    @provider
    def create_local_rank(self) -> LocalRank:
        return self.local_rank

    
