from .utils.distributed import get_rank_safe
from injector import Module, provider, singleton
from flame.argument import BasicArgs, add_basic_arguments
from .typing_prelude import ExperimentDir, RootConfig
import flame


class BaseModule(Module):

    @singleton
    @provider
    def configure_args(self) -> BasicArgs:
        args = add_basic_arguments().parse_known_args()
        return args

    @singleton
    @provider
    def configure_cfg(self, args: BasicArgs) -> RootConfig:
        cfg = flame.config.parse_config(args.local, args.config)
        return cfg

    @singleton
    @provider
    def configure_experiment_dir(self, args: BasicArgs) -> ExperimentDir:
        experiment_dir = flame.utils.experiment.get_experiment_dir(
            args.output_dir, args.experiment_name, debug=args.debug
        )

        rank = get_rank_safe()
        if rank == 0:
            flame.utils.experiment.make_experiment_dir(
                experiment_dir, yes=args.yes
            )
        flame.logging.init_logger(
            rank=rank, filename=experiment_dir / 'experiment.log'
        )

        return experiment_dir
