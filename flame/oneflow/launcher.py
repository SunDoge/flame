from oneflow.distributed import launch
from .arguments import BaseArgs
from flame.core.config_parser import ConfigParser
from typing import Callable
from flame.core.logging import init_logging, remove_all_handlers, log_runtime_env
import oneflow.env

MainWorker = Callable[[BaseArgs, dict], None]


def default_main_worker(
    args: BaseArgs, config: dict
):
    ConfigParser(
        args=args,
        config=config
    ).parse_root_config(
        config
    )


def run_distributed(
    args: BaseArgs,
    main_worker: MainWorker = default_main_worker
):
    rank = oneflow.env.get_rank()
    config = args.config()

    if rank == 0:
        args.try_make_experiment_dir()
        args.save_config(config=config)
        args.save_command()
        init_logging(
            filename=args.experiment_dir / 'experiment.log',
            debug=args.debug,
        )
    else:
        # 不确定是否应该这样做
        remove_all_handlers()

    log_runtime_env()

    main_worker(args, config)
