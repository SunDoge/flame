from flame.pytorch.helpers.cudnn import cudnn_benchmark_if_possible
from .arguments import BaseArgs
from typing import Callable, Optional
from flame.core.config_parser import ConfigParser
from flame.core.logging import init_logging, log_runtime_env, remove_all_handlers
import torch.multiprocessing as mp
import logging

_logger = logging.getLogger(__name__)

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


def _init_distributed(
    local_rank: int,
    args: BaseArgs,
    config: dict,
    main_worker: MainWorker,
):

    if args.dist_url:
        rank = args.init_process_group_from_tcp(local_rank)
    else:
        # Use share file by default
        rank = args.init_process_group_from_file(local_rank)

    if rank == 0:
        init_logging(
            filename=args.experiment_dir / 'experiment.log',
            debug=args.debug,
        )
    else:
        # 不确定是否应该这样做
        remove_all_handlers()

    log_runtime_env()

    args.try_cuda_set_device(local_rank)

    if not args.debug:
        cudnn_benchmark_if_possible()

    main_worker(args, config)


def run_distributed(
    args: BaseArgs,
    config: Optional[dict] = None,
    main_worker: MainWorker = default_main_worker,
):
    init_logging(debug=args.debug)

    if config is None:
        config = args.config()

    if args.rank_start == 0:
        args.try_make_experiment_dir()
        args.save_config(config=config)
        args.save_command()

    num_procs = 1
    if len(args.gpu) > 1:
        num_procs = len(args.gpu)

    if len(args.gpu) > args.world_size:
        world_size = len(args.gpu)
        _logger.info('set args.world_size=%d', world_size)
        args.world_size = world_size

    proc_args = (args, config, main_worker)

    if num_procs == 1:
        _init_distributed(
            0, *proc_args
        )
    else:
        mp.spawn(
            _init_distributed,
            args=proc_args,
            nprocs=num_procs,
        )
