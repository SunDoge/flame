from flame.pytorch.helpers.cudnn import cudnn_benchmark_if_possible
from .arguments import BaseArgs
from typing import Callable
from flame.core.config_parser import ConfigParser
from flame.core.logging import init_logging
import torch.multiprocessing as mp

MainWorker = Callable[[BaseArgs], None]


def _default_main_worker(
    args: BaseArgs
):
    assert args.config_file, "please provide config file by -c/--config-file"
    config = args.config

    ConfigParser(
        args=args,
        config=config
    ).parse_root_config(
        config
    )


def _init_distributed(
    local_rank: int,
    args: BaseArgs,
    main_worker: MainWorker,
):
    rank = args.init_process_group_from_file(local_rank)
    if rank == 0:
        init_logging(
            filename=args.experiment_dir / 'experiment.log',
            debug=args.debug,
        )

    args.try_cuda_set_device(local_rank)

    if not args.debug:
        cudnn_benchmark_if_possible()

    main_worker(args)


def run_distributed(
    args: BaseArgs,
    main_worker: MainWorker = _default_main_worker,
):
    init_logging(debug=args.debug)

    if args.rank_start == 0:
        args.try_make_experiment_dir()
        args.save_config()

    num_procs = 1
    if len(args.gpu) > 1:
        num_procs = len(args.gpu)

    proc_args = (args, main_worker)

    if num_procs == 1:
        _init_distributed(
            0, *proc_args
        )
    else:
        mp.spawn(
            _init_distributed,
            args=proc_args
        )
