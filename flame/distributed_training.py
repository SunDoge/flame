from flame.config_parser import ConfigParser, require
import logging
from typing import Callable

from flame.arguments import BaseArgs
from flame.utils.experiment import get_experiment_dir, make_experiment_dir
from flame.logging import init_logger
import torch.multiprocessing as mp
import torch.distributed as dist
import torch
from flame.config import from_file
from injector import Injector, inject


_logger = logging.getLogger(__name__)


MainWorker = Callable[[BaseArgs], None]
KEY_ENGINE = 'engine'
KEY_MAIN_WORKER = 'main_worker'


def _default_main_worker(
    args: BaseArgs,
):
    assert args.config, "please provide config file"
    # config = from_file(args.config)
    config = args.parse_config()
    # ic(config)
    # engine_name = config[KEY_ENGINE]
    # engine_class: BaseEngine = require(engine_name)
    # container = Injector([engine_class.ProviderModule(args, config)])
    # engine = container.get(engine_class)
    # container.call_with_injection(inject(engine.run))

    # main_worker_name = config[KEY_MAIN_WORKER]
    # main_worker = require(main_worker_name)
    # config['args'] = args
    # main_worker(**config)
    ConfigParser(args=args).parse(config, depth=1)


def _init_process_group(
    local_rank: int,  # provide by mp.spawn
    args: BaseArgs,
    main_worker: MainWorker,
):
    if len(args.gpu) > 0:
        args.device_id = args.gpu[local_rank]
        _logger.info('set device_id: %d', args.device_id)
        torch.cuda.set_device(args.device_id)
    
    if len(args.gpu) > args.world_size:
        world_size = len(args.gpu)
        _logger.info('set args.world_size=%d', world_size)
        args.world_size = world_size

    args.rank = args.rank_start + local_rank
    print(f'start rank {args.rank}')
    init_logger(
        rank=args.rank,
        filename=args.experiment_dir / 'experiment.log',
        debug=args.debug,
        force=True
    )

    dist.init_process_group(
        args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    main_worker(args)

    # dist.destroy_process_group()


def start_distributed_training(
    args: BaseArgs,
    main_worker: Callable[[BaseArgs], None] = _default_main_worker
):
    init_logger(debug=args.debug)
    # step 1, check output dir, experiment dir
    if args.rank_start == 0:

        # ask if user want to remove exists exp dir
        # if not, quit
        if args.experiment_dir.exists():

            print('Move exists experiment dir to debug folder? [Y/n]')
            if args.debug:
                response = 'yes'
            else:
                response = input().strip().lower()
            if response in {'y', 'yes', ''}:
                new_experiment_dir = get_experiment_dir(
                    args.output_dir,
                    args.experiment_name,
                    debug=True,
                    with_timestamp=True
                )
                new_experiment_dir.parent.mkdir(parents=True, exist_ok=True)
                args.experiment_dir.rename(new_experiment_dir)
            else:
                print('quit')
                exit(0)

        # make experiment dir
        args.experiment_dir.mkdir(parents=True)

    # step 2 init logger with experiment dir
    num_procs = 1
    if len(args.gpu) > 1:
        num_procs = len(args.gpu)

    proc_args = (args, main_worker)
    if num_procs == 1:
        _init_process_group(
            0,
            *proc_args
        )
    else:
        mp.spawn(
            _init_process_group,
            args=proc_args,
            nprocs=num_procs
        )
