"""
方法全部以build为前缀，和detection一致
"""

import logging

from pydantic.main import BaseModel

import flame
import torch
from flame.argument import BasicArgs
from injector import ClassAssistedBuilder, Injector, Module, provider, singleton, AssistedBuilder, CallableProvider
from typing import Any, TypeVar

from .typing_prelude import Device, ExperimentDir, LocalRank, DictConfig, DebugMode
from .utils.distributed import get_rank_safe
from torch.utils.data.distributed import DistributedSampler
import inspect

_logger = logging.getLogger(__name__)

T = TypeVar('T')


class CallableAssistedBuilder(AssistedBuilder[T]):

    def build(self, **kwargs: Any) -> T:
        binder = self._injector.binder
        binding, _ = binder.get_binding(self._target)
        provider = binding.provider
        if not isinstance(provider, CallableProvider):
            raise Exception(
                'Assisted interface building works only with ClassProviders, '
                'got %r for %r' % (provider, binding.interface)
            )
        return self._injector.call_with_injection(
            provider._callable, kwargs=kwargs
        )


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
    def create_config(self, args: BasicArgs, experiment_dir: ExperimentDir) -> DictConfig:
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

    # ===============================================

    @singleton
    @provider
    def create_debug_mode(self, args: BasicArgs) -> DebugMode:
        return args.debug


class BaseModule(Module):

    def __init__(self, args: BasicArgs) -> None:
        super().__init__()
        self.args = args

    @singleton
    @provider
    def create_args(self) -> BasicArgs:
        # args, _ = BasicArgs.from_known_args()
        # return args
        return self.args

    @singleton
    @provider
    def create_config(self, args: BasicArgs, experiment_dir: ExperimentDir) -> DictConfig:
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
    def create_local_rank(self, args: BasicArgs) -> LocalRank:
        # return self.local_rank
        return args.local_rank

    # ===============================================

    @singleton
    @provider
    def create_debug_mode(self, args: BasicArgs) -> DebugMode:
        return args.debug


def build_from_config_with_container(container: Injector, cfg: dict, type_key='_type'):
    type_path: str = cfg[type_key]
    type_factory = flame.auto_builder.import_from_path(
        type_path
    )
    if inspect.isclass(type_factory):
        assisted_builder = ClassAssistedBuilder[type_factory]
    elif inspect.isfunction(type_factory):
        assisted_builder = CallableAssistedBuilder[type_factory]
    else:
        raise Exception()

    builder: AssistedBuilder = container.get(assisted_builder)
    kwargs = flame.auto_builder.get_kwargs_from_config(cfg, type_key=type_key)
    obj = builder.build(**kwargs)
    return obj
