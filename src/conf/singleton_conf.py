import dataclasses
from dataclasses import replace
from functools import cache
from pprint import pprint
from typing import Any, cast

import tyro

import wandb
from conf.config import (
    Config,
    EnvConfig,
    ScheduleOptimizerConfig,
    SweepConfig,
    WandbConfig,
)
from conf.config_util import DistributionConfig, dist_config_helper


def get_wandb_run_conf(wandb_conf: WandbConfig, run_id: str) -> dict:
    """Fetch the saved config dict for a prior W&B run."""
    run = wandb.Api().run(
        f"{wandb_conf.entity}/{wandb_conf.project}/{run_id}",
    )
    return run.config


@cache
def _get_config_classes() -> dict[str, type]:
    """Build a name→config-class map from all existing registries.

    Deferred until call time to avoid circular-import issues at module load.
    Concrete modules are imported here so their @register decorators fire and
    populate the registries before they are read.
    """
    import importlib

    for _mod in [
        "policy.base_schedules.constant",
        "policy.base_schedules.exponential",
        "policy.base_schedules.clipped",
        "policy.schedules.alternating",
        "policy.schedules.sigma_and_clip",
        "policy.schedules.policy_and_clip",
        "policy.schedules.dynamic_dpsgd",
        "policy.schedules.warmup_alternating",
        "policy.schedules.warmup_sigma_and_clip",
        "policy.stateful_schedules.median_gradient",
        "networks.mlp.MLP",
        "networks.cnn.CNN",
    ]:
        importlib.import_module(_mod)

    from networks._registry import _REGISTRY as _network_reg
    from networks.auto.config import AutoNetworkConfig
    from policy.base_schedules._registry import _REGISTRY as _base_reg
    from policy.schedules._registry import _REGISTRY as _sched_reg
    from policy.stateful_schedules._registry import _REGISTRY as _stateful_reg

    result = {
        config_cls.__name__: config_cls
        for registry in (_base_reg, _sched_reg, _stateful_reg, _network_reg)
        for config_cls in registry
    }
    result["AutoNetworkConfig"] = AutoNetworkConfig  # sentinel, not in registry
    return result


def _reconstruct_from_dict(obj, d: dict):
    """Recursively reconstruct a dataclass from a W&B run-config dict.

    For plain (non-Union) nested dataclasses the existing instance type is
    preserved.  For Union-typed fields the stored ``_type`` key is used to
    look up the correct config class from the registry, allowing full
    reconstruction even when the variant differs from the CLI default.

    ``DistributionConfig`` fields are treated specially: a scalar value from
    W&B is wrapped back into a constant ``DistributionConfig``.
    """
    if not dataclasses.is_dataclass(obj):
        return obj

    config_classes = _get_config_classes()
    updates: dict[str, object] = {}
    valid_fields = {f.name for f in dataclasses.fields(obj)}

    for key, item in d.items():
        if key == "_type":
            continue  # consumed by the parent call, not a dataclass field
        if key not in valid_fields:
            continue  # extra keys from other Union variants — ignore silently

        current = getattr(obj, key, None)

        if isinstance(current, DistributionConfig):
            if isinstance(item, dict):
                updates[key] = dist_config_helper(**item)
            else:
                # W&B stores the sampled scalar; wrap it back into a constant dist.
                updates[key] = dist_config_helper(value=item, distribution="constant")

        elif isinstance(item, dict) and "_type" in item:
            # Union-typed field: use the stored class name to pick the variant.
            cls_name = item["_type"]
            if cls_name not in config_classes:
                raise ValueError(
                    f"Cannot reconstruct field '{key}': unknown config class '{cls_name}'. "
                    f"Known: {sorted(config_classes)}",
                )
            target_cls = config_classes[cls_name]
            # Build a default instance then recurse with the remaining keys.
            inner = {k: v for k, v in item.items() if k != "_type"}
            updates[key] = _reconstruct_from_dict(target_cls(), inner)

        elif isinstance(item, dict) and dataclasses.is_dataclass(current):
            updates[key] = _reconstruct_from_dict(current, item)

        else:
            updates[key] = item

    return replace(cast(Any, obj), **updates)


def _get_config():
    """Parse CLI args into SingletonConfig, merging a prior W&B run's config if requested."""
    SingletonConfig.config = tyro.cli(
        Config,
        config=(tyro.conf.SuppressFixed, tyro.conf.CascadeSubcommandArgs),
    )
    wandb_conf = SingletonConfig.get_wandb_config_instance()
    if wandb_conf.restart_run_id is not None or wandb_conf.checkpoint_run_id is not None:
        run_id = wandb_conf.restart_run_id or wandb_conf.checkpoint_run_id
        assert run_id is not None  # For type checker, ensured via 'if' statement
        run_conf = get_wandb_run_conf(wandb_conf, run_id)
        SingletonConfig.config = replace(
            SingletonConfig.config,
            sweep=_reconstruct_from_dict(SingletonConfig.config.sweep, run_conf),
        )


class SingletonConfig:
    config: Config | None = None

    @classmethod
    def get_instance(cls) -> Config:
        if cls.config is None:
            _get_config()
        assert cls.config is not None
        return cls.config

    @classmethod
    def get_sweep_config_instance(cls) -> SweepConfig:
        return cls.get_instance().sweep

    @classmethod
    def get_environment_config_instance(cls) -> EnvConfig:
        return cls.get_sweep_config_instance().env

    @classmethod
    def get_schedule_optimizer_config(cls) -> ScheduleOptimizerConfig:
        return cls.get_sweep_config_instance().schedule_optimizer

    @classmethod
    def get_wandb_config_instance(cls) -> WandbConfig:
        return cls.get_instance().wandb_conf

    @classmethod
    def get_object(cls, obj):
        """Convert a dataclass instance to a plain dict."""
        return dataclasses.asdict(obj)


if __name__ == "__main__":
    pprint(SingletonConfig.get_instance())
    pprint(SingletonConfig.get_sweep_config_instance())
    pprint(SingletonConfig.get_environment_config_instance())
    pprint(SingletonConfig.get_wandb_config_instance())

    sweep_config = SingletonConfig.get_object(
        SingletonConfig.get_sweep_config_instance(),
    )
    pprint(sweep_config)
