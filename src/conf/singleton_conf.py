import tyro
from conf.config import (
    Config,
    EnvConfig,
    SweepConfig,
    WandbConfig,
    PolicyConfig,
    DistributionConfig,
    dist_config_helper,
)
from pprint import pprint
from dataclasses import asdict, Field, replace
import wandb


def get_wandb_run_conf(wandb_conf: WandbConfig) -> dict:
    run = wandb.Api().run(
        f"{wandb_conf.entity}/{wandb_conf.project}/{wandb_conf.restart_run_id}"
    )

    return run.config


def _populate_conf_from_dict(conf: Config, dictionary: dict) -> Config:
    for key, item in dictionary.items():
        conf_type = type(getattr(conf, key))
        if isinstance(conf_type, key), DistributionConfig):
            conf = replace(
                conf, **{key: dist_config_helper(value=item, distribution="constant")}
            )
        elif isinstance(item, dict):
            # nested configuration class
            conf = replace(
                conf, **{key: _populate_conf_from_dict(getattr(conf, key), item)}
            )
        else:
            conf = replace(conf, **{key: item})

    return conf


def _get_config():
    SingletonConfig.config = tyro.cli(Config)
    wandb_conf = SingletonConfig.get_wandb_config_instance()
    if wandb_conf.restart_run_id is not None:
        conf = get_wandb_run_conf(wandb_conf)
        SingletonConfig.config = replace(
            SingletonConfig.config,
            # replace the experiment portion, wandb remains the same
            sweep=_populate_conf_from_dict(SingletonConfig.config.sweep, conf),
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
    def get_policy_config_instance(cls) -> PolicyConfig:
        return cls.get_sweep_config_instance().policy

    @classmethod
    def get_wandb_config_instance(cls) -> WandbConfig:
        return cls.get_instance().wandb_conf

    @classmethod
    def get_object(cls, obj):
        return asdict(obj)


if __name__ == "__main__":
    # This is just for testing purposes
    pprint(SingletonConfig.get_instance())
    pprint(SingletonConfig.get_experiment_config_instance())
    pprint(SingletonConfig.get_sweep_config_instance())
    pprint(SingletonConfig.get_environment_config_instance())
    pprint(SingletonConfig.get_wandb_config_instance())

    sweep_config = SingletonConfig.get_object(
        SingletonConfig.get_sweep_config_instance()
    )
    pprint(sweep_config)
