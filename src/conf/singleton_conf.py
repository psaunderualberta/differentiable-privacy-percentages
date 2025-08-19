import tyro
from conf.config import Config, EnvConfig, StreamACConfig, StreamQConfig, StreamSarsaConfig, SweepConfig, ExperimentConfig, WandbConfig
from pprint import pprint
from dataclasses import asdict

def _get_config():
    SingletonConfig.config = tyro.cli(Config)


class SingletonConfig:
    config: Config | None = None

    @classmethod
    def get_instance(cls) -> Config:
        if cls.config is None:
            _get_config()
        assert cls.config is not None
        return cls.config

    @classmethod
    def get_experiment_config_instance(cls) -> ExperimentConfig:
        return cls.get_instance().experiment

    @classmethod
    def get_sweep_config_instance(cls) -> SweepConfig:
        return cls.get_experiment_config_instance().sweep
    @classmethod
    def get_environment_config_instance(cls) -> EnvConfig:
        return cls.get_sweep_config_instance().env

    @classmethod
    def get_algorithm_config_instance(cls) -> StreamSarsaConfig | StreamQConfig | StreamACConfig:
        sweep_instance = cls.get_sweep_config_instance()
        return getattr(sweep_instance, sweep_instance.algorithm)

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
    pprint(SingletonConfig.get_algorithm_config_instance())
    pprint(SingletonConfig.get_wandb_config_instance())

    sweep_config = SingletonConfig.get_object(SingletonConfig.get_sweep_config_instance())
    pprint(sweep_config)