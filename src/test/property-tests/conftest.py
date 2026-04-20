import importlib

import pytest

for _mod in [
    "policy.base_schedules.constant",
    "policy.base_schedules.exponential",
    "policy.base_schedules.clipped",
    "policy.base_schedules.bspline",
    "policy.schedules.alternating",
    "policy.schedules.sigma_and_clip",
    "policy.schedules.policy_and_clip",
    "policy.schedules.dynamic_dpsgd",
    "policy.schedules.warmup_alternating",
    "policy.schedules.parallel_sigma_and_clip",
    "policy.stateful_schedules.median_gradient",
    "networks.mlp.MLP",
    "networks.cnn.CNN",
]:
    importlib.import_module(_mod)


@pytest.fixture()
def _singleton_max_sigma():
    from conf.config import Config, EnvConfig, ScheduleOptimizerConfig, SweepConfig, WandbConfig
    from conf.singleton_conf import SingletonConfig

    SingletonConfig.config = Config(
        wandb_conf=WandbConfig(),
        sweep=SweepConfig(
            env=EnvConfig(), schedule_optimizer=ScheduleOptimizerConfig(max_sigma=10.0)
        ),
    )
    yield 10.0
    SingletonConfig.config = None
