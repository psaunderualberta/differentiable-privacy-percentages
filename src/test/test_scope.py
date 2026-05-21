import pytest

from conf.config import Config, EnvConfig, ScheduleOptimizerConfig, SweepConfig, WandbConfig
from conf.scope import RunContext, current, using


@pytest.fixture
def config():
    yield Config(
        wandb_conf=WandbConfig(),
        sweep=SweepConfig(
            env=EnvConfig(), schedule_optimizer=ScheduleOptimizerConfig(max_sigma=10.0)
        ),
    )


@pytest.fixture
def sweep_config():
    yield (
        SweepConfig(env=EnvConfig(), schedule_optimizer=ScheduleOptimizerConfig(max_sigma=10.0)),
    )


@pytest.fixture
def wandb_config():
    yield WandbConfig()


class TestRunContext:
    def test_current_without_using_throws_error(self):
        with pytest.raises(LookupError):
            _ = current()

    def test_current_with_using_throws_no_error(self, config):
        with using(RunContext(config)):
            _ = current()

    def test_using_produces_expected_config(self, config):
        with using(RunContext(config)):
            assert current().config is config

    def test_nesting_using_sets_correctly(self, wandb_config, sweep_config):
        ca = Config(data_dir="a", wandb_conf=wandb_config, sweep=sweep_config)
        cb = Config(data_dir="b", wandb_conf=wandb_config, sweep=sweep_config)
        cc = Config(data_dir="c", wandb_conf=wandb_config, sweep=sweep_config)
        with using(RunContext(ca)):
            assert current().config.data_dir == "a"
            with using(RunContext(cb)):
                assert current().config.data_dir == "b"
                with using(RunContext(cc)):
                    assert current().config.data_dir == "c"

    def test_nesting_using_unsets_correctly(self, wandb_config, sweep_config):
        ca = Config(data_dir="a", wandb_conf=wandb_config, sweep=sweep_config)
        cb = Config(data_dir="b", wandb_conf=wandb_config, sweep=sweep_config)
        cc = Config(data_dir="c", wandb_conf=wandb_config, sweep=sweep_config)
        with using(RunContext(ca)):
            with using(RunContext(cb)):
                with using(RunContext(cc)):
                    assert current().config.data_dir == "c"
                assert current().config.data_dir == "b"
            assert current().config.data_dir == "a"

    def test_exception_restores_context(self, wandb_config, sweep_config):
        ca = Config(data_dir="a", wandb_conf=wandb_config, sweep=sweep_config)
        cb = Config(data_dir="b", wandb_conf=wandb_config, sweep=sweep_config)
        with using(RunContext(ca)):
            try:
                with using(RunContext(cb)):
                    _ = 1 / 0
            except ZeroDivisionError:
                assert current().config.data_dir == "a"
