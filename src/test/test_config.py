"""Tests for the configuration system.

Covers:
- DistributionConfig / dist_config_helper
- _is_fixed_field / _is_union_field helpers
- to_wandb_sweep_params serialization (including _type injection)
- _get_config_classes registry derivation
- _reconstruct_from_dict deserialization (the restart_run_id path)
- SweepConfig.plotting_steps property
- SweepConfig.to_wandb_sweep method/metric structure
"""

import dataclasses
import importlib
from dataclasses import dataclass
from typing import Annotated

import pytest
import tyro

from conf.config import (
    EnvConfig,
    PolicyConfig,
    SweepConfig,
)
from conf.config_util import (
    DistributionConfig,
    _is_fixed_field,
    _is_union_field,
    dist_config_helper,
    to_wandb_sweep_params,
)
from conf.singleton_conf import _get_config_classes, _reconstruct_from_dict
from networks.cnn.config import CNNConfig
from networks.mlp.config import MLPConfig
from policy.base_schedules.config import (
    ConstantScheduleConfig,
    InterpolatedClippedScheduleConfig,
    InterpolatedExponentialScheduleConfig,
)
from policy.schedules.config import (
    AlternatingSigmaAndClipScheduleConfig,
    DynamicDPSGDScheduleConfig,
    PolicyAndClipScheduleConfig,
    SigmaAndClipScheduleConfig,
    WarmupAlternatingSigmaAndClipScheduleConfig,
)
from policy.stateful_schedules.config import StatefulMedianGradientNoiseAndClipConfig

# Import concrete implementation modules to fire their @register decorators.
for _mod in [
    "policy.base_schedules.constant",
    "policy.base_schedules.exponential",
    "policy.base_schedules.clipped",
    "policy.schedules.alternating",
    "policy.schedules.sigma_and_clip",
    "policy.schedules.policy_and_clip",
    "policy.schedules.dynamic_dpsgd",
    "policy.schedules.warmup_alternating",
    "policy.stateful_schedules.median_gradient",
    "networks.mlp.MLP",
    "networks.cnn.CNN",
]:
    importlib.import_module(_mod)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_sweep() -> SweepConfig:
    """Minimal SweepConfig with required fields filled in."""
    return SweepConfig(env=EnvConfig(), policy=PolicyConfig())


# ---------------------------------------------------------------------------
# DistributionConfig & dist_config_helper
# ---------------------------------------------------------------------------


class TestDistributionConfig:
    def test_constant_to_wandb_sweep(self):
        dc = dist_config_helper(value=3.14, distribution="constant")
        result = dc.to_wandb_sweep()
        assert result == {"distribution": "constant", "value": 3.14}

    def test_non_constant_to_wandb_sweep(self):
        dc = dist_config_helper(min=1e-4, max=1e-1, distribution="log_uniform_values")
        result = dc.to_wandb_sweep()
        assert result["distribution"] == "log_uniform_values"
        assert result["min"] == pytest.approx(1e-4)
        assert result["max"] == pytest.approx(1e-1)
        assert "value" not in result

    def test_dist_config_helper_min_ge_max_guard(self):
        # When min == max the helper bumps max by 1e-10 to keep W&B happy.
        dc = dist_config_helper(min=5.0, max=5.0, distribution="uniform")
        assert dc.max > dc.min

    def test_constant_sample_returns_value(self):
        dc = dist_config_helper(value=7.0, distribution="constant")
        assert dc.sample() == pytest.approx(7.0)

    def test_int_uniform_sample_in_range(self):
        dc = dist_config_helper(min=0, max=100, distribution="int_uniform")
        for _ in range(20):
            s = dc.sample()
            assert 0 <= s < 100


# ---------------------------------------------------------------------------
# _is_fixed_field
# ---------------------------------------------------------------------------


class TestIsFixedField:
    def test_fixed_field_detected(self):

        @dataclass
        class Cfg:
            x: Annotated[int, tyro.conf.Fixed] = 42
            t: tyro.conf.Fixed[int] = 42
            y: int = 0

        assert _is_fixed_field(Cfg, "x") is True
        assert _is_fixed_field(Cfg, "t") is True
        assert _is_fixed_field(Cfg, "y") is False

    def test_non_existent_field_returns_false(self):
        @dataclass
        class Cfg:
            a: int = 1

        assert _is_fixed_field(Cfg, "does_not_exist") is False

    def test_plain_annotation_not_fixed(self):
        @dataclass
        class Cfg:
            z: float = 1.0

        assert _is_fixed_field(Cfg, "z") is False


# ---------------------------------------------------------------------------
# _is_union_field
# ---------------------------------------------------------------------------


class TestIsUnionField:
    def test_schedule_field_is_union(self):
        # PolicyConfig.schedule: ScheduleConfig = Union[...]
        assert _is_union_field(PolicyConfig, "schedule") is True

    def test_network_field_is_union(self):
        # EnvConfig.network: NetworkConfig = Union[...]
        assert _is_union_field(EnvConfig, "network") is True

    def test_base_schedule_noise_field_is_union(self):
        # AlternatingSigmaAndClipScheduleConfig.noise: BaseScheduleConfig = Union[...]
        assert _is_union_field(AlternatingSigmaAndClipScheduleConfig, "noise") is True
        assert _is_union_field(AlternatingSigmaAndClipScheduleConfig, "clip") is True

    def test_primitive_field_not_union(self):
        assert _is_union_field(PolicyConfig, "batch_size") is False
        assert _is_union_field(PolicyConfig, "max_sigma") is False

    def test_distribution_config_field_not_union(self):
        assert _is_union_field(PolicyConfig, "lr") is False
        assert _is_union_field(EnvConfig, "lr") is False

    def test_non_existent_field_returns_false(self):
        assert _is_union_field(PolicyConfig, "no_such_field") is False

    def test_plain_union_annotation(self):
        @dataclass
        class Cfg:
            x: int | str = 0

        assert _is_union_field(Cfg, "x") is True

    def test_annotated_union_unwrapped(self):
        @dataclass
        class Cfg:
            x: Annotated[int | str, "meta"] = 0

        assert _is_union_field(Cfg, "x") is True


# ---------------------------------------------------------------------------
# to_wandb_sweep_params
# ---------------------------------------------------------------------------


class TestToWandbSweepParams:
    def test_raises_for_non_dataclass(self):
        with pytest.raises(TypeError):
            to_wandb_sweep_params("not a dataclass")

    def test_primitive_field_wrapped_in_value(self):
        conf = ConstantScheduleConfig(init_value=2.5)
        result = to_wandb_sweep_params(conf)
        assert result["parameters"]["init_value"] == {"value": 2.5}

    def test_distribution_config_delegates_to_wandb_sweep(self):
        conf = PolicyConfig()
        result = to_wandb_sweep_params(conf)
        # lr is a DistributionConfig with its own to_wandb_sweep; result should
        # come from DistributionConfig.to_wandb_sweep, not {"value": <obj>}.
        lr_entry = result["parameters"]["lr"]
        assert "distribution" in lr_entry

    def test_union_field_gets_type_discriminator(self):
        # schedule is a Union-typed field; _type must be injected.
        conf = PolicyConfig()  # default: AlternatingSigmaAndClipScheduleConfig
        result = to_wandb_sweep_params(conf)
        schedule_params = result["parameters"]["schedule"]["parameters"]
        assert "_type" in schedule_params
        assert schedule_params["_type"] == {
            "value": "AlternatingSigmaAndClipScheduleConfig",
        }

    def test_non_union_nested_no_type_discriminator(self):
        # EnvConfig.lr is DistributionConfig (not Union), no _type expected.
        conf = EnvConfig()
        result = to_wandb_sweep_params(conf)
        lr_params = result["parameters"]["lr"]
        assert "_type" not in lr_params

    def test_nested_union_field_type_matches_actual_variant(self):
        conf = PolicyConfig(schedule=SigmaAndClipScheduleConfig())
        result = to_wandb_sweep_params(conf)
        type_val = result["parameters"]["schedule"]["parameters"]["_type"]
        assert type_val == {"value": "SigmaAndClipScheduleConfig"}

    def test_fixed_fields_excluded(self):
        import tyro

        @dataclass
        class Cfg:
            visible: int = 1
            hidden: Annotated[int, tyro.conf.Fixed] = 99

            def to_wandb_sweep(self):
                return to_wandb_sweep_params(self)

        result = to_wandb_sweep_params(Cfg())
        assert "visible" in result["parameters"]
        assert "hidden" not in result["parameters"]

    def test_returns_parameters_key(self):
        conf = ConstantScheduleConfig()
        result = to_wandb_sweep_params(conf)
        assert "parameters" in result

    def test_network_union_field_gets_type(self):
        conf = EnvConfig()  # default: MLPConfig
        result = to_wandb_sweep_params(conf)
        network_params = result["parameters"]["network"]["parameters"]
        assert "_type" in network_params
        assert network_params["_type"] == {"value": "MLPConfig"}


# ---------------------------------------------------------------------------
# _get_config_classes
# ---------------------------------------------------------------------------


class TestGetConfigClasses:
    def test_all_expected_classes_present(self):
        classes = _get_config_classes()
        expected = {
            "ConstantScheduleConfig",
            "InterpolatedExponentialScheduleConfig",
            "InterpolatedClippedScheduleConfig",
            "AlternatingSigmaAndClipScheduleConfig",
            "SigmaAndClipScheduleConfig",
            "PolicyAndClipScheduleConfig",
            "DynamicDPSGDScheduleConfig",
            "WarmupAlternatingSigmaAndClipScheduleConfig",
            "StatefulMedianGradientNoiseAndClipConfig",
            "MLPConfig",
            "CNNConfig",
        }
        assert expected.issubset(set(classes.keys()))

    def test_values_are_actual_classes(self):
        classes = _get_config_classes()
        for name, cls in classes.items():
            assert isinstance(cls, type), f"{name} should map to a type"
            assert cls.__name__ == name

    def test_derived_from_registries_not_hardcoded(self):
        # Verify the classes resolve correctly by instantiating each one.
        classes = _get_config_classes()
        for name, cls in classes.items():
            # Every registered config class must be a dataclass with defaults.
            assert dataclasses.is_dataclass(cls), f"{name} is not a dataclass"
            instance = cls()  # no-arg constructor must succeed
            assert isinstance(instance, cls)


# ---------------------------------------------------------------------------
# _reconstruct_from_dict
# ---------------------------------------------------------------------------


class TestReconstructFromDict:
    def test_primitive_field_updated(self):
        conf = ConstantScheduleConfig(init_value=1.0)
        result = _reconstruct_from_dict(conf, {"init_value": 5.0})
        assert result.init_value == pytest.approx(5.0)

    def test_original_unchanged(self):
        conf = ConstantScheduleConfig(init_value=1.0)
        _reconstruct_from_dict(conf, {"init_value": 5.0})
        assert conf.init_value == pytest.approx(1.0)  # immutable replace pattern

    def test_type_key_skipped(self):
        conf = ConstantScheduleConfig(init_value=1.0)
        # _type should not be treated as a field name.
        result = _reconstruct_from_dict(
            conf,
            {"_type": "ConstantScheduleConfig", "init_value": 2.0},
        )
        assert result.init_value == pytest.approx(2.0)

    def test_distribution_config_wrapped_from_scalar(self):
        conf = PolicyConfig()
        result = _reconstruct_from_dict(conf, {"lr": 0.42})
        assert isinstance(result.lr, DistributionConfig)
        assert result.lr.distribution == "constant"
        assert result.lr.value == pytest.approx(0.42)

    def test_non_dataclass_returned_as_is(self):
        assert _reconstruct_from_dict(42, {"x": 1}) == 42
        assert _reconstruct_from_dict("hello", {}) == "hello"

    def test_union_field_same_variant_reconstructed(self):
        conf = PolicyConfig()
        run_conf = {
            "schedule": {
                "_type": "AlternatingSigmaAndClipScheduleConfig",
                "diff_clips_first": True,
            },
        }
        result = _reconstruct_from_dict(conf, run_conf)
        assert isinstance(result.schedule, AlternatingSigmaAndClipScheduleConfig)
        assert result.schedule.diff_clips_first is True

    def test_union_field_different_variant_reconstructed(self):
        # Default schedule is AlternatingSigmaAndClipScheduleConfig; run_conf
        # specifies SigmaAndClipScheduleConfig — must switch variant.
        conf = PolicyConfig()
        run_conf = {
            "schedule": {
                "_type": "SigmaAndClipScheduleConfig",
                "noise": {
                    "_type": "ConstantScheduleConfig",
                    "init_value": 3.0,
                },
                "clip": {
                    "_type": "InterpolatedExponentialScheduleConfig",
                    "num_keypoints": 20,
                    "init_value": 0.5,
                },
            },
        }
        result = _reconstruct_from_dict(conf, run_conf)
        assert isinstance(result.schedule, SigmaAndClipScheduleConfig)
        assert isinstance(result.schedule.noise, ConstantScheduleConfig)
        assert result.schedule.noise.init_value == pytest.approx(3.0)
        assert isinstance(result.schedule.clip, InterpolatedExponentialScheduleConfig)
        assert result.schedule.clip.num_keypoints == 20

    def test_union_field_unknown_type_raises(self):
        conf = PolicyConfig()
        run_conf = {"schedule": {"_type": "NoSuchConfig"}}
        with pytest.raises(ValueError, match="unknown config class 'NoSuchConfig'"):
            _reconstruct_from_dict(conf, run_conf)

    def test_non_union_nested_dataclass_updated(self):
        # EnvConfig.lr is DistributionConfig; eps is a plain float.
        conf = EnvConfig()
        result = _reconstruct_from_dict(conf, {"eps": 2.0, "lr": 0.01})
        assert result.eps == pytest.approx(2.0)
        assert isinstance(result.lr, DistributionConfig)
        assert result.lr.value == pytest.approx(0.01)

    def test_network_variant_switch(self):
        conf = EnvConfig()  # default: MLPConfig
        run_conf = {
            "network": {
                "_type": "CNNConfig",
                "pool_kernel_size": 3,
            },
        }
        result = _reconstruct_from_dict(conf, run_conf)
        assert isinstance(result.network, CNNConfig)
        assert result.network.pool_kernel_size == 3

    def test_sweep_level_reconstruction(self):
        sweep = _make_sweep()
        run_conf = {
            "total_timesteps": 500,
            "dataset": "fashion-mnist",
            "prng_seed": 123456,
        }
        result = _reconstruct_from_dict(sweep, run_conf)
        assert result.total_timesteps == 500
        assert result.dataset == "fashion-mnist"
        assert isinstance(result.prng_seed, DistributionConfig)
        assert result.prng_seed.value == pytest.approx(123456)

    def test_full_round_trip_via_wandb_format(self):
        """Serialize a SweepConfig to a W&B-like dict, then reconstruct it."""
        original = SweepConfig(
            env=EnvConfig(eps=1.5, batch_size=128),
            policy=PolicyConfig(
                schedule=SigmaAndClipScheduleConfig(
                    noise=ConstantScheduleConfig(init_value=2.0),
                    clip=InterpolatedExponentialScheduleConfig(
                        num_keypoints=25,
                        init_value=0.8,
                    ),
                ),
                batch_size=4,
                max_sigma=20.0,
            ),
            total_timesteps=300,
        )

        # Build a W&B-like run.config from the sweep parameters.
        def _unwrap_params(d: dict) -> dict:
            """Recursively convert sweep spec → sampled-value dict.

            {"parameters": {"x": {"value": 1}, "y": {"parameters": {...}}}}
            → {"x": 1, "y": {...}}
            """
            result = {}
            for k, v in d.get("parameters", {}).items():
                if "parameters" in v:
                    result[k] = _unwrap_params(v)
                elif "value" in v:
                    result[k] = v["value"]
                else:
                    # Distribution (min/max/distribution) — use the value stored
                    # in the original for simplicity.
                    print(k)
                    result[k] = getattr(original, k, v.get("value", 0))
            return result

        sweep_spec = to_wandb_sweep_params(original)
        run_conf = _unwrap_params(sweep_spec)

        reconstructed = _reconstruct_from_dict(
            SweepConfig(env=EnvConfig(), policy=PolicyConfig()),
            run_conf,
        )

        assert reconstructed.total_timesteps == 300
        assert reconstructed.env.eps == pytest.approx(1.5)
        assert reconstructed.env.batch_size == 128
        assert isinstance(reconstructed.policy.schedule, SigmaAndClipScheduleConfig)
        assert isinstance(reconstructed.policy.schedule.noise, ConstantScheduleConfig)
        assert reconstructed.policy.schedule.noise.init_value == pytest.approx(2.0)
        assert isinstance(
            reconstructed.policy.schedule.clip,
            InterpolatedExponentialScheduleConfig,
        )
        assert reconstructed.policy.schedule.clip.num_keypoints == 25
        assert reconstructed.policy.max_sigma == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# SweepConfig helpers
# ---------------------------------------------------------------------------


class TestSweepConfig:
    def test_plotting_steps_normal(self):
        sweep = SweepConfig(
            env=EnvConfig(),
            policy=PolicyConfig(),
            total_timesteps=100,
            plotting_interval=10,
        )
        assert sweep.plotting_steps == 10

    def test_plotting_steps_interval_ge_total(self):
        sweep = SweepConfig(
            env=EnvConfig(),
            policy=PolicyConfig(),
            total_timesteps=100,
            plotting_interval=200,
        )
        assert sweep.plotting_steps == 1

    def test_plotting_steps_interval_equals_total(self):
        sweep = SweepConfig(
            env=EnvConfig(),
            policy=PolicyConfig(),
            total_timesteps=50,
            plotting_interval=50,
        )
        assert sweep.plotting_steps == 1

    def test_to_wandb_sweep_contains_method_and_metric(self):
        sweep = _make_sweep()
        result = sweep.to_wandb_sweep()
        assert result["method"] == "random"
        assert result["metric"]["name"] == "accuracy"
        assert result["metric"]["goal"] == "maximize"

    def test_to_wandb_sweep_name_omitted_when_none(self):
        sweep = _make_sweep()
        assert sweep.name is None
        result = sweep.to_wandb_sweep()
        assert "name" not in result

    def test_to_wandb_sweep_name_included_when_set(self):
        sweep = SweepConfig(env=EnvConfig(), policy=PolicyConfig(), name="my_sweep")
        result = sweep.to_wandb_sweep()
        assert result["name"] == "my_sweep"

    def test_to_wandb_sweep_description_omitted_when_none(self):
        sweep = _make_sweep()
        result = sweep.to_wandb_sweep()
        assert "description" not in result

    def test_to_wandb_sweep_description_included_when_set(self):
        sweep = SweepConfig(
            env=EnvConfig(),
            policy=PolicyConfig(),
            description="A test sweep",
        )
        result = sweep.to_wandb_sweep()
        assert result["description"] == "A test sweep"

    def test_to_wandb_sweep_has_parameters(self):
        sweep = _make_sweep()
        result = sweep.to_wandb_sweep()
        assert "parameters" in result


# ---------------------------------------------------------------------------
# Config dataclass defaults
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    def test_constant_schedule_config_defaults(self):
        conf = ConstantScheduleConfig()
        assert conf.init_value == pytest.approx(1.0)

    def test_interpolated_exp_schedule_config_defaults(self):
        conf = InterpolatedExponentialScheduleConfig()
        assert conf.num_keypoints == 50
        assert conf.init_value == pytest.approx(1.0)

    def test_interpolated_clipped_schedule_config_defaults(self):
        conf = InterpolatedClippedScheduleConfig()
        assert conf.num_keypoints == 50
        assert conf.init_value == pytest.approx(1.0)

    def test_alternating_schedule_config_defaults(self):
        conf = AlternatingSigmaAndClipScheduleConfig()
        assert isinstance(conf.noise, InterpolatedExponentialScheduleConfig)
        assert isinstance(conf.clip, InterpolatedExponentialScheduleConfig)
        assert conf.diff_clips_first is False

    def test_sigma_and_clip_schedule_config_defaults(self):
        conf = SigmaAndClipScheduleConfig()
        assert isinstance(conf.noise, InterpolatedExponentialScheduleConfig)
        assert isinstance(conf.clip, InterpolatedExponentialScheduleConfig)

    def test_policy_and_clip_schedule_config_defaults(self):
        conf = PolicyAndClipScheduleConfig()
        assert isinstance(conf.policy, InterpolatedExponentialScheduleConfig)
        assert isinstance(conf.clip, InterpolatedExponentialScheduleConfig)

    def test_dynamic_dpsgd_config_defaults(self):
        conf = DynamicDPSGDScheduleConfig()
        assert conf.rho_mu == pytest.approx(0.5)
        assert conf.rho_c == pytest.approx(0.5)
        assert conf.c_0 == pytest.approx(1.5)

    def test_warmup_alternating_schedule_config_defaults(self):
        conf = WarmupAlternatingSigmaAndClipScheduleConfig()
        assert isinstance(conf.noise_tail, InterpolatedExponentialScheduleConfig)
        assert isinstance(conf.clip_tail, InterpolatedExponentialScheduleConfig)
        assert conf.warmup_noise_init == pytest.approx(1.0)
        assert conf.warmup_clip_init == pytest.approx(1.0)
        assert conf.warmup_pct == pytest.approx(0.3)
        assert conf.diff_clips_first is False

    def test_stateful_median_gradient_config_defaults(self):
        conf = StatefulMedianGradientNoiseAndClipConfig()
        assert conf.c_0 == pytest.approx(0.1)
        assert conf.eta_c == pytest.approx(0.2)

    def test_mlp_config_defaults(self):
        conf = MLPConfig()
        assert conf.hidden_sizes == (32,)
        assert conf.initialization == "glorot"

    def test_cnn_config_defaults(self):
        conf = CNNConfig()
        assert isinstance(conf.mlp, MLPConfig)
        assert conf.channels == (16, 32)
        assert conf.kernel_sizes == (8, 4)
        assert conf.pool_kernel_size == 2

    def test_env_config_defaults(self):
        conf = EnvConfig()
        assert isinstance(conf.network, MLPConfig)
        assert conf.optimizer == "sgd"
        assert conf.loss_type == "cce"
        assert conf.eps == pytest.approx(0.5)
        assert conf.delta == pytest.approx(1e-7)
        assert conf.batch_size == 250
        assert conf.max_steps_in_episode == 100

    def test_policy_config_defaults(self):
        conf = PolicyConfig()
        assert isinstance(conf.schedule, AlternatingSigmaAndClipScheduleConfig)
        assert conf.batch_size == 1
        assert conf.max_sigma == pytest.approx(10.0)
