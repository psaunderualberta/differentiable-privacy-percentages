"""Property-based tests for conf/config_util.py and conf/config.py.

Covers: dist_config_helper (DistributionConfig), SweepConfig.plotting_steps.
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from conf.config import EnvConfig, ScheduleOptimizerConfig, SweepConfig
from conf.config_util import dist_config_helper


class TestDistConfigHelperProperties:
    """dist_config_helper always produces a config with min < max."""

    @given(v=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_equal_min_max_gets_bumped(self, v):
        """When min == max, the helper bumps max so that max > min (W&B requirement)."""
        dc = dist_config_helper(min=v, max=v, distribution="uniform")
        assert dc.max > dc.min

    @given(v=st.floats(allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_constant_sample_returns_value(self, v):
        """A constant distribution always returns its stored value exactly."""
        assume(not (np.isnan(v) or np.isinf(v)))
        dc = dist_config_helper(value=v, distribution="constant")
        assert dc.sample() == pytest.approx(v)

    @given(
        lo=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        hi=st.floats(min_value=50.1, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30)
    def test_uniform_sample_in_range(self, lo, hi):
        """Each sample from a uniform distribution lies in [lo, hi]."""
        dc = dist_config_helper(min=lo, max=hi, distribution="uniform")
        for _ in range(5):
            s = dc.sample()
            assert lo <= s <= hi

    @given(
        lo=st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        hi=st.floats(min_value=10.1, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30)
    def test_log_uniform_sample_in_range(self, lo, hi):
        """Each sample from a log_uniform_values distribution lies in [lo, hi]."""
        dc = dist_config_helper(min=lo, max=hi, distribution="log_uniform_values")
        for _ in range(5):
            s = dc.sample()
            assert lo * 0.999 <= s <= hi * 1.001

    @given(
        lo=st.integers(min_value=0, max_value=50),
        size=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=50)
    def test_int_uniform_sample_in_range(self, lo, size):
        """Each int_uniform sample lies in [lo, lo + size) for any valid range."""
        hi = lo + size
        dc = dist_config_helper(min=lo, max=hi, distribution="int_uniform")
        for _ in range(10):
            s = dc.sample()
            assert lo <= s < hi


class TestPlottingStepsProperties:
    """plotting_steps is always a valid, bounded integer regardless of inputs."""

    @given(
        num_outer_steps=st.integers(min_value=1, max_value=10_000),
        plotting_interval=st.integers(min_value=1, max_value=10_000),
    )
    @settings(max_examples=100)
    def test_plotting_steps_at_least_one(self, num_outer_steps, plotting_interval):
        """plotting_steps ≥ 1: even if interval > total, we get at least one plot."""
        sweep = SweepConfig(
            env=EnvConfig(),
            schedule_optimizer=ScheduleOptimizerConfig(),
            num_outer_steps=num_outer_steps,
            plotting_interval=plotting_interval,
        )
        assert sweep.plotting_steps >= 1

    @given(
        num_outer_steps=st.integers(min_value=1, max_value=10_000),
        plotting_interval=st.integers(min_value=1, max_value=10_000),
    )
    @settings(max_examples=100)
    def test_plotting_steps_at_most_num_outer_steps(self, num_outer_steps, plotting_interval):
        """plotting_steps ≤ num_outer_steps: can't plot more times than we train."""
        sweep = SweepConfig(
            env=EnvConfig(),
            schedule_optimizer=ScheduleOptimizerConfig(),
            num_outer_steps=num_outer_steps,
            plotting_interval=plotting_interval,
        )
        assert sweep.plotting_steps <= num_outer_steps
