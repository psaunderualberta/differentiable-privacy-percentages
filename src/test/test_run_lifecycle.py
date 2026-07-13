"""Tests for util/run_lifecycle.py — the RunLifecycle coordinator.

These are boundary tests for the decision logic the old main.py could not
assert: the latched stop decision, the periodic-vs-forced checkpoint cadence,
the restore device-uncommit, and the resubmit-vs-suppress truth table.

All I/O leaves are mocked: ``save_checkpoint`` and ``resubmit_if_requested``
are monkeypatched (the latter is the sbatch boundary and must never spawn a
real job), the wall clock is injected via ``now``, and SingletonConfig is
seeded with an in-memory Config.  No real filesystem, W&B, subprocess, or
signals are touched.
"""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import util.job_chain as job_chain
import util.run_lifecycle as rl
from conf.config import Config, EnvConfig, ScheduleOptimizerConfig, SweepConfig, WandbConfig
from conf.singleton_conf import SingletonConfig
from util.run_lifecycle import RunLifecycle, TrainingState


@pytest.fixture(autouse=True)
def _clear_shutdown_event():
    """The SIGUSR1 stop signal is a module-global Event; clear it around every
    test so a latched stop never leaks into the next test."""
    job_chain._shutdown_requested.clear()
    yield
    job_chain._shutdown_requested.clear()


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_config(**wandb_overrides: Any) -> Config:
    return Config(
        wandb_conf=WandbConfig(**wandb_overrides),
        sweep=SweepConfig(
            env=EnvConfig(), schedule_optimizer=ScheduleOptimizerConfig(max_sigma=10.0)
        ),
    )


@pytest.fixture
def seed_config():
    """Seed SingletonConfig with an in-memory Config; yields a setter so a
    test can choose wandb/sweep overrides before constructing RunLifecycle."""
    ctx: dict[str, Any] = {}

    def _seed(**wandb_overrides: Any) -> Config:
        cfg = _make_config(**wandb_overrides)
        ctx["cm"] = SingletonConfig.override(cfg)
        ctx["cm"].__enter__()
        return cfg

    yield _seed
    if "cm" in ctx:
        ctx["cm"].__exit__(None, None, None)


def _make_training_state(step: int = 0) -> TrainingState:
    return TrainingState(
        schedule={"w": jnp.array([1.0, 2.0, 3.0])},
        opt_state={"m": jnp.array([0.0, 0.0, 0.0])},
        key=jr.PRNGKey(0),
        init_key=jr.PRNGKey(1),
        es_state=None,
        step=jnp.array(step, dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# TrainingState — typed façade over the 6-key Orbax dict
# ---------------------------------------------------------------------------


class TestTrainingState:
    def test_orbax_dict_round_trip_preserves_fields(self):
        state = _make_training_state(step=7)
        d = state.as_orbax_dict()
        restored = TrainingState.from_orbax_dict(d)

        assert restored.schedule == state.schedule
        assert restored.opt_state == state.opt_state
        assert jnp.array_equal(restored.key, state.key)
        assert jnp.array_equal(restored.init_key, state.init_key)
        assert restored.es_state is None
        assert int(restored.step) == 7

    def test_orbax_dict_has_the_six_wire_keys(self):
        d = _make_training_state().as_orbax_dict()
        assert set(d.keys()) == {
            "schedule",
            "opt_state",
            "key",
            "init_key",
            "step",
            "es_state",
        }

    def test_orbax_dict_preserves_int32_step(self):
        # The on-disk wire contract: step is a jnp.int32 array (Orbax restores
        # into a template that must match this dtype exactly).
        d = _make_training_state(step=42).as_orbax_dict()
        assert d["step"].dtype == jnp.int32
        assert int(d["step"]) == 42

    def test_orbax_dict_carries_es_state_through(self):
        es = {"log_sigma": jnp.float32(jnp.log(0.1))}
        state = dataclasses.replace(_make_training_state(), es_state=es)
        assert state.as_orbax_dict()["es_state"] is es


# ---------------------------------------------------------------------------
# restore — startup, before wandb.init
# ---------------------------------------------------------------------------


class TestRestore:
    def test_no_checkpoint_run_id_returns_none_and_zero(self, seed_config):
        seed_config(checkpoint_run_id=None)
        lifecycle = RunLifecycle()
        restored, start_step = lifecycle.restore(_make_training_state())
        assert restored is None
        assert start_step == 0

    def test_round_trip_uncommits_leaves_and_propagates_start_step(self, seed_config, monkeypatch):
        seed_config(checkpoint_run_id="src-run")

        # Simulate Orbax: every array leaf comes back committed to device 0.
        dev = jax.devices()[0]
        committed = TrainingState(
            schedule={"w": jax.device_put(jnp.array([1.0, 2.0, 3.0]), dev)},
            opt_state={"m": jax.device_put(jnp.array([0.0]), dev)},
            key=jax.device_put(jr.PRNGKey(0), dev),
            init_key=jax.device_put(jr.PRNGKey(1), dev),
            es_state=None,
            step=jax.device_put(jnp.array(41, jnp.int32), dev),
        )
        assert committed.schedule["w"].committed  # precondition

        monkeypatch.setattr(rl, "load_checkpoint", lambda *a, **k: (committed.as_orbax_dict(), 42))

        lifecycle = RunLifecycle()
        restored, start_step = lifecycle.restore(_make_training_state())

        assert start_step == 42
        assert restored is not None
        leaves = jax.tree_util.tree_leaves(
            (restored.schedule, restored.opt_state, restored.key, restored.init_key)
        )
        assert all(not leaf.committed for leaf in leaves)
        assert jnp.array_equal(restored.schedule["w"], jnp.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# should_stop — the latched stop decision
# ---------------------------------------------------------------------------


class TestShouldStop:
    def test_not_stopped_when_no_signal(self, seed_config):
        seed_config()
        lifecycle = RunLifecycle()
        assert lifecycle.should_stop() is False
        assert lifecycle.stopped_for_chain is False

    def test_sigusr1_event_latches_job_chain(self, seed_config):
        seed_config()
        lifecycle = RunLifecycle()

        job_chain._shutdown_requested.set()
        assert lifecycle.should_stop() is True
        assert lifecycle.stopped_for_chain is True

        # Latched: stays True even after the underlying signal is cleared.
        job_chain._shutdown_requested.clear()
        assert lifecycle.should_stop() is True
        assert lifecycle.stopped_for_chain is True

    def test_wall_clock_deadline_latches_and_stays(self, seed_config, monkeypatch):
        seed_config()  # shutdown_buffer_secs default = 180
        monkeypatch.setenv("SLURM_JOB_END_TIME", "1000")

        clock = {"t": 800.0}  # 800 < 1000 - 180 = 820 -> not yet approaching
        lifecycle = RunLifecycle(now=lambda: clock["t"])

        assert lifecycle.should_stop() is False
        assert lifecycle.stopped_for_chain is False

        clock["t"] = 850.0  # 850 >= 820 -> deadline crossed
        assert lifecycle.should_stop() is True
        assert lifecycle.stopped_for_chain is True

        # Latched: even if the clock somehow rewinds, the decision holds.
        clock["t"] = 0.0
        assert lifecycle.should_stop() is True
        assert lifecycle.stopped_for_chain is True

    def test_no_slurm_env_never_stops(self, seed_config, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
        seed_config()
        lifecycle = RunLifecycle(now=lambda: 1e18)
        assert lifecycle.should_stop() is False

    def test_slow_step_widens_buffer_and_stops_earlier(self, seed_config, monkeypatch):
        # A step that takes 400s must reserve >> the static 180s buffer: with
        # SLURM_JOB_END_TIME=10000 and buffer 180, a fixed window would only
        # stop at >=9820; but one more 400s step (1.5x = 600s) would overrun the
        # deadline, so the run must latch a stop well before 9820.
        seed_config()  # shutdown_buffer_secs default = 180
        monkeypatch.setenv("SLURM_JOB_END_TIME", "10000")

        clock = {"t": 9000.0}
        lifecycle = RunLifecycle(now=lambda: clock["t"])

        assert lifecycle.should_stop() is False  # first call: no step measured yet
        clock["t"] = 9400.0  # one 400s step elapsed -> window = 180 + 600 = 780
        # 9400 >= 10000 - 780 = 9220 -> stop, even though 9400 < 9820 (static).
        assert lifecycle.should_stop() is True
        assert lifecycle.stopped_for_chain is True

    def test_wall_clock_latch_arms_resubmit_event(self, seed_config, monkeypatch):
        # A stop latched by the wall-clock deadline must set the module-level
        # shutdown event so resubmit_if_requested fires on this path too.
        seed_config()
        monkeypatch.setenv("SLURM_JOB_END_TIME", "1000")
        clock = {"t": 0.0}
        lifecycle = RunLifecycle(now=lambda: clock["t"])

        lifecycle.should_stop()  # measure baseline
        clock["t"] = 990.0  # past the deadline window
        assert lifecycle.should_stop() is True
        assert job_chain._shutdown_requested.is_set()


# ---------------------------------------------------------------------------
# checkpoint — periodic vs forced cadence
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _MockRun:
    id: str = "this-run"

    def log_artifact(self, *a: Any, **k: Any) -> None:
        pass


class TestCheckpoint:
    @pytest.fixture
    def saves(self, monkeypatch):
        """Record (step) for every save_checkpoint call instead of writing."""
        recorded: list[int] = []
        monkeypatch.setattr(rl, "save_checkpoint", lambda state, step, run: recorded.append(step))
        return recorded

    def test_periodic_only_on_interval_multiples(self, seed_config, saves):
        seed_config(checkpoint_every=5)
        lifecycle = RunLifecycle()
        lifecycle.attach(_MockRun())

        for t in range(7):  # steps 0..6; (t+1)%5==0 only at t=4
            lifecycle.checkpoint(_make_training_state(step=t))
        assert saves == [4]

    def test_force_always_saves(self, seed_config, saves):
        seed_config(checkpoint_every=5)
        lifecycle = RunLifecycle()
        lifecycle.attach(_MockRun())

        lifecycle.checkpoint(_make_training_state(step=2), force=True)
        assert saves == [2]

    def test_forced_and_periodic_coexist_without_double_write(self, seed_config, saves):
        seed_config(checkpoint_every=5)
        lifecycle = RunLifecycle()
        lifecycle.attach(_MockRun())

        # Drive the loop the way main.py will: forced checkpoint on stop, else
        # periodic.  Stop fires at t=7; periodic multiple is t=4.
        for t in range(20):
            state = _make_training_state(step=t)
            if t == 7:
                lifecycle.checkpoint(state, force=True)
                break
            lifecycle.checkpoint(state)

        assert saves == [4, 7]
        assert len(saves) == len(set(saves))  # no step written twice


# ---------------------------------------------------------------------------
# finalize — the resubmit-vs-suppress truth table
# ---------------------------------------------------------------------------


class TestFinalize:
    @pytest.fixture
    def resubmits(self, monkeypatch):
        """Record run ids passed to the (mocked) sbatch boundary.  Never spawns
        a real job."""
        recorded: list[str] = []
        monkeypatch.setattr(rl, "resubmit_if_requested", lambda run_id: recorded.append(run_id))
        return recorded

    def test_job_chain_latch_resubmits_once_with_run_id(self, seed_config, resubmits):
        seed_config()
        lifecycle = RunLifecycle()
        lifecycle.attach(_MockRun(id="chain-run"))

        job_chain._shutdown_requested.set()
        assert lifecycle.should_stop()  # latch JOB_CHAIN

        lifecycle.finalize()
        assert resubmits == ["chain-run"]

    def test_keyboard_interrupt_suppresses_resubmit(self, seed_config, resubmits):
        seed_config()
        lifecycle = RunLifecycle()
        lifecycle.attach(_MockRun(id="chain-run"))

        # Even after a job-chain stop was latched, a deliberate Ctrl+C wins.
        job_chain._shutdown_requested.set()
        assert lifecycle.should_stop()
        lifecycle.mark_interrupted()

        lifecycle.finalize()
        assert resubmits == []
        assert lifecycle.stopped_for_chain is False

    def test_normal_completion_does_not_resubmit(self, seed_config, resubmits):
        seed_config()
        lifecycle = RunLifecycle()
        lifecycle.attach(_MockRun())

        # Loop ran to completion: should_stop never latched.
        assert lifecycle.should_stop() is False
        lifecycle.finalize()
        assert resubmits == []
