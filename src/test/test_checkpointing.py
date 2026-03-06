"""Tests for util/checkpointing.py.

Covers:
- make_state structure and dtypes
- save_checkpoint writes to the correct local directory
- save_checkpoint + load_checkpoint round-trip preserves array values
- load_checkpoint with step=None returns the highest-numbered (latest) step
- load_checkpoint with a specific step returns that step
- load_checkpoint returns start_step = saved_step + 1
- Multiple saves coexist; any step can be restored independently
- load_checkpoint returns None when no local checkpoint exists and
  entity/project are not provided (no W&B network call is attempted)

All tests run fully offline: W&B is replaced by a lightweight mock run object
so no network calls are made during any part of the test suite.
"""

import dataclasses
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import util.checkpointing as ckpt
from util.checkpointing import load_checkpoint, make_state, save_checkpoint

# ---------------------------------------------------------------------------
# Minimal helpers shared across tests
# ---------------------------------------------------------------------------


class _SimpleSchedule(eqx.Module):
    """A minimal eqx.Module that stands in for a real noise/clip schedule."""

    weights: jnp.ndarray


@dataclasses.dataclass
class _MockRun:
    """Offline replacement for a wandb Run.

    Provides the two attributes/methods that checkpointing.py uses:
    ``id`` (str) and ``log_artifact`` (no-op).  No wandb.init() is ever called.
    """

    id: str = "offline-test-run"

    def log_artifact(self, *args: Any, **kwargs: Any) -> None:
        pass  # no-op: we are not uploading anything in tests


def _make_schedule() -> _SimpleSchedule:
    return _SimpleSchedule(weights=jnp.array([1.0, 2.0, 3.0]))


def _make_opt_state(schedule: _SimpleSchedule) -> Any:
    optimizer = optax.sgd(learning_rate=0.01)
    return optimizer.init(schedule)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_run() -> _MockRun:
    return _MockRun()


@pytest.fixture
def schedule() -> _SimpleSchedule:
    return _make_schedule()


@pytest.fixture
def full_state(schedule) -> dict[str, Any]:
    opt_state = _make_opt_state(schedule)
    key = jr.PRNGKey(0)
    init_key = jr.PRNGKey(1)
    return make_state(schedule, opt_state, key, init_key, step=42)


@pytest.fixture(autouse=True)
def _patch_project_root(tmp_path, monkeypatch):
    """Redirect all checkpoint I/O to a temporary directory for every test."""
    monkeypatch.setattr(ckpt, "_PROJECT_ROOT", tmp_path)


# ---------------------------------------------------------------------------
# make_state
# ---------------------------------------------------------------------------


class TestMakeState:
    def test_has_all_required_keys(self, full_state):
        assert set(full_state.keys()) == {"schedule", "opt_state", "key", "init_key", "step"}

    def test_step_is_int32_array(self, full_state):
        assert full_state["step"].dtype == jnp.int32

    def test_step_value_stored(self, full_state):
        assert int(full_state["step"]) == 42

    def test_schedule_preserved(self, full_state, schedule):
        assert jnp.array_equal(full_state["schedule"].weights, schedule.weights)

    def test_keys_are_jax_arrays(self, full_state):
        assert full_state["key"].shape == jr.PRNGKey(0).shape
        assert full_state["init_key"].shape == jr.PRNGKey(0).shape


# ---------------------------------------------------------------------------
# save_checkpoint — local I/O
# ---------------------------------------------------------------------------


class TestSaveCheckpoint:
    def test_creates_checkpoint_directory(self, full_state, mock_run, tmp_path):
        save_checkpoint(full_state, 10, mock_run)
        expected = tmp_path / "checkpoints" / mock_run.id / "10"
        assert expected.exists()

    def test_directory_contains_orbax_files(self, full_state, mock_run, tmp_path):
        save_checkpoint(full_state, 10, mock_run)
        step_dir = tmp_path / "checkpoints" / mock_run.id / "10"
        assert any(step_dir.iterdir()), "Orbax should write at least one file"

    def test_multiple_steps_create_separate_directories(self, full_state, mock_run, tmp_path):
        save_checkpoint(full_state, 5, mock_run)
        save_checkpoint(full_state, 10, mock_run)
        run_dir = tmp_path / "checkpoints" / mock_run.id
        saved_steps = {d.name for d in run_dir.iterdir() if d.is_dir()}
        assert {"5", "10"} == saved_steps


# ---------------------------------------------------------------------------
# load_checkpoint — round-trip correctness
# ---------------------------------------------------------------------------


class TestLoadCheckpointRoundTrip:
    def test_schedule_weights_preserved(self, full_state, mock_run):
        save_checkpoint(full_state, 10, mock_run)
        result = load_checkpoint(mock_run.id, 10, full_state, None, None)
        assert result is not None
        restored_state, _ = result
        assert jnp.allclose(
            restored_state["schedule"].weights,
            full_state["schedule"].weights,
        )

    def test_step_value_preserved(self, full_state, mock_run):
        save_checkpoint(full_state, 10, mock_run)
        result = load_checkpoint(mock_run.id, 10, full_state, None, None)
        assert result is not None
        restored_state, _ = result
        assert int(restored_state["step"]) == 42  # value inside full_state

    def test_key_preserved(self, full_state, mock_run):
        save_checkpoint(full_state, 10, mock_run)
        result = load_checkpoint(mock_run.id, 10, full_state, None, None)
        assert result is not None
        restored_state, _ = result
        assert jnp.array_equal(restored_state["key"], full_state["key"])

    def test_init_key_preserved(self, full_state, mock_run):
        save_checkpoint(full_state, 10, mock_run)
        result = load_checkpoint(mock_run.id, 10, full_state, None, None)
        assert result is not None
        restored_state, _ = result
        assert jnp.array_equal(restored_state["init_key"], full_state["init_key"])


# ---------------------------------------------------------------------------
# load_checkpoint — step selection
# ---------------------------------------------------------------------------


class TestLoadCheckpointStepSelection:
    def test_start_step_is_saved_step_plus_one(self, full_state, mock_run):
        # full_state has step=42; save under label 10; start_step must be 43.
        save_checkpoint(full_state, 10, mock_run)
        result = load_checkpoint(mock_run.id, 10, full_state, None, None)
        assert result is not None
        _, start_step = result
        assert start_step == 43

    def test_load_latest_returns_highest_step(self, schedule, mock_run):
        opt_state = _make_opt_state(schedule)
        key = jr.PRNGKey(0)
        init_key = jr.PRNGKey(1)

        state_5 = make_state(schedule, opt_state, key, init_key, step=5)
        state_10 = make_state(schedule, opt_state, key, init_key, step=10)

        save_checkpoint(state_5, 5, mock_run)
        save_checkpoint(state_10, 10, mock_run)

        result = load_checkpoint(mock_run.id, None, state_10, None, None)
        assert result is not None
        restored_state, start_step = result
        assert int(restored_state["step"]) == 10
        assert start_step == 11

    def test_load_specific_step_ignores_later_saves(self, schedule, mock_run):
        opt_state = _make_opt_state(schedule)
        key = jr.PRNGKey(0)
        init_key = jr.PRNGKey(1)

        state_5 = make_state(schedule, opt_state, key, init_key, step=5)
        state_10 = make_state(schedule, opt_state, key, init_key, step=10)

        save_checkpoint(state_5, 5, mock_run)
        save_checkpoint(state_10, 10, mock_run)

        result = load_checkpoint(mock_run.id, 5, state_5, None, None)
        assert result is not None
        restored_state, start_step = result
        assert int(restored_state["step"]) == 5
        assert start_step == 6

    def test_load_both_steps_independently(self, schedule, mock_run):
        opt_state = _make_opt_state(schedule)
        key = jr.PRNGKey(0)
        init_key = jr.PRNGKey(1)

        weights_a = jnp.array([1.0, 1.0, 1.0])
        weights_b = jnp.array([9.0, 9.0, 9.0])
        state_a = make_state(_SimpleSchedule(weights=weights_a), opt_state, key, init_key, step=5)
        state_b = make_state(_SimpleSchedule(weights=weights_b), opt_state, key, init_key, step=10)

        save_checkpoint(state_a, 5, mock_run)
        save_checkpoint(state_b, 10, mock_run)

        result_a = load_checkpoint(mock_run.id, 5, state_a, None, None)
        result_b = load_checkpoint(mock_run.id, 10, state_b, None, None)

        assert result_a is not None
        assert result_b is not None
        assert jnp.allclose(result_a[0]["schedule"].weights, weights_a)
        assert jnp.allclose(result_b[0]["schedule"].weights, weights_b)


# ---------------------------------------------------------------------------
# load_checkpoint — None returns
# ---------------------------------------------------------------------------


class TestLoadCheckpointNoneReturns:
    def test_unknown_run_id_returns_none(self, full_state):
        # No local checkpoint for this ID and no entity/project provided.
        result = load_checkpoint("nonexistent-run-id", None, full_state, None, None)
        assert result is None

    def test_unknown_step_returns_none(self, full_state, mock_run):
        save_checkpoint(full_state, 10, mock_run)
        result = load_checkpoint(mock_run.id, 99, full_state, None, None)
        assert result is None

    def test_no_entity_no_project_skips_wandb(self, full_state):
        # Confirm that None entity/project does not attempt a W&B API call.
        # If it did, an exception would propagate instead of returning None.
        result = load_checkpoint("any-run", None, full_state, None, None)
        assert result is None
