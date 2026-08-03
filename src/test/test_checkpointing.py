"""Tests for util/checkpointing.py.

Covers:
- save_checkpoint writes to the correct local directory
- save_checkpoint + load_checkpoint round-trip preserves array values
- load_checkpoint with step=None returns the highest-numbered (latest) step
- load_checkpoint with a specific step returns that step
- load_checkpoint returns start_step = saved_step + 1
- Multiple saves coexist; any step can be restored independently
- load_checkpoint returns None when no local checkpoint exists and
  entity/project are not provided (no W&B network call is attempted)
- Transient W&B failures are retried at both the artifact download and the
  fail-closed existence probe, while permanent ones fail fast

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
import requests
import urllib3

import util.checkpointing as ckpt
import util.wandb_retry as retry_mod
from environments.nes import ESState
from util.checkpointing import load_checkpoint, save_checkpoint
from util.run_lifecycle import TrainingState


def make_state(schedule, opt_state, key, init_key, step, es_state) -> dict[str, Any]:
    """Build the Orbax wire-format dict the checkpointer round-trips.

    Local test helper replacing the removed ``checkpointing.make_state``: it
    bundles state through the ``TrainingState`` façade so these I/O tests
    exercise the exact wire format ``main.py`` writes.  The façade's own
    structure/dtype contract is asserted in ``test_run_lifecycle.py``.
    """
    return TrainingState(
        schedule=schedule,
        opt_state=opt_state,
        key=key,
        init_key=init_key,
        es_state=es_state,
        step=jnp.array(step, dtype=jnp.int32),
    ).as_orbax_dict()


# ---------------------------------------------------------------------------
# Minimal helpers shared across tests
# ---------------------------------------------------------------------------


class _SimpleSchedule(eqx.Module):
    """A minimal eqx.Module that stands in for a real noise/clip schedule."""

    weights: jnp.ndarray

    def get_private_noise_scales(self):
        return self.weights

    def get_private_clips(self):
        return self.weights


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
    return optimizer.init(eqx.filter(schedule, eqx.is_array))


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
    return make_state(schedule, opt_state, key, init_key, step=42, es_state=None)


@pytest.fixture(autouse=True)
def _patch_project_root(tmp_path, monkeypatch):
    """Redirect all checkpoint I/O to a temporary directory for every test."""
    monkeypatch.setattr(ckpt, "_PROJECT_ROOT", tmp_path)


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

    def test_es_state_preserved(self, schedule, mock_run):
        opt_state = _make_opt_state(schedule)
        es_state = ESState(log_sigma=jnp.float32(jnp.log(0.1)), eta_sigma=jnp.float32(0.05))
        state = make_state(
            schedule, opt_state, jr.PRNGKey(0), jr.PRNGKey(1), step=7, es_state=es_state
        )
        save_checkpoint(state, 7, mock_run)
        result = load_checkpoint(mock_run.id, 7, state, None, None)
        assert result is not None
        restored, _ = result
        assert jnp.allclose(restored["es_state"].log_sigma, es_state.log_sigma)
        assert jnp.allclose(restored["es_state"].eta_sigma, es_state.eta_sigma)

    def test_es_state_none_preserved(self, full_state, mock_run):
        save_checkpoint(full_state, 10, mock_run)
        result = load_checkpoint(mock_run.id, 10, full_state, None, None)
        assert result is not None
        restored, _ = result
        assert restored["es_state"] is None


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

        state_5 = make_state(schedule, opt_state, key, init_key, step=5, es_state=None)
        state_10 = make_state(schedule, opt_state, key, init_key, step=10, es_state=None)

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

        state_5 = make_state(schedule, opt_state, key, init_key, step=5, es_state=None)
        state_10 = make_state(schedule, opt_state, key, init_key, step=10, es_state=None)

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
        state_a = make_state(
            _SimpleSchedule(weights=weights_a), opt_state, key, init_key, step=5, es_state=None
        )
        state_b = make_state(
            _SimpleSchedule(weights=weights_b), opt_state, key, init_key, step=10, es_state=None
        )

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


# ---------------------------------------------------------------------------
# load_checkpoint — W&B fetch failure must not silently restart from step 0
# ---------------------------------------------------------------------------


class _FakeApi:
    """Fake wandb.Api: artifact() always fails to download;
    artifact_collection_exists() reports whether the checkpoint collection
    exists (i.e. whether any checkpoint was ever saved for the run)."""

    def __init__(self, collection_exists: bool):
        self._collection_exists = collection_exists

    def artifact(self, path):
        raise RuntimeError("simulated download failure")

    def artifact_collection_exists(self, name, type_):
        return self._collection_exists


class TestLoadCheckpointFetchFailure:
    def test_raises_when_checkpoint_exists_but_download_fails(self, full_state, monkeypatch):
        # No local checkpoint for this run + a checkpoint DOES exist remotely,
        # but the download fails: restarting from step 0 would clobber the run,
        # so load_checkpoint must raise rather than return None.
        monkeypatch.setattr(ckpt.wandb, "Api", lambda: _FakeApi(collection_exists=True))
        with pytest.raises(RuntimeError, match="clobber"):
            load_checkpoint("some-run", None, full_state, "entity", "project")

    def test_returns_none_when_no_remote_checkpoint_exists(self, full_state, monkeypatch):
        # First job of a chain: no local checkpoint and no remote collection.
        # Starting fresh (None) is correct; must not raise.
        monkeypatch.setattr(ckpt.wandb, "Api", lambda: _FakeApi(collection_exists=False))
        result = load_checkpoint("first-run", None, full_state, "entity", "project")
        assert result is None


# ---------------------------------------------------------------------------
# load_checkpoint — transient network failures must not kill the job
#
# Two of the 30 "never started" FirSweep runs died here: a ~65 s api.wandb.ai
# proxy outage broke the artifact download, and the fail-closed existence probe
# then hit the same dead proxy and reported "a checkpoint exists" for a run that
# had never saved one.  Both calls now retry before any conclusion is drawn.
# ---------------------------------------------------------------------------


def _blip() -> Exception:
    """The 503 tunnel failure seen in the FirSweep logs."""
    return requests.exceptions.ProxyError(
        urllib3.exceptions.ProxyError(
            "Unable to connect to proxy",
            OSError("Tunnel connection failed: 503 Service Unavailable"),
        ),
    )


class _FakeArtifact:
    def __init__(self, path):
        self._path = str(path)

    def download(self):
        return self._path


class _FlakyApi:
    """Serves a checkpoint only after ``artifact_failures`` transient blips.

    ``probe_failures`` blips are likewise raised by the existence probe before
    it answers ``collection_exists``.  Call counts are recorded on the class so
    they survive the ``wandb.Api()`` construction in the code under test.
    """

    artifact_calls = 0
    probe_calls = 0

    def __init__(self, path, artifact_failures=0, probe_failures=0, collection_exists=False):
        self._path = path
        self._artifact_failures = artifact_failures
        self._probe_failures = probe_failures
        self._collection_exists = collection_exists

    def artifact(self, path):
        _FlakyApi.artifact_calls += 1
        if _FlakyApi.artifact_calls <= self._artifact_failures:
            raise _blip()
        if self._path is None:
            raise ValueError(f"artifact {path!r} not found in 'entity/project'")
        return _FakeArtifact(self._path)

    def artifact_collection_exists(self, name, type_):
        _FlakyApi.probe_calls += 1
        if _FlakyApi.probe_calls <= self._probe_failures:
            raise _blip()
        return self._collection_exists


@pytest.fixture
def flaky_api(monkeypatch):
    """Install a _FlakyApi factory; returns a configure(**kwargs) callable."""
    monkeypatch.setattr(retry_mod.time, "sleep", lambda _: None)
    _FlakyApi.artifact_calls = 0
    _FlakyApi.probe_calls = 0

    def configure(**kwargs):
        monkeypatch.setattr(ckpt.wandb, "Api", lambda: _FlakyApi(**kwargs))

    return configure


@pytest.fixture
def remote_checkpoint(full_state, tmp_path):
    """A saved checkpoint standing in for one downloaded from W&B."""
    save_checkpoint(full_state, 42, _MockRun(id="uploaded-run"))
    return tmp_path / "checkpoints" / "uploaded-run" / "42"


class TestLoadCheckpointRetries:
    def test_transient_download_failure_is_retried_then_restores(
        self, full_state, remote_checkpoint, flaky_api
    ):
        flaky_api(path=remote_checkpoint, artifact_failures=2)
        result = load_checkpoint("chain-run", None, full_state, "entity", "project")

        assert result is not None
        _, start_step = result
        assert start_step == 43
        assert _FlakyApi.artifact_calls == 3, "the blip should have been retried, not fatal"
        assert _FlakyApi.probe_calls == 0, "a successful download must not probe at all"

    def test_missing_artifact_is_not_retried(self, full_state, flaky_api):
        # First job of a chain: the artifact genuinely does not exist.  That is
        # a permanent ValueError, so burning the retry window on it would just
        # waste allocated GPU time at every chain start.
        flaky_api(path=None, collection_exists=False)
        assert load_checkpoint("first-run", None, full_state, "entity", "project") is None
        assert _FlakyApi.artifact_calls == 1

    def test_existence_probe_retries_before_failing_closed(self, full_state, flaky_api):
        # The exact shape of the 8oz33jkx / p5avu99n failures: no checkpoint was
        # ever saved, but the probe hit the same dead proxy and failed closed.
        # With retries the probe recovers and the run starts fresh, as it should.
        flaky_api(path=None, artifact_failures=99, probe_failures=2, collection_exists=False)
        assert load_checkpoint("first-run", None, full_state, "entity", "project") is None
        assert _FlakyApi.probe_calls == 3

    def test_still_fails_closed_when_the_outage_outlasts_the_retries(self, full_state, flaky_api):
        # Safety property from the NoMomentumSweep fix is preserved: if we still
        # cannot tell whether a checkpoint exists, refuse to restart from step 0.
        flaky_api(path=None, artifact_failures=99, probe_failures=99)
        with pytest.raises(RuntimeError, match="clobber"):
            load_checkpoint("chain-run", None, full_state, "entity", "project")
        assert _FlakyApi.artifact_calls == retry_mod.DEFAULT_ATTEMPTS
        assert _FlakyApi.probe_calls == retry_mod.DEFAULT_ATTEMPTS

    def test_local_checkpoint_never_touches_the_network(self, full_state, mock_run, flaky_api):
        # Restoring from disk must stay offline-capable.
        flaky_api(path=None, artifact_failures=99, probe_failures=99)
        save_checkpoint(full_state, 42, mock_run)
        result = load_checkpoint(mock_run.id, None, full_state, "entity", "project")

        assert result is not None
        assert _FlakyApi.artifact_calls == 0
