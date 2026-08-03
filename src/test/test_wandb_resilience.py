"""Tests for the network-resilience layer around the startup W&B calls.

Background: 30 FirSweep runs died with `_runtime: 0` because a ~65-second
`api.wandb.ai` proxy outage hit one of the two *mandatory-online* calls that
happen before `wandb.init`:

  1. ``conf/singleton_conf.get_wandb_run_conf`` — fetches the source run's
     config when resuming (`restart_run_id` / `checkpoint_run_id`).
  2. ``util/checkpointing.load_checkpoint`` — downloads the checkpoint
     artifact, and (on failure) probes whether one exists remotely.

Neither retried, so a blip was fatal and the job died before anything was
logged.  This module covers the two mitigations:

- ``util/wandb_retry.py``     — retry transient failures with exponential backoff.
- ``util/run_conf_cache.py``  — persist a fetched run config so a later chain
                                job can start even while W&B is unreachable.

Everything here runs offline: W&B is mocked and ``time.sleep`` is replaced so
the backoff costs no wall-clock time.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests
import urllib3

import util.run_conf_cache as cache_mod
import util.wandb_retry as retry_mod
import wandb
from conf.config import WandbConfig
from conf.singleton_conf import get_wandb_run_conf
from util.run_conf_cache import read_run_conf, write_run_conf
from util.wandb_retry import backoff_delays, is_transient, retry_transient

# ---------------------------------------------------------------------------
# Helpers — exceptions shaped like the ones seen in the failure logs
# ---------------------------------------------------------------------------


def _proxy_error() -> Exception:
    """The 503 tunnel failure that killed 28 of the 30 runs."""
    return requests.exceptions.ProxyError(
        urllib3.exceptions.ProxyError(
            "Unable to connect to proxy",
            OSError("Tunnel connection failed: 503 Service Unavailable"),
        ),
    )


def _auth_error_from_timeout() -> Exception:
    """What wandb raises when the API-key check times out through the proxy."""
    err = wandb.errors.AuthenticationError("Unable to connect to server to verify API token.")
    err.__cause__ = requests.exceptions.ReadTimeout("Read timed out. (read timeout=20)")
    return err


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Replace the backoff sleep with a recorder so tests never actually wait."""
    slept: list[float] = []
    monkeypatch.setattr(retry_mod.time, "sleep", slept.append)
    return slept


@pytest.fixture(autouse=True)
def _cache_in_tmp(tmp_path, monkeypatch):
    """Scope the on-disk run-config cache to tmp_path for every test."""
    monkeypatch.setenv(cache_mod.CACHE_DIR_ENV_VAR, str(tmp_path / "run-configs"))


def _wandb_conf() -> WandbConfig:
    return WandbConfig(entity="psaunder", project="FirSweep")


# ---------------------------------------------------------------------------
# is_transient — which failures are worth retrying
# ---------------------------------------------------------------------------


class TestIsTransient:
    @pytest.mark.parametrize(
        "exc",
        [
            _proxy_error(),
            _auth_error_from_timeout(),
            requests.exceptions.ReadTimeout("read timed out"),
            requests.exceptions.ConnectionError("connection reset"),
            urllib3.exceptions.ReadTimeoutError(None, "url", "timed out"),
            wandb.errors.CommError("network failure"),
            TimeoutError("timed out"),
            ConnectionResetError("peer reset"),
        ],
    )
    def test_network_failures_are_transient(self, exc):
        assert is_transient(exc)

    @pytest.mark.parametrize(
        "exc",
        [
            ValueError("Could not find run 'nope'"),
            wandb.errors.UsageError("bad argument"),
            KeyError("missing config key"),
        ],
    )
    def test_programming_and_lookup_errors_are_not_transient(self, exc):
        assert not is_transient(exc)

    def test_follows_the_cause_chain(self):
        """wandb wraps network errors; the wrapper type alone must not decide."""
        outer = RuntimeError("checkpoint could not be restored")
        outer.__cause__ = _proxy_error()
        assert is_transient(outer)

    def test_follows_the_context_chain(self):
        try:
            try:
                raise _proxy_error()
            except Exception:
                raise RuntimeError("wrapped")  # noqa: B904 — implicit __context__ is the point
        except RuntimeError as e:
            assert is_transient(e)

    def test_chain_walk_survives_a_cycle(self):
        a = RuntimeError("a")
        b = RuntimeError("b")
        a.__cause__ = b
        b.__cause__ = a
        assert not is_transient(a)  # terminates rather than hanging


# ---------------------------------------------------------------------------
# backoff_delays — the retry schedule
# ---------------------------------------------------------------------------


class TestBackoffDelays:
    def test_one_fewer_delay_than_attempts(self):
        # N attempts means N-1 waits between them; no sleep after the last.
        assert len(backoff_delays(attempts=5, base_delay=1.0, max_delay=100.0)) == 4

    def test_single_attempt_never_sleeps(self):
        assert backoff_delays(attempts=1, base_delay=1.0, max_delay=100.0) == []

    def test_delays_double(self):
        assert backoff_delays(attempts=4, base_delay=2.0, max_delay=100.0) == [2.0, 4.0, 8.0]

    def test_delays_are_capped(self):
        delays = backoff_delays(attempts=6, base_delay=4.0, max_delay=10.0)
        assert delays == [4.0, 8.0, 10.0, 10.0, 10.0]

    def test_default_window_outlasts_the_observed_outage(self):
        """The Jul-30 outage lasted ~65 s (01:12:10 → 01:13:15).

        The default schedule must keep trying past that, or the fix does not
        actually save a run caught at the start of the window.
        """
        default_window = sum(
            backoff_delays(
                attempts=retry_mod.DEFAULT_ATTEMPTS,
                base_delay=retry_mod.DEFAULT_BASE_DELAY_SECS,
                max_delay=retry_mod.DEFAULT_MAX_DELAY_SECS,
            ),
        )
        assert default_window >= 90.0


# ---------------------------------------------------------------------------
# retry_transient — the retry loop itself
# ---------------------------------------------------------------------------


class TestRetryTransient:
    def test_returns_value_without_sleeping_on_first_success(self, _no_real_sleep):
        calls = []

        def fn():
            calls.append(1)
            return "ok"

        assert retry_transient(fn, what="fetch") == "ok"
        assert len(calls) == 1
        assert _no_real_sleep == []

    def test_retries_a_transient_failure_then_succeeds(self, _no_real_sleep):
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise _proxy_error()
            return "ok"

        assert retry_transient(fn, what="fetch", base_delay=2.0, jitter=0.0) == "ok"
        assert len(calls) == 3
        assert _no_real_sleep == [2.0, 4.0]

    def test_reraises_the_last_error_after_exhausting_attempts(self, _no_real_sleep):
        calls = []

        def fn():
            calls.append(1)
            raise _auth_error_from_timeout()

        with pytest.raises(wandb.errors.AuthenticationError):
            retry_transient(fn, what="fetch", attempts=4, jitter=0.0)
        assert len(calls) == 4
        assert len(_no_real_sleep) == 3

    def test_non_transient_failure_is_not_retried(self, _no_real_sleep):
        calls = []

        def fn():
            calls.append(1)
            raise ValueError("Could not find run 'nope'")

        with pytest.raises(ValueError):
            retry_transient(fn, what="fetch")
        assert len(calls) == 1, "a permanent error must fail fast, not burn the retry window"
        assert _no_real_sleep == []

    def test_jitter_stays_within_bounds(self, _no_real_sleep):
        def fn():
            raise _proxy_error()

        with pytest.raises(requests.exceptions.ProxyError):
            retry_transient(fn, what="fetch", attempts=3, base_delay=10.0, jitter=0.25)

        assert len(_no_real_sleep) == 2
        assert 7.5 <= _no_real_sleep[0] <= 12.5
        assert 15.0 <= _no_real_sleep[1] <= 25.0


# ---------------------------------------------------------------------------
# run_conf_cache — the offline fallback store
# ---------------------------------------------------------------------------


class TestRunConfCache:
    def test_round_trips_a_nested_config(self):
        conf = {"env": {"eps": 10.0, "T": 5000}, "dataset": "mnist", "seed": 3}
        write_run_conf("psaunder", "FirSweep", "abc123", conf)
        assert read_run_conf("psaunder", "FirSweep", "abc123") == conf

    def test_miss_returns_none(self):
        assert read_run_conf("psaunder", "FirSweep", "never-written") is None

    def test_entries_are_keyed_by_entity_project_and_run(self):
        write_run_conf("psaunder", "FirSweep", "abc123", {"which": "a"})
        write_run_conf("psaunder", "OtherProject", "abc123", {"which": "b"})
        write_run_conf("someone", "FirSweep", "abc123", {"which": "c"})

        assert read_run_conf("psaunder", "FirSweep", "abc123") == {"which": "a"}
        assert read_run_conf("psaunder", "OtherProject", "abc123") == {"which": "b"}
        assert read_run_conf("someone", "FirSweep", "abc123") == {"which": "c"}

    def test_rewrite_replaces_the_entry(self):
        write_run_conf("psaunder", "FirSweep", "abc123", {"n": 1})
        write_run_conf("psaunder", "FirSweep", "abc123", {"n": 2})
        assert read_run_conf("psaunder", "FirSweep", "abc123") == {"n": 2}

    def test_corrupt_entry_reads_as_a_miss(self, tmp_path):
        write_run_conf("psaunder", "FirSweep", "abc123", {"n": 1})
        path = cache_mod.cache_path("psaunder", "FirSweep", "abc123")
        path.write_text("{ this is not json")
        # A truncated write must degrade to "no cache", not crash the job.
        assert read_run_conf("psaunder", "FirSweep", "abc123") is None

    def test_unserialisable_config_does_not_raise(self):
        # Caching is best-effort: it must never take down a run that fetched fine.
        write_run_conf("psaunder", "FirSweep", "abc123", {"bad": object()})
        assert read_run_conf("psaunder", "FirSweep", "abc123") is None

    def test_unwritable_cache_dir_does_not_raise(self, tmp_path, monkeypatch):
        blocked = tmp_path / "blocked"
        blocked.write_text("i am a file, not a directory")
        monkeypatch.setenv(cache_mod.CACHE_DIR_ENV_VAR, str(blocked))
        write_run_conf("psaunder", "FirSweep", "abc123", {"n": 1})
        assert read_run_conf("psaunder", "FirSweep", "abc123") is None

    def test_writes_are_atomic(self, tmp_path):
        """No partial file is left behind if serialisation dies mid-write."""
        write_run_conf("psaunder", "FirSweep", "abc123", {"n": 1})
        path = cache_mod.cache_path("psaunder", "FirSweep", "abc123")
        assert json.loads(path.read_text()) == {"n": 1}
        assert list(path.parent.glob("*.tmp*")) == []


# ---------------------------------------------------------------------------
# get_wandb_run_conf — touchpoint 1, end to end
# ---------------------------------------------------------------------------


def _patched_api(side_effects):
    """Patch conf.singleton_conf.wandb so Api().run() replays ``side_effects``."""
    mock_wandb = MagicMock()
    mock_wandb.Api.return_value.run.side_effect = side_effects
    return patch("conf.singleton_conf.wandb", mock_wandb), mock_wandb


def _api_run(conf: dict) -> MagicMock:
    run = MagicMock()
    run.config = conf
    return run


class TestGetWandbRunConf:
    def test_transient_failure_is_retried_then_succeeds(self):
        conf = {"dataset": "mnist", "num_outer_steps": 1000}
        ctx, mock_wandb = _patched_api([_proxy_error(), _auth_error_from_timeout(), _api_run(conf)])
        with ctx:
            assert get_wandb_run_conf(_wandb_conf(), "abc123") == conf
        assert mock_wandb.Api.return_value.run.call_count == 3

    def test_successful_fetch_populates_the_cache(self):
        conf = {"dataset": "mnist", "num_outer_steps": 1000}
        ctx, _ = _patched_api([_api_run(conf)])
        with ctx:
            get_wandb_run_conf(_wandb_conf(), "abc123")
        assert read_run_conf("psaunder", "FirSweep", "abc123") == conf

    def test_falls_back_to_the_cache_when_the_network_stays_down(self):
        conf = {"dataset": "mnist", "num_outer_steps": 1000}
        write_run_conf("psaunder", "FirSweep", "abc123", conf)

        ctx, mock_wandb = _patched_api(_proxy_error())
        with ctx:
            # A chain continuation must survive an outage longer than the
            # retry window, since its config was already fetched once.
            assert get_wandb_run_conf(_wandb_conf(), "abc123") == conf
        assert mock_wandb.Api.return_value.run.call_count == retry_mod.DEFAULT_ATTEMPTS

    def test_raises_when_the_network_is_down_and_the_cache_is_cold(self):
        ctx, _ = _patched_api(_proxy_error())
        with ctx, pytest.raises(requests.exceptions.ProxyError):
            get_wandb_run_conf(_wandb_conf(), "abc123")

    def test_permanent_error_is_not_masked_by_a_stale_cache(self):
        """A renamed/deleted run must surface, not silently resume old config."""
        write_run_conf("psaunder", "FirSweep", "abc123", {"dataset": "mnist"})
        ctx, mock_wandb = _patched_api(ValueError("Could not find run 'abc123'"))
        with ctx, pytest.raises(ValueError):
            get_wandb_run_conf(_wandb_conf(), "abc123")
        assert mock_wandb.Api.return_value.run.call_count == 1
