"""Tests for the job-orchestration logic in symbolic_regression.py.

All offline — no live cluster, no PySR/Julia. Covers:
- the chain-control decisions (``should_resubmit``, ``should_restore_mirror``)
- the mirror create/restore plumbing (``_rsync``, ``_start_mirror_daemon``) — the
  file-level checkpoint copy/restore round-trips, with PySR output treated as
  opaque bytes
- the fresh-vs-resume branch of ``run_regression`` and the crash-safe final sync
  of ``_fit_with_mirror``, with ``PySRRegressor`` stubbed out
"""

import time

import pandas as pd
import pytest

import symbolic_regression as sr
from symbolic_regression import (
    _RUN_ID,
    _fit_with_mirror,
    _rsync,
    _start_mirror_daemon,
    run_regression,
    should_restore_mirror,
    should_resubmit,
)

_TIMEOUT = 9900  # 2h45m
_PAD = 600  # 10m


class TestShouldResubmit:
    def test_natural_completion_does_not_resubmit(self):
        # fit() returned well before the timeout window -> synthesis is done.
        assert not should_resubmit(
            elapsed_seconds=100,
            timeout_seconds=_TIMEOUT,
            pad_seconds=_PAD,
            chain_depth=0,
            max_chain_jobs=16,
        )

    def test_timeout_resubmits_when_under_cap(self):
        # fit() ran to (near) the timeout -> not done -> resubmit a successor.
        assert should_resubmit(
            elapsed_seconds=_TIMEOUT,
            timeout_seconds=_TIMEOUT,
            pad_seconds=_PAD,
            chain_depth=3,
            max_chain_jobs=16,
        )

    def test_just_inside_pad_window_resubmits(self):
        # elapsed == timeout - pad is NOT a natural completion (boundary).
        assert should_resubmit(
            elapsed_seconds=_TIMEOUT - _PAD,
            timeout_seconds=_TIMEOUT,
            pad_seconds=_PAD,
            chain_depth=0,
            max_chain_jobs=16,
        )

    def test_just_under_pad_window_completes(self):
        assert not should_resubmit(
            elapsed_seconds=_TIMEOUT - _PAD - 1,
            timeout_seconds=_TIMEOUT,
            pad_seconds=_PAD,
            chain_depth=0,
            max_chain_jobs=16,
        )

    def test_depth_cap_stops_chain(self):
        # Hit the timeout but the chain has reached its depth cap -> stop.
        assert not should_resubmit(
            elapsed_seconds=_TIMEOUT,
            timeout_seconds=_TIMEOUT,
            pad_seconds=_PAD,
            chain_depth=16,
            max_chain_jobs=16,
        )

    def test_last_allowed_depth_resubmits(self):
        assert should_resubmit(
            elapsed_seconds=_TIMEOUT,
            timeout_seconds=_TIMEOUT,
            pad_seconds=_PAD,
            chain_depth=15,
            max_chain_jobs=16,
        )


class TestShouldRestoreMirror:
    def test_restore_when_scratch_gone_but_mirror_present(self):
        assert should_restore_mirror(scratch_run_dir_exists=False, mirror_exists=True)

    def test_no_restore_when_scratch_present(self):
        # Back-to-back chained job: scratch survived, resume in place.
        assert not should_restore_mirror(scratch_run_dir_exists=True, mirror_exists=True)

    def test_no_restore_when_no_mirror(self):
        # First job in a chain: nothing to restore, start fresh.
        assert not should_restore_mirror(scratch_run_dir_exists=False, mirror_exists=False)

    def test_no_restore_when_both_present(self):
        assert not should_restore_mirror(scratch_run_dir_exists=True, mirror_exists=False)


# ---------------------------------------------------------------------------
# Tier 1: mirror create / restore plumbing (file-level, PySR-free)
# ---------------------------------------------------------------------------


def _write(path, name, content):
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(content)


def _wait_until(predicate, timeout=3.0, poll=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return predicate()


class TestRsyncMirror:
    def test_creates_mirror_with_identical_contents(self, tmp_path):
        src = tmp_path / "scratch" / "pysr_run"
        dst = tmp_path / "mirror" / "pysr_run"
        _write(src, "checkpoint.pkl", "state-v1")
        _write(src, "hall_of_fame.csv", "eqns")

        _rsync(src, dst)

        assert (dst / "checkpoint.pkl").read_text() == "state-v1"
        assert (dst / "hall_of_fame.csv").read_text() == "eqns"

    def test_delete_removes_stale_mirror_files(self, tmp_path):
        # A file PySR removed from the run dir must not survive in the mirror,
        # or a restore could resurrect a deleted checkpoint.
        src = tmp_path / "scratch"
        dst = tmp_path / "mirror"
        _write(src, "keep.txt", "fresh")
        _write(dst, "keep.txt", "old")
        _write(dst, "stale.txt", "should be deleted")

        _rsync(src, dst)

        assert (dst / "keep.txt").read_text() == "fresh"
        assert not (dst / "stale.txt").exists()

    def test_noop_when_source_missing(self, tmp_path):
        src = tmp_path / "does-not-exist"
        dst = tmp_path / "mirror"

        _rsync(src, dst)

        # No source ⇒ nothing created, nothing raised.
        assert not dst.exists()

    def test_restore_round_trip_after_scratch_purge(self, tmp_path):
        # Simulate a chained job landing after /scratch was wiped: only the
        # mirror exists, so we restore it onto the (absent) run directory.
        run_directory = tmp_path / "scratch" / "sigma" / _RUN_ID
        mirror = tmp_path / "mirror" / "sigma" / _RUN_ID
        _write(mirror, "checkpoint.pkl", "durable-state")

        assert should_restore_mirror(run_directory.exists(), mirror.exists())
        _rsync(mirror, run_directory)

        assert (run_directory / "checkpoint.pkl").read_text() == "durable-state"
        # Now that scratch is populated, a subsequent job resumes in place.
        assert not should_restore_mirror(run_directory.exists(), mirror.exists())


class TestMirrorDaemon:
    def test_daemon_syncs_periodically_then_stops(self, tmp_path):
        src = tmp_path / "scratch"
        mirror = tmp_path / "mirror"
        _write(src, "checkpoint.pkl", "v1")

        ckpt = mirror / "checkpoint.pkl"
        stop, thread = _start_mirror_daemon(src, mirror, interval_secs=0.05)
        try:
            assert _wait_until(lambda: ckpt.exists() and ckpt.read_text() == "v1")
            # A later, larger checkpoint write is picked up by the next sync tick.
            # (Different size avoids rsync's same-size/same-second quick-check skip;
            # real PySR checkpoints grow, so this is not a concern in practice.)
            (src / "checkpoint.pkl").write_text("v2-grown-checkpoint")
            assert _wait_until(lambda: ckpt.read_text() == "v2-grown-checkpoint")
        finally:
            stop.set()
            thread.join(timeout=5)

        assert not thread.is_alive()


# ---------------------------------------------------------------------------
# Tier 1.5: run_regression fresh-vs-resume branch + crash-safe final sync
# ---------------------------------------------------------------------------


class _FakePySR:
    """Stand-in for PySRRegressor that records how it was constructed/fitted."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mode = "fresh"
        self.fit_called = False
        self.fit_variable_names = None

    @classmethod
    def from_file(cls, **kwargs):
        inst = cls(**kwargs)
        inst.mode = "resume"
        return inst

    def fit(self, X, y, variable_names=None):
        self.fit_called = True
        self.fit_variable_names = variable_names


def _df():
    return pd.DataFrame({"sigma": [1.0, 2.0, 3.0], "T": [10, 20, 30], "eps": [0.5, 1.0, 2.0]})


def _conf():
    return sr.PySRConfig(cache_dir="", mirror_sync_secs=3600)


class TestRunRegressionBranch:
    def test_fresh_construction_when_run_dir_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sr, "PySRRegressor", _FakePySR)
        monkeypatch.delenv("SLURM_NTASKS", raising=False)
        output_directory = tmp_path / "sigma"  # run dir absent

        model, _ = run_regression(_df(), "sigma", _conf(), output_directory, procs=2)

        assert model.mode == "fresh"
        assert model.fit_called
        assert model.fit_variable_names == ["T", "eps"]
        assert model.kwargs["output_directory"] == str(output_directory)
        assert model.kwargs["run_id"] == _RUN_ID
        assert model.kwargs["populations"] == 6  # 3 * procs
        assert model.kwargs["warm_start"] is True
        assert model.kwargs["cluster_manager"] is None  # not on SLURM

    def test_resume_when_run_dir_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sr, "PySRRegressor", _FakePySR)
        output_directory = tmp_path / "sigma"
        (output_directory / _RUN_ID).mkdir(parents=True)  # existing run dir

        model, _ = run_regression(_df(), "sigma", _conf(), output_directory, procs=2)

        assert model.mode == "resume"
        assert model.fit_called
        assert model.kwargs["run_directory"] == str(output_directory / _RUN_ID)
        # Resume must not re-pin output_directory/run_id — those come from the pickle.
        assert "output_directory" not in model.kwargs
        assert "run_id" not in model.kwargs

    def test_cluster_manager_slurm_when_ntasks_set(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sr, "PySRRegressor", _FakePySR)
        monkeypatch.setenv("SLURM_NTASKS", "32")
        output_directory = tmp_path / "sigma"

        model, _ = run_regression(_df(), "sigma", _conf(), output_directory, procs=4)

        assert model.kwargs["cluster_manager"] == "slurm"


class TestFitWithMirrorFinalSync:
    def test_final_sync_runs_on_success(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(sr, "_rsync", lambda src, dst: calls.append((src, dst)))
        monkeypatch.setattr(sr, "run_regression", lambda *a, **k: ("MODEL", 1.23))
        run_directory = tmp_path / "scratch" / _RUN_ID
        mirror = tmp_path / "mirror" / _RUN_ID

        model, elapsed = _fit_with_mirror(
            _df(), "sigma", _conf(), tmp_path / "scratch", run_directory, mirror, procs=2
        )

        assert (model, elapsed) == ("MODEL", 1.23)
        assert (run_directory, mirror) in calls  # freshest checkpoint mirrored

    def test_final_sync_runs_even_when_fit_raises(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(sr, "_rsync", lambda src, dst: calls.append((src, dst)))

        def boom(*a, **k):
            raise RuntimeError("julia exploded")

        monkeypatch.setattr(sr, "run_regression", boom)
        run_directory = tmp_path / "scratch" / _RUN_ID
        mirror = tmp_path / "mirror" / _RUN_ID

        with pytest.raises(RuntimeError, match="julia exploded"):
            _fit_with_mirror(
                _df(), "sigma", _conf(), tmp_path / "scratch", run_directory, mirror, procs=2
            )

        # Crash still mirrors the latest state (chain ends; no successor submitted).
        assert (run_directory, mirror) in calls
