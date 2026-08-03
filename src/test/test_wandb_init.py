"""Teardown ordering and run-dir placement for offline W&B runs.

Both behaviours here were regressions that silently destroyed finished runs'
data in the FirSweep project: 61 runs completed all 1000 outer steps but lost
their sigmas/clips table artifacts because the final `wandb sync` ran before
`run.finish()` had flushed them, and the run dir they were left in was wiped
when the SLURM job ended.
"""

from conf.config import WandbConfig
from util.wandb_init import _resolve_wandb_dir, finish_and_sync


class _RecordingRun:
    def __init__(self, calls):
        self._calls = calls
        self.dir = "/tmp/wandb/offline-run-x/files"

    def finish(self):
        self._calls.append("finish")


class TestFinishAndSyncOrdering:
    """`wandb sync` must run AFTER `run.finish()`.

    In offline mode `wandb.log` hands records to the wandb service
    asynchronously; `wandb sync` uploads whatever is in the transaction log at
    that instant, and `run.finish()` is what flushes the rest. Syncing first
    therefore drops any table the service had not yet written — and since
    nothing syncs afterwards, that data never reaches the cloud.
    """

    def test_syncs_after_finishing(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(
            "util.wandb_init.sync_offline_run", lambda mode, run_dir: calls.append("sync")
        )

        finish_and_sync(_RecordingRun(calls), "offline", "/tmp/wandb/offline-run-x")

        assert calls == ["finish", "sync"]

    def test_syncs_even_when_no_background_daemon_ran(self, monkeypatch):
        # wandb_sync_interval_secs == 0 disables the periodic daemon but must
        # NOT disable the one final sync — otherwise an offline run with the
        # daemon switched off never reaches the cloud at all.
        seen: list[tuple[str, str]] = []
        monkeypatch.setattr(
            "util.wandb_init.sync_offline_run", lambda mode, run_dir: seen.append((mode, run_dir))
        )

        finish_and_sync(_RecordingRun([]), "offline", "/run/dir")

        assert seen == [("offline", "/run/dir")]

    def test_finishes_the_run_for_every_mode(self, monkeypatch):
        # sync_offline_run itself no-ops for non-offline modes; finish must not.
        for mode in ("online", "disabled", "offline"):
            calls: list[str] = []
            monkeypatch.setattr("util.wandb_init.sync_offline_run", lambda mode, run_dir: None)
            finish_and_sync(_RecordingRun(calls), mode, "/run/dir")
            assert calls == ["finish"], mode


class TestResolveWandbDir:
    """Offline runs must be written somewhere that outlives the SLURM job.

    An offline run only reaches the cloud when `wandb sync` succeeds. Putting it
    in SLURM_TMPDIR means a missed sync is unrecoverable — the directory is wiped
    at job end. Online runs stream out live, so the fast scratch dir is right for
    them.
    """

    def test_offline_does_not_use_slurm_tmpdir(self, monkeypatch):
        monkeypatch.setenv("SLURM_TMPDIR", "/scratch/job-123")

        assert _resolve_wandb_dir(WandbConfig(mode="offline")) != "/scratch/job-123"

    def test_online_uses_slurm_tmpdir(self, monkeypatch):
        monkeypatch.setenv("SLURM_TMPDIR", "/scratch/job-123")

        assert _resolve_wandb_dir(WandbConfig(mode="online")) == "/scratch/job-123"

    def test_explicit_wandb_dir_always_wins(self, monkeypatch):
        monkeypatch.setenv("SLURM_TMPDIR", "/scratch/job-123")

        for mode in ("offline", "online", "disabled"):
            conf = WandbConfig(mode=mode, wandb_dir="/persistent/wandb")
            assert _resolve_wandb_dir(conf) == "/persistent/wandb"

    def test_no_slurm_tmpdir_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.delenv("SLURM_TMPDIR", raising=False)

        for mode in ("offline", "online"):
            assert _resolve_wandb_dir(WandbConfig(mode=mode)) is None
