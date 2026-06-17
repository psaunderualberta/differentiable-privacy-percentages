"""W&B run initialisation helper.

Encapsulates the three-case branching logic for starting a run:
  1. Branching from a specific historical checkpoint step → new run in
     ``{project}-branched`` with the source run/step recorded in notes.
  2. Fresh start (no restart_run_id) → new run in the configured project,
     with ``resume="allow"`` so an accidental re-launch won't duplicate runs.
  3. Resuming an existing run (restart_run_id set) → continue that run
     without overwriting its stored config.
"""

import os
import subprocess
import threading
from typing import Any

import wandb

from conf.config import SweepConfig, WandbConfig


def _run_wandb_sync(run_dir: str, label: str) -> bool:
    """Run ``wandb sync`` on a run directory once.  Returns True on success.

    ``wandb sync`` is incremental — it only uploads transaction-log entries
    not yet sent — so it is safe to call repeatedly, including on a run that is
    still in progress.  Failures are logged, not raised, so a flaky network
    never propagates into the training process.
    """
    result = subprocess.run(
        ["wandb", "sync", run_dir],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"WARNING: wandb sync ({label}) failed:\n{result.stderr}")
        return False
    return True


class OfflineSyncDaemon:
    """Periodically ``wandb sync``s an in-progress offline run in the background.

    Offline runs never touch the network during training (so they cannot be
    marked "crashed" mid-run), but that also means the dashboard shows nothing
    until a sync happens.  This daemon runs incremental syncs on a timer to
    give near-live dashboards while keeping the training loop itself fully
    decoupled from the network: a stalled sync runs in its own thread and never
    blocks or crashes the run.
    """

    def __init__(self, run_dir: str, interval_secs: int):
        self._run_dir = run_dir
        self._interval = interval_secs
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        # Event.wait returns True only when stop() is signalled; on timeout it
        # returns False, so this syncs every `interval` seconds until stopped.
        while not self._stop.wait(self._interval):
            _run_wandb_sync(self._run_dir, label="periodic")

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="wandb-offline-sync", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the daemon to stop and wait briefly for any in-flight sync.

        Joins with a timeout so a sync hung on the network can never block
        shutdown; the thread is a daemon, so the process can exit regardless.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None


def start_offline_sync_daemon(
    mode: str, run_dir: str, interval_secs: int
) -> OfflineSyncDaemon | None:
    """Start and return a background sync daemon, or None if not applicable.

    Returns None unless ``mode == "offline"`` and ``interval_secs > 0``.
    """
    if mode != "offline" or interval_secs <= 0:
        return None
    daemon = OfflineSyncDaemon(run_dir, interval_secs)
    daemon.start()
    print(f"Started background wandb sync every {interval_secs}s for {run_dir}")
    return daemon


def _resolve_wandb_dir(wandb_config: WandbConfig) -> str | None:
    """Pick the directory for W&B local run storage.

    Online runs stream data out live, so we redirect to the fast per-job
    ``SLURM_TMPDIR`` to avoid filling the shared filesystem.  Offline runs,
    however, only reach the cloud when ``wandb sync`` runs — if that sync
    fails (e.g. the network is still down at job end) the data must remain
    recoverable, so we keep offline runs on persistent storage rather than
    the transient ``SLURM_TMPDIR``.  An explicit ``wandb_dir`` always wins.
    """
    if wandb_config.wandb_dir is not None:
        return wandb_config.wandb_dir
    if wandb_config.mode == "offline":
        return None  # -> ./wandb in the (persistent) working directory
    return os.environ.get("SLURM_TMPDIR", None)


def sync_offline_run(mode: str, run_dir: str) -> None:
    """Upload a finished offline run's local directory to the W&B cloud.

    No-op unless ``mode == "offline"``.  Call *after* ``run.finish()`` so the
    run directory is fully flushed to disk.  Both the normal-exit and the
    job-chain-shutdown paths converge on the same finish sequence, so a single
    call covers both.  A failed sync is logged but not raised: the local run
    dir is left intact for a later manual ``wandb sync``.
    """
    if mode != "offline":
        return
    print(f"Syncing offline run dir to W&B cloud: {run_dir}")
    if _run_wandb_sync(run_dir, label="final"):
        print("wandb sync complete.")
    else:
        print(f"Run dir kept for manual sync: {run_dir}")


def init_wandb_run(
    wandb_config: WandbConfig,
    sweep_config: SweepConfig,
) -> Any:
    """Initialise and return a W&B run.

    Parameters
    ----------
    wandb_config:
        Top-level W&B settings (project, entity, mode, restart/checkpoint IDs).
    sweep_config:
        Experiment config serialised into the W&B run config on fresh starts.
    """
    # When running under SLURM, redirect all W&B local storage to the fast
    # per-job temp directory so the shared filesystem doesn't fill up.
    wandb_dir = _resolve_wandb_dir(wandb_config)

    is_branching = wandb_config.checkpoint_step is not None

    if is_branching:
        branch_project = (wandb_config.project or "runs") + "-branched"
        notes = (
            f"Branched from run {wandb_config.checkpoint_run_id} "
            f"at step {wandb_config.checkpoint_step} "
            f"(original project: {wandb_config.project})"
        )
        return wandb.init(
            project=branch_project,
            entity=wandb_config.entity,
            mode=wandb_config.mode,
            config=sweep_config.to_wandb_conf(),
            notes=notes,
            dir=wandb_dir,
        )

    if wandb_config.restart_run_id is None and wandb_config.checkpoint_run_id is None:
        # Fresh run — pass id=None explicitly so wandb auto-generates an ID.
        return wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity,
            id=wandb_config.restart_run_id,
            mode=wandb_config.mode,
            config=sweep_config.to_wandb_conf(),
            resume="allow",
            dir=wandb_dir,
        )

    # Resuming an existing run — omit config so it is not overwritten.
    run_id = wandb_config.restart_run_id or wandb_config.checkpoint_run_id
    print(f"Continuing run {run_id}")
    return wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        id=run_id,
        mode=wandb_config.mode,
        resume="allow",
        dir=wandb_dir,
    )
