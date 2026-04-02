"""SLURM job-chaining via SIGUSR1.

When a job is submitted with ``--runtime.short`` (via run-starter.py), SLURM
sends SIGUSR1 to the process 120 seconds before the walltime expires.  This
module installs a handler that sets a flag the training loop can check, and
provides the resubmit logic that launches a continuation job.

Usage in main.py::

    from util.job_chain import register_signal_handler, resubmit_if_requested, shutdown_requested

    register_signal_handler()

    for t in iterator:
        ...
        if shutdown_requested():
            # force checkpoint here, then break
            break

    if shutdown_requested():
        logger.finish()
        run.finish()
        resubmit_if_requested(run.id)
        return
"""

import os
import signal
import subprocess
import threading

_shutdown_requested = threading.Event()


def _handle_sigusr1(signum, frame):
    print("SIGUSR1: requesting graceful job-chain shutdown after current step")
    _shutdown_requested.set()


def register_signal_handler() -> None:
    """Register the SIGUSR1 handler.  Call this after wandb.init() so run.id is available."""
    signal.signal(signal.SIGUSR1, _handle_sigusr1)


def shutdown_requested() -> bool:
    """Return True if SIGUSR1 has been received."""
    return _shutdown_requested.is_set()


def resubmit_if_requested(run_id: str) -> None:
    """If SIGUSR1 was received, resubmit a continuation job via run-starter.py.

    Reads job context from CHAIN_* environment variables injected by run-starter.py.
    Does nothing (with a warning) if CHAIN_RESUBMIT_SCRIPT is not set, which is
    the case when running outside SLURM.
    """
    if not _shutdown_requested.is_set():
        return

    resubmit_script = os.environ.get("CHAIN_RESUBMIT_SCRIPT")
    if resubmit_script is None:
        print("WARNING: CHAIN_RESUBMIT_SCRIPT not set — job chain ends here.")
        return

    cmd = [
        "uv",
        "run",
        resubmit_script,
        "--run_id",
        run_id,
        "--wandb-proj",
        os.environ.get("CHAIN_WANDB_PROJ", ""),
        "--jobname",
        os.environ.get("CHAIN_JOBNAME", "chain-job"),
        "--account",
        os.environ.get("CHAIN_ACCOUNT", ""),
        "--runtime.short",
    ]
    print(f"Resubmitting: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: resubmit failed:\n{result.stderr}")
    else:
        print(f"Resubmit successful: {result.stdout.strip()}")
