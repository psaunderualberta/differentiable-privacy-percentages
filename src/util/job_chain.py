"""SLURM job-chaining via wall-clock polling and SIGUSR1 fallback.

The primary mechanism polls ``SLURM_JOB_END_TIME`` (a Unix timestamp set by
SLURM in the job environment) at each outer loop iteration.  When the
remaining wall time drops below ``SweepConfig.shutdown_buffer_secs``, the
training loop checkpoints and resubmits.

SIGUSR1 is kept as a fallback for clusters or environments where
``SLURM_JOB_END_TIME`` is not set.

Usage in main.py::

    from util.job_chain import (
        register_signal_handler,
        resubmit_if_requested,
        shutdown_requested,
        time_limit_approaching,
    )

    register_signal_handler()

    for t in iterator:
        ...
        if shutdown_requested() or time_limit_approaching():
            # force checkpoint here, then break
            break

    if shutdown_requested() or time_limit_approaching():
        logger.finish()
        run.finish()
        resubmit_if_requested(run.id)
        return
"""

import os
import signal
import subprocess
import threading
import time

_shutdown_requested = threading.Event()


def _handle_sigusr1(signum, frame):
    print("SIGUSR1: requesting graceful job-chain shutdown after current step")
    _shutdown_requested.set()


def register_signal_handler() -> None:
    """Register the SIGUSR1 handler.  Call this after wandb.init() so run.id is available."""
    print("SIGUSR1: registering signal handler()")
    signal.signal(signal.SIGUSR1, _handle_sigusr1)


def shutdown_requested() -> bool:
    """Return True if SIGUSR1 has been received."""
    return _shutdown_requested.is_set()


def time_limit_approaching() -> bool:
    """Return True if SLURM wall time will expire within the configured buffer window.

    Reads ``SLURM_JOB_END_TIME`` (Unix timestamp) from the environment and
    compares against ``SweepConfig.shutdown_buffer_secs``.  Returns False
    safely if the variable is absent (local runs, non-SLURM environments).
    """
    job_end = os.environ.get("SLURM_JOB_END_TIME", None)
    if job_end is None:
        return False
    try:
        from conf.singleton_conf import SingletonConfig

        buffer = SingletonConfig.get_sweep_config_instance().shutdown_buffer_secs
        return time.time() >= int(job_end) - buffer
    except ValueError:
        return False


def resubmit_if_requested(run_id: str) -> None:
    """Resubmit a continuation job via run-starter.py if a graceful shutdown was triggered.

    Triggers on either SIGUSR1 receipt or wall-clock time limit approaching.
    Reads job context from CHAIN_* environment variables injected by run-starter.py.
    Does nothing (with a warning) if CHAIN_RESUBMIT_SCRIPT is not set, which is
    the case when running outside SLURM.
    """
    if not (_shutdown_requested.is_set() or time_limit_approaching()):
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
