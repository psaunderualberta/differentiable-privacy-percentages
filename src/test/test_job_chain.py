"""What a job-chain continuation inherits from the segment that spawned it."""

import pytest

from util import job_chain

_CHAIN_VARS = (
    "CHAIN_RESUBMIT_SCRIPT",
    "CHAIN_WANDB_PROJ",
    "CHAIN_JOBNAME",
    "CHAIN_ACCOUNT",
    "CHAIN_MEM_PER_GPU",
    "CHAIN_WANDB_DIR",
    "SLURM_JOB_ID",
)


@pytest.fixture
def submitted(monkeypatch):
    """Capture the argv `resubmit_if_requested` would submit."""
    calls: list[list[str]] = []

    class _Result:
        returncode = 0
        stdout = "Submitted batch job 1"
        stderr = ""

    monkeypatch.setattr(
        job_chain.subprocess, "run", lambda cmd, **kw: (calls.append(cmd), _Result())[1]
    )
    for var in _CHAIN_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CHAIN_RESUBMIT_SCRIPT", "/repo/cc/slurm/run-starter.py")
    # resubmit_if_requested is a no-op unless a shutdown was actually latched.
    job_chain._shutdown_requested.set()
    monkeypatch.setattr(job_chain._shutdown_requested, "clear", lambda: None)
    yield calls
    job_chain._shutdown_requested = type(job_chain._shutdown_requested)()


class TestWandbDirInheritance:
    """A continuation must write its offline run dir where the first segment did.

    Nothing else carries the choice across the process boundary: the continuation
    is a fresh `sbatch` of run-starter.py, which otherwise re-derives its own
    default. An operator who redirected the first segment (because scratch was
    full, or shared between users) would silently get the default from segment
    two onwards.
    """

    def test_forwards_the_dir_when_set(self, submitted, monkeypatch):
        monkeypatch.setenv("CHAIN_WANDB_DIR", "/elsewhere/wandb")

        job_chain.resubmit_if_requested("abc123")

        argv = submitted[0]
        assert "--wandb-dir" in argv
        assert argv[argv.index("--wandb-dir") + 1] == "/elsewhere/wandb"

    def test_omits_the_flag_when_unset(self, submitted):
        # Blank would be forwarded to tyro as an empty path; omitting lets
        # run-starter fall back to its own default, as with --account.
        job_chain.resubmit_if_requested("abc123")

        assert "--wandb-dir" not in submitted[0]

    def test_omits_the_flag_when_blank(self, submitted, monkeypatch):
        monkeypatch.setenv("CHAIN_WANDB_DIR", "   ")

        job_chain.resubmit_if_requested("abc123")

        assert "--wandb-dir" not in submitted[0]

    def test_still_forwards_the_run_id_and_memory(self, submitted, monkeypatch):
        monkeypatch.setenv("CHAIN_MEM_PER_GPU", "8G")
        monkeypatch.setenv("CHAIN_WANDB_DIR", "/elsewhere/wandb")

        job_chain.resubmit_if_requested("abc123")

        argv = submitted[0]
        assert argv[argv.index("--run_id") + 1] == "abc123"
        assert argv[argv.index("--mem-per-gpu") + 1] == "8G"
