"""The generated sbatch script's contract with SLURM and with main.py.

``run-starter.py`` lives in ``cc/slurm/`` and is loaded by path here: it is a
launcher, not an importable module, and its filename is not a valid identifier.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_STARTER = Path(__file__).resolve().parents[2] / "cc" / "slurm" / "run-starter.py"
# The training invocation itself — "main.py" alone also matches the CHAIN_* comment.
_TRAIN_CMD = "uv run --no-sync main.py"


def _load_starter():
    sys.path.insert(0, str(_STARTER.parent))  # for `_slurm_account`
    try:
        spec = importlib.util.spec_from_file_location("run_starter", _STARTER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(_STARTER.parent))


@pytest.fixture(scope="module")
def starter():
    return _load_starter()


@pytest.fixture
def conf(starter):
    return starter.SlurmConfig(runtime=starter.Runtime(), run_id="abc123")


class TestExitCodePropagation:
    """The job's exit status must be main.py's.

    The script ends with bookkeeping (a re-sync loop, an echo), and a bash script
    exits with the status of its *last* command — so without an explicit `exit`
    every job reports COMPLETED to SLURM no matter how training ended. That turns
    `sacct` into a useless debugging tool at exactly the moment it is needed: a
    crashed run looks like a successful one.
    """

    def test_the_script_exits_with_the_training_exit_code(self, conf):
        assert conf.sbatch_file.rstrip().endswith("exit $TRAIN_EXIT")

    def test_the_training_exit_code_is_captured_before_the_resync(self, conf):
        body = conf.sbatch_file
        # $? is clobbered by the very next command, so it has to be saved on the
        # line immediately after training.
        train_line = next(i for i, ln in enumerate(body.splitlines()) if _TRAIN_CMD in ln)
        assert body.splitlines()[train_line + 1].strip() == "TRAIN_EXIT=$?"

    @pytest.mark.parametrize("code", [0, 1, 42])
    def test_bash_reproduces_the_code_through_trailing_commands(self, tmp_path, code):
        # The pattern itself, exercised for real: a failing "training" command
        # followed by bookkeeping that succeeds must still exit non-zero.
        script = tmp_path / "job.sh"
        script.write_text(
            f"#!/bin/bash\n(exit {code})\nTRAIN_EXIT=$?\necho bookkeeping\nexit $TRAIN_EXIT\n"
        )
        assert subprocess.run(["bash", str(script)], capture_output=True).returncode == code


class TestOfflineRunDirPlacement:
    """Offline run dirs must outlive the job, but not fill the shared filesystem."""

    def test_defaults_to_persistent_scratch(self, conf, monkeypatch):
        monkeypatch.setenv("USER", "someuser")
        assert conf.resolved_wandb_dir == "/scratch/someuser/wandb"

    def test_an_explicit_dir_wins(self, starter):
        conf = starter.SlurmConfig(
            runtime=starter.Runtime(), run_id="abc123", wandb_dir="/elsewhere/wandb"
        )
        assert conf.resolved_wandb_dir == "/elsewhere/wandb"

    def test_main_is_told_where_to_write(self, conf):
        assert f'--wandb-conf.wandb-dir "{conf.resolved_wandb_dir}"' in conf.main_args

    def test_the_dir_is_created_before_training_starts(self, conf):
        body = conf.sbatch_file
        assert f'mkdir -p "{conf.resolved_wandb_dir}"' in body
        assert body.index("mkdir -p") < body.index(_TRAIN_CMD)

    def test_the_resync_targets_this_run_not_the_latest_symlink(self, conf):
        # `latest-run` races when several jobs share the wandb directory.
        code = [ln for ln in conf.sbatch_file.splitlines() if not ln.lstrip().startswith("#")]
        glob_line = next(ln for ln in code if "offline-run-" in ln)
        assert f"offline-run-*-{conf.run_id}" in glob_line
        assert not any("latest-run" in ln for ln in code)


class TestChainInheritance:
    """Continuations must inherit what the operator chose for the first segment."""

    def test_exports_the_wandb_dir_for_the_continuation(self, conf):
        # Without this a chain launched with an explicit --wandb-dir silently
        # reverts to the default from segment two onwards — the same failure the
        # CHAIN_MEM_PER_GPU export exists to prevent.
        assert f'export CHAIN_WANDB_DIR="{conf.resolved_wandb_dir}"' in conf.sbatch_file

    def test_the_job_name_carries_the_run_id(self, conf):
        # --dependency=singleton is scoped by name, so the name must be per-run.
        assert conf.run_id in conf.slurm_job_name
