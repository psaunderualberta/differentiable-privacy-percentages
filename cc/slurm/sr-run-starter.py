"""Launch (and chain) symbolic-regression syntheses on SLURM.

Each *target* (sigma, clip, mu) is an independent **synthesis** run as a
self-chaining job: this launcher submits one job per target; that job runs
``symbolic_regression.py`` for ~2h45m of PySR search, then — if it timed out
rather than naturally completing — resubmits its own successor (incrementing
``CHAIN_DEPTH``) by calling this launcher again. See
``docs/adr/0002-symbolic-regression-slurm-job-chaining.md``.

Two single-launch / multi-node gotchas this script honours (do NOT change
without reading the ADR):
  * The synthesis requests ``--ntasks=N`` with ``--cpus-per-task=1`` and NO
    ``--nodes`` pin, so the scheduler can backfill across any node layout.
  * The launcher line is a BARE ``uv run`` (not wrapped in a multi-task
    ``srun``). PySR's ``cluster_manager="slurm"`` runs ``srun`` itself to spawn
    workers; wrapping it would start N copies of the script and shadow that.

Examples:
    # Submit all three target chains (sigma, clip, mu):
    uv run cc/slurm/sr-run-starter.py --cache_dir <cache-dir>

    # Submit one target chain:
    uv run cc/slurm/sr-run-starter.py --cache_dir <cache-dir> --targets sigma

    # Dry run — print the sbatch script(s), submit nothing:
    uv run cc/slurm/sr-run-starter.py --cache_dir <cache-dir> --dry-run
"""

import os
import subprocess
from dataclasses import dataclass, field
from tempfile import NamedTemporaryFile
from typing import Literal

import tyro

os.environ["PROJECT_ROOT"] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."),
)
os.environ["PROJECT_SOURCE_ROOT"] = os.path.abspath(
    os.path.join(os.environ["PROJECT_ROOT"], "src"),
)

_THIS_SCRIPT = os.path.abspath(__file__)


@dataclass
class SRSlurmConfig:
    cache_dir: str
    """Path to the <entity>__<project> dir produced by compile_results_fetch.py."""
    targets: tuple[Literal["sigma", "clip", "mu"], ...] = ("sigma", "clip", "mu")
    """One independent synthesis (and chain) is submitted per target."""
    ntasks: int = 32
    """SLURM tasks = PySR worker processes. No --nodes pin, so this backfills freely."""
    chain_depth: int = 0
    """Depth of the job being submitted; set by resubmission, not by hand."""
    max_chain_jobs: int = 16
    """Hard cap on chain depth, forwarded to symbolic_regression.py."""
    niterations: int = 100_000
    maxsize: int = 25
    timeout_in_seconds: int = 9900  # 2h45m, below the 2h55m wall time
    pad_seconds: int = 600  # 10m natural-completion slack
    scratch_dir: str = "/scratch/$USER/pysr"
    walltime: str = "00-02:55:00"  # ~2h55m: PySR timeout + setup/teardown pad
    jobname: str = ""
    """Defaults to 'sr-<target>' per submitted job when empty."""
    account: str = "aip-nidhih"
    mem_per_cpu: str = "4G"
    logfile: str = os.path.join(os.environ["PROJECT_ROOT"], "cc", "logs", "%j", "%x.log")
    project_dir: str = os.environ["PROJECT_SOURCE_ROOT"]
    prerequisites: tuple[str, ...] = field(default_factory=tuple)
    """Job ids this submission must run after (`-d after:`); set by resubmission."""
    dry_run: bool = False
    """Print the sbatch script(s) without submitting (off-cluster check)."""

    def jobname_for(self, target: str) -> str:
        return self.jobname or f"sr-{target}"

    def sbatch_file(self, target: str) -> str:
        jobname = self.jobname_for(target)
        main_args = (
            f"--cache_dir '{self.cache_dir}'"
            f" --targets {target}"
            f" --niterations {self.niterations}"
            f" --maxsize {self.maxsize}"
            f" --timeout_in_seconds {self.timeout_in_seconds}"
            f" --pad_seconds {self.pad_seconds}"
            f" --max_chain_jobs {self.max_chain_jobs}"
            f" --scratch_dir '{self.scratch_dir}'"
            f" --procs {self.ntasks}"
        )
        return f"""#!/bin/bash
#SBATCH --ntasks={self.ntasks}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu={self.mem_per_cpu}
#SBATCH --time={self.walltime}
#SBATCH --output={self.logfile}
#SBATCH --job-name={jobname}
#SBATCH --chdir={self.project_dir}
#SBATCH --account={self.account}

# Synthesis job-chaining context (read by symbolic_regression.py to resubmit)
export CHAIN_RESUBMIT_SCRIPT="{_THIS_SCRIPT}"
export CHAIN_CACHE_DIR="{self.cache_dir}"
export CHAIN_TARGET="{target}"
export CHAIN_DEPTH="{self.chain_depth}"
export CHAIN_MAX_JOBS="{self.max_chain_jobs}"
export CHAIN_NTASKS="{self.ntasks}"
export CHAIN_ACCOUNT="{self.account}"
export CHAIN_JOBNAME="{jobname}"

echo "Current working directory: `pwd`"
echo "Starting synthesis '{target}' (chain depth {self.chain_depth}) at: `date`"
echo "SLURM_NTASKS: $SLURM_NTASKS   nodes: $SLURM_JOB_NODELIST"
echo

# Single launch: bare `uv run` (NOT srun) so PySR's slurm cluster_manager owns srun.
time uv run symbolic_regression.py {main_args}

echo "Job finished with exit code $? at: `date`"
""".strip()


def _submit(conf: SRSlurmConfig, target: str) -> None:
    sbatch = conf.sbatch_file(target)
    print(sbatch)
    if conf.dry_run:
        print(f"[dry-run] would submit synthesis '{target}'\n")
        return

    tmpdir = os.path.expandvars(os.path.abspath("/scratch/$USER"))
    os.makedirs(tmpdir, exist_ok=True)
    with NamedTemporaryFile(mode="w", suffix=".sh", dir=tmpdir) as f:
        f.write(sbatch)
        f.flush()

        cmd_list = ["sbatch"]
        if conf.prerequisites:
            cmd_list.append("-d after:" + ",".join(conf.prerequisites))
        cmd_list.append(f.name)
        cmd = " ".join(cmd_list)
        print(cmd)

        process_out = subprocess.run(cmd, shell=True, capture_output=True)
        process_stderr = process_out.stderr.decode("utf-8").strip()
        if process_stderr:
            raise Exception("Could not start job: " + process_stderr)
        output = process_out.stdout.decode("utf-8").strip()
        slurm_job_id = output[-8:].strip()
        out_dir = os.path.abspath(os.path.dirname(conf.logfile.replace("%j", slurm_job_id)))
        os.makedirs(out_dir, exist_ok=True)
        print(f"submitted '{target}' as job {slurm_job_id}\n")


if __name__ == "__main__":
    conf = tyro.cli(SRSlurmConfig)
    for tgt in conf.targets:
        _submit(conf, tgt)
