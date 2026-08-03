import os
import subprocess
from dataclasses import dataclass
from tempfile import NamedTemporaryFile

import tyro
from _slurm_account import account_argv

os.environ["PROJECT_ROOT"] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."),
)
os.environ["PROJECT_SOURCE_ROOT"] = os.path.abspath(
    os.path.join(os.environ["PROJECT_ROOT"], "src"),
)

_THIS_SCRIPT = os.path.abspath(__file__)


@dataclass
class Runtime:
    days: int = 0
    hours: int = 12
    minutes: int = 0
    seconds: int = 0
    short: bool = False  # sub-3hr job-chaining preset (2h55m)
    medium: bool = False  # sub-12hr job-chaining preset (11h55m)

    @property
    def slurm_timestamp(self):
        """
        Convert a timeframe to a format acceptable by slurm. i.e. dd-hh:mm:ss
        """
        if self.short:
            return "00-02:55:00"
        if self.medium:
            return "00-11:55:00"
        minutes = self.minutes + 5  # Add 5 minutes to account for setup and teardown
        minutes += self.seconds // 60
        seconds = self.seconds % 60
        hours = self.hours + minutes // 60
        minutes = minutes % 60
        days = self.days + hours // 24
        hours = hours % 24
        return f"{days:02}-{hours:02}:{minutes:02}:{seconds:02}"


@dataclass
class SlurmConfig:
    runtime: Runtime
    run_id: str
    jobname: str = "test"
    logfile: str = os.path.join(
        os.environ["PROJECT_ROOT"],
        "cc",
        "logs",
        "%j",
        "%x.log",
    )
    project_dir: str = os.environ["PROJECT_SOURCE_ROOT"]
    cpus_per_task: int = 2
    gpus: int = 1
    mem_per_gpu: str = "12G"
    account: str = "aip-nidhih"
    wandb_proj: str = "Testing Mu-gdp"
    prerequisites: tuple[str, ...] = ()
    wandb_dir: str = ""
    """Parent directory for W&B local run storage. Defaults to /scratch/$USER/wandb.

    Offline runs only reach the cloud when `wandb sync` succeeds, so the run dir
    must outlive the job — SLURM_TMPDIR is wiped at job end, which made a missed
    sync unrecoverable. It cannot go in the project directory either: a run dir
    is ~300MB-1GB, so a full sweep would put hundreds of GB on the shared
    filesystem. Persistent scratch is the one location that is both.
    """

    @property
    def resolved_wandb_dir(self) -> str:
        return self.wandb_dir or os.path.expandvars("/scratch/$USER/wandb")

    @property
    def slurm_job_name(self) -> str:
        """SLURM job name, guaranteed to contain the run_id.

        ``--dependency=singleton`` (see below) limits execution to one job per
        (name, user) at a time, so the name MUST encode the run_id for that
        scoping to be per-run_id.  This is idempotent: the initial submit already
        carries the run_id in ``--jobname`` (create_experiments.py), and chain
        continuations inherit this exact name via ``CHAIN_JOBNAME`` — so we only
        append when it is missing (e.g. a manual submit with a generic name).
        """
        if self.run_id in self.jobname:
            return self.jobname
        return f"{self.jobname}-{self.run_id}"

    @property
    def main_args(self) -> str:
        return (
            f' --wandb_conf.project="{self.wandb_proj}"'
            f' --wandb-conf.entity "psaunder"'
            f" --wandb-conf.mode offline"
            f" --wandb-conf.wandb-sync-interval-secs 300"
            f' --wandb-conf.wandb-dir "{self.resolved_wandb_dir}"'
            f' --wandb-conf.restart_run_id="{self.run_id}"'
            f' --wandb-conf.checkpoint_run_id="{self.run_id}"'
        )

    @property
    def sbatch_file(self) -> str:
        return f"""#!/bin/bash
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH --gpus={self.gpus} # Remove this line to run using CPU only
#SBATCH --gpus-per-node={self.gpus}
#SBATCH --mem-per-gpu={self.mem_per_gpu}
#SBATCH --time={self.runtime.slurm_timestamp}
#SBATCH --output={self.logfile}
#SBATCH --job-name={self.slurm_job_name}
#SBATCH --chdir={self.project_dir}
#SBATCH --account={self.account}

# Job-chaining context (read by main.py to resubmit on graceful shutdown).
# CHAIN_JOBNAME carries the run_id-bearing name unchanged so every continuation
# resubmits under the SAME --job-name, keeping the singleton dependency scoped
# to this run_id across the whole chain.
export CHAIN_RESUBMIT_SCRIPT="{_THIS_SCRIPT}"
export CHAIN_WANDB_PROJ="{self.wandb_proj}"
export CHAIN_JOBNAME="{self.slurm_job_name}"
export CHAIN_ACCOUNT="{self.account}"
export CHAIN_MEM_PER_GPU="{self.mem_per_gpu}"
export CHAIN_WANDB_DIR="{self.resolved_wandb_dir}"

# Startup printing
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
echo

echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

echo "starting training..."
echo tmpdir: $SLURM_TMPDIR
echo wandb_dir: {self.resolved_wandb_dir}
echo main_args: {self.main_args}
mkdir -p "{self.resolved_wandb_dir}"
time uv run --no-sync main.py {self.main_args}
TRAIN_EXIT=$?

# Belt-and-braces re-sync.  main.py already syncs after run.finish(), so this is
# normally a no-op; it exists for the case where the process never got that far
# (SIGKILL on wall clock, OOM, node failure).  Offline run dirs live in persistent
# scratch rather than SLURM_TMPDIR, so they are still here after such a kill, and
# `wandb sync` is incremental and idempotent.
# Keyed on the run id rather than the `latest-run` symlink, which races when
# several jobs share this wandb directory.
for run_dir in "{self.resolved_wandb_dir}"/wandb/offline-run-*-{self.run_id}; do
    if [ -d "$run_dir" ]; then
        echo "Re-syncing $run_dir"
        uv run --no-sync wandb sync "$run_dir"
    fi
done

# End printing
echo "Job finished with exit code $TRAIN_EXIT at: `date`"

# Exit with main.py's status, not the bookkeeping above it.  A bash script exits
# with the status of its last command, so without this every job reports
# COMPLETED to SLURM however training ended — and `sacct` stops being able to
# tell a crashed run from a successful one.
exit $TRAIN_EXIT
""".strip()


if __name__ == "__main__":
    conf = tyro.cli(SlurmConfig)

    tmpdir = os.path.expandvars(os.path.abspath("/scratch/$USER"))
    with NamedTemporaryFile(mode="w", suffix=".sh", dir=tmpdir) as f:
        print(conf.sbatch_file)
        f.write(conf.sbatch_file)
        f.flush()

        # `-A` on the command line, not just the in-script `#SBATCH --account`,
        # so a stray SBATCH_ACCOUNT in the environment cannot win. See _slurm_account.
        cmd_list = ["sbatch", *account_argv(conf.account)]
        # `singleton`: at most one job with this name (which encodes the run_id)
        # per user runs/suspends at a time.  This serializes chain continuations
        # AND blocks accidental concurrent submissions for the same run_id — the
        # fork that let a second job restart from step 0 and clobber the run in
        # NoMomentumSweep.  `afterany` (predecessor must terminate, any exit
        # status) is kept for explicit ordering; unlike the old `after` (which is
        # satisfied the moment the predecessor merely *starts*) it does not allow
        # the continuation to run alongside a still-live / requeued predecessor.
        deps = ["singleton"]
        if len(conf.prerequisites) > 0:
            deps.append("afterany:" + ",".join(jobid for jobid in conf.prerequisites))
        cmd_list += ["-d", ",".join(deps)]
        cmd_list.append(f"{f.name}")
        cmd = " ".join(cmd_list)

        print(cmd)
        process_out = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
        )
        process_stderr = process_out.stderr.decode("utf-8").strip()
        if len(process_stderr) != 0:
            raise Exception("Could not start job: " + process_stderr)
        output = process_out.stdout.decode("utf-8").strip()
        slurm_job_id = output[-8:].strip()
        out_dir = os.path.abspath(
            os.path.dirname(conf.logfile.replace("%j", slurm_job_id)),
        )
        os.makedirs(out_dir, exist_ok=True)


# cat <sweep-file> | parallel -q uv run cc/slurm/run-starter.py --run_id={} --jobname='"<jobname>"'
# cat <sweep-file> | while read -r id; do python <this-file> --run_id=$id; done
# cat cc/sweeps/xwf6g25p.txt | parallel -q uv run cc/slurm/run-starter.py --run_id={} --jobname='"mnist, e=3.0, T=3000, sigma_and_clip_schedule"'
