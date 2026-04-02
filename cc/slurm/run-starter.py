import os
import subprocess
from dataclasses import dataclass
from tempfile import NamedTemporaryFile

import tyro

os.environ["PROJECT_ROOT"] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."),
)
os.environ["PROJECT_SOURCE_ROOT"] = os.path.abspath(
    os.path.join(os.environ["PROJECT_ROOT"], "src"),
)

_THIS_SCRIPT = os.path.abspath(__file__)


@dataclass
class Runtime:
    days: int = 1
    hours: int = 0
    minutes: int = 0
    seconds: int = 0
    short: bool = False  # sub-3hr job-chaining preset (2h50m + 5m pad = 2h55m)

    @property
    def slurm_timestamp(self):
        """
        Convert a timeframe to a format acceptable by slurm. i.e. dd-hh:mm:ss
        """
        if self.short:
            return "00-02:55:00"
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
    gpus: int = 3
    mem_per_gpu: str = "6G"
    account: str = "aip-nidhih"
    wandb_proj: str = "Testing Mu-gdp"

    @property
    def main_args(self) -> str:
        return (
            f'--wandb_conf.project="{self.wandb_proj}"'
            f' --wandb-conf.entity "psaunder"'
            f' --wandb-conf.mode "online"'
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
#SBATCH --job-name={self.jobname}
#SBATCH --chdir={self.project_dir}
#SBATCH --account={self.account}
{"#SBATCH --signal=USR1@900" if self.runtime.short else ""}

# Job-chaining context (read by main.py's SIGUSR1 handler to resubmit)
export CHAIN_RESUBMIT_SCRIPT="{_THIS_SCRIPT}"
export CHAIN_WANDB_PROJ="{self.wandb_proj}"
export CHAIN_JOBNAME="{self.jobname}"
export CHAIN_ACCOUNT="{self.account}"

# Startup printing
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
echo

echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

echo "starting training..."
echo tmpdir: $SLURM_TMPDIR
echo main_args: {self.main_args}
time uv run main.py {self.main_args}

# End printing
echo "Job finished with exit code $? at: `date`"
""".strip()


if __name__ == "__main__":
    conf = tyro.cli(SlurmConfig)

    with NamedTemporaryFile(mode="w", suffix=".sh", dir="/tmp") as f:
        print(conf.sbatch_file)
        f.write(conf.sbatch_file)
        f.flush()

        print(f"sbatch {f.name}")
        process_out = subprocess.run(
            f"sbatch {f.name}",
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
