import os
import subprocess
from dataclasses import dataclass
from tempfile import NamedTemporaryFile

import tyro

os.environ["PROJECT_ROOT"] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
os.environ["PROJECT_SOURCE_ROOT"] = os.path.abspath(
    os.path.join(os.environ["PROJECT_ROOT"], "src")
)


@dataclass
class Runtime:
    days: int = 1
    hours: int = 0
    minutes: int = 0
    seconds: int = 0

    @property
    def slurm_timestamp(self):
        """
        Convert a timeframe to a format acceptable by slurm. i.e. dd-hh:mm:ss
        """
        minutes = self.minutes + 5  # Add 5 minutes to account for setup and teardown
        minutes += self.seconds // 60
        seconds = self.seconds % 60
        hours = self.hours + minutes // 60
        minutes = minutes % 60
        days = self.days + hours // 24
        hours = hours % 24
        return "{:02}-{:02}:{:02}:{:02}".format(days, hours, minutes, seconds)


@dataclass
class SlurmConfig:
    runtime: Runtime
    run_id: str
    jobname: str = "test"
    logfile: str = os.path.join(
        os.environ["PROJECT_ROOT"], "cc", "logs", "%j", "%x.log"
    )
    project_dir: str = os.environ["PROJECT_SOURCE_ROOT"]
    cpus_per_task: int = 2
    gpus: int = 3
    mem_per_gpu: str = "6G"
    account: str = "aip-lelis"

    @property
    def main_args(self) -> str:
        return f'--wandb_conf.project="Testing Mu-gdp" --wandb-conf.entity "psaunder" --wandb-conf.project="Testing Mu-gdp" --wandb-conf.mode "online" --wandb-conf.restart_run_id="{self.run_id}"'

    @property
    def sbatch_file(self) -> str:
        return f"""#!/bin/bash
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH --gpus={self.gpus} # Remove this line to run using CPU only
#SBATCH --gpus-per-node={self.gpus}
#SBATCH --mem-per-gpu={self.mem_per_gpu}
#SBATCH --time={self.runtime.days}-{self.runtime.hours}:{self.runtime.minutes}:{self.runtime.seconds}
#SBATCH --output={self.logfile}
#SBATCH --job-name={self.jobname}
#SBATCH --chdir={self.project_dir}
#SBATCH --account={self.account}

# Startup printing
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
echo

echo "$CUDA_VISIBLE_DEVICES"

echo "starting training..."
echo $SLURM_TMPDIR
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
            f"sbatch {f.name}", shell=True, capture_output=True
        )
        process_stderr = process_out.stderr.decode("utf-8").strip()
        if len(process_stderr) != 0:
            raise Exception("Could not start job: " + process_stderr)
        output = process_out.stdout.decode("utf-8").strip()
        slurm_job_id = output[-8:].strip()
        out_dir = os.path.abspath(
            os.path.dirname(conf.logfile.replace("%j", slurm_job_id))
        )
        os.makedirs(out_dir, exist_ok=True)


# cat <sweep-file> | parallel -q uv run cc/slurm/run-starter.py --run_id={} --jobname='"<jobname>"'
# cat <sweep-file> | while read -r id; do python <this-file> --run_id=$id; done
# cat cc/sweeps/xwf6g25p.txt | parallel -q uv run cc/slurm/run-starter.py --run_id={} --jobname='"mnist, e=3.0, T=3000, sigma_and_clip_schedule"'
