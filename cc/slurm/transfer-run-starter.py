"""Launch the whole transfer-evaluation DAG on SLURM (ADR 0008, ADR 0015).

One entry point submits five jobs:

    preflight (1 sequential CPU job)
        └─ curve      (job array, 1 task per source regime × target)
        └─ equation   (job array, 1 task per on-grid target)
        └─ reference  (job array, 1 task per reference × target)
              └─ plot (1 CPU job, only with --with-plot)

Each stage is **one job array over a launcher-written manifest**: line *i* of the
manifest is the exact ``uv run transfer_*.py ...`` command array task *i* runs, so
the manifest is the complete record of what was launched. Jobs whose cell parquets
already exist are filtered out at manifest-build time — there is deliberately no
producer-side resumption.

Two conventions that are load-bearing (see ``transfer_launch``):
  * Dependencies are ``afterok`` only, **never** ``--dependency=singleton``.
    ``singleton`` is ``run-starter.py``'s chain-serialisation device and would
    serialise each array to one task at a time, destroying the fan-out.
  * The preflight is sequential and runs alone, because ``util/dataloaders.py``
    downloads into the repo with no locking.

Examples:
    # Dry run — print the manifests, counts and GPU-hour estimate, submit nothing:
    uv run cc/slurm/transfer-run-starter.py \\
        --cache_root /scratch/$USER/transfer \\
        --schedules_parquet <cache>/schedules.parquet \\
        --eval_dir <cache>/pysr_eval/<slug> \\
        --target_datasets eyepacs imagenet chexpert \\
        --target_eps 1.0 8.0 --target_T 200 5000 --dry-run

    # Submit, including the assembler:
    uv run cc/slurm/transfer-run-starter.py ... --with-plot
"""

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile

import tyro
from _slurm_account import account_argv

os.environ["PROJECT_ROOT"] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."),
)
os.environ["PROJECT_SOURCE_ROOT"] = os.path.abspath(
    os.path.join(os.environ["PROJECT_ROOT"], "src"),
)

# transfer_launch is importable from src/ and pulls in no jax, so the launcher can
# compute cell filenames and manifests off-cluster. Same trick as sr_identity.py.
sys.path.insert(0, os.environ["PROJECT_SOURCE_ROOT"])
from transfer_launch import (
    ProducerArgs,
    absolute_path,
    array_sbatch,
    condition_grid,
    expand_targets,
    manifest_text,
    plan_jobs,
    preflight_command,
    serial_sbatch,
)

# Per-stage wall clocks. Reference is the long pole: it sweeps a reference's
# hyperparameters and then evaluates, where the two shape producers only evaluate.
_WALLTIMES = {
    "preflight": "00-02:55:00",
    "curve": "00-02:55:00",
    "equation": "00-02:55:00",
    "reference": "00-11:55:00",
    "plot": "00-02:55:00",
}

# Rough per-task GPU-hours, used only for the --dry-run estimate (analytic, ±3×).
# Per *task*, which is not per cell for every stage: a curve task is one source
# policy (one cell), but an equation task walks every condition at its target and a
# reference task sweeps before it evaluates, so those two scale with what the task
# bundles. Keep these honest against the wall clocks in _WALLTIMES — an estimate
# below the clock is what hides a stage whose tasks cannot finish in time.
_GPU_HOURS = {"curve": 0.8, "equation": 1.2, "reference": 1.7}


@dataclass
class TransferSlurmConfig:
    """Submit the transfer-evaluation DAG for one target cross-product."""

    cache_root: str
    """Where cells are written and the skip filter looks. Required, no default."""

    target_datasets: tuple[str, ...] = ("eyepacs", "imagenet", "chexpert")
    target_eps: tuple[float, ...] = (1.0, 8.0)
    target_T: tuple[int, ...] = (200, 5000)
    target_delta: float = 1e-7

    schedules_parquet: str = ""
    """Source schedules.parquet from compile_results_fetch. Required for the curve stage."""
    eval_dir: str = ""
    """SR eval dir (equations.csv + category_map.json). Required for the equation stage."""

    stages: tuple[str, ...] = ("curve", "equation", "reference")
    """Producer stages to submit."""
    with_plot: bool = False
    """Also submit the assembler, gated behind every producer array."""

    num_reps: int = 8
    """Evaluation seeds per cell, uniform across all three stages."""
    seed: int = 0
    batch_size: int = 250
    throttle: int = 8
    """Max concurrently running tasks per array."""

    account: str = "aip-nidhih"
    cpus_per_task: int = 2
    gpus: int = 1
    mem_per_gpu: str = "12G"
    project_dir: str = os.environ["PROJECT_SOURCE_ROOT"]
    manifest_dir: str = os.path.join(os.environ["PROJECT_ROOT"], "cc", "manifests")
    logdir: str = os.path.join(os.environ["PROJECT_ROOT"], "cc", "logs", "transfer")
    prerequisites: tuple[str, ...] = field(default_factory=tuple)
    dry_run: bool = False
    """Print the manifests, per-stage counts and GPU-hour estimate; submit nothing."""

    @property
    def producer_args(self) -> ProducerArgs:
        # Absolute, always — see transfer_launch.absolute_path. The array tasks run
        # under `#SBATCH --chdir=src/` while this launcher is invoked from wherever
        # you happen to be, so a relative path means one thing here and another on
        # the compute node.
        return ProducerArgs(
            cache_root=absolute_path(self.cache_root),
            schedules_parquet=absolute_path(self.schedules_parquet),
            eval_dir=absolute_path(self.eval_dir),
            num_reps=self.num_reps,
            seed=self.seed,
            batch_size=self.batch_size,
        )


def build_stage_jobs(conf: TransferSlurmConfig) -> dict[str, list]:
    """The surviving jobs for each requested stage — see ``transfer_launch.plan_jobs``."""
    args = conf.producer_args
    return plan_jobs(
        conf.stages,
        expand_targets(conf.target_datasets, conf.target_eps, conf.target_T, conf.target_delta),
        args,
        condition_grid(Path(args.eval_dir) / "category_map.json") if args.eval_dir else set(),
    )


# ---------------------------------------------------------------------------
# Submission (integration glue; exercised via --dry-run, not unit-tested)
# ---------------------------------------------------------------------------


def _write_manifest(conf: TransferSlurmConfig, stage: str, jobs: list) -> str:
    """Write the stage's manifest and return its path (a dry run only computes it).

    A dry run must not touch the filesystem — it prints every line it would have
    written — but it still resolves the path so the printed sbatch script is the
    one that would really be submitted.
    """
    path = Path(conf.manifest_dir) / f"transfer-{stage}.txt"
    if not conf.dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(manifest_text(jobs))
    return str(path)


def _submit(sbatch: str, label: str, dry_run: bool, account: str = "") -> str:
    """Submit one sbatch script, returning its job id (empty on a dry run).

    `account` goes on the command line as well as in the script's `#SBATCH
    --account`, since only the command line outranks a stray SBATCH_ACCOUNT in
    the environment. See _slurm_account.
    """
    print(sbatch)
    if dry_run:
        print(f"[dry-run] would submit {label}\n")
        return ""

    tmpdir = os.path.expandvars(os.path.abspath("/scratch/$USER"))
    os.makedirs(tmpdir, exist_ok=True)
    with NamedTemporaryFile(mode="w", suffix=".sh", dir=tmpdir) as f:
        f.write(sbatch)
        f.flush()
        process_out = subprocess.run(
            ["sbatch", *account_argv(account), f.name],
            capture_output=True,
        )
        stderr = process_out.stderr.decode("utf-8").strip()
        if stderr:
            raise Exception(f"Could not start {label}: {stderr}")
        job_id = process_out.stdout.decode("utf-8").strip().split()[-1]
        print(f"submitted {label} as job {job_id}\n")
        return job_id


def main(conf: TransferSlurmConfig) -> None:
    print("=== planning ===")
    jobs = build_stage_jobs(conf)
    jobs = {stage: stage_jobs for stage, stage_jobs in jobs.items() if stage_jobs}
    if not jobs:
        raise SystemExit("nothing to do: every requested cell already exists")

    estimate = sum(_GPU_HOURS.get(stage, 1.0) * len(js) for stage, js in jobs.items())
    print(f"  estimated {estimate:.0f} GPU-hours total (analytic, +/-3x)")
    if conf.dry_run:
        # The DAG edges are `-d afterok:<job id>`, and no ids exist on a dry run, so
        # the printed scripts carry no #SBATCH --dependency line. Say so rather than
        # let the output read as "this DAG has no ordering constraints".
        print(
            "  [dry-run] dependency lines are omitted below: producers gate on the\n"
            "            preflight and the plot gates on the producer arrays, but the\n"
            "            job ids those reference only exist on a real submission."
        )

    targets = expand_targets(
        conf.target_datasets, conf.target_eps, conf.target_T, conf.target_delta
    )

    print("\n=== preflight ===")
    preflight_id = _submit(
        serial_sbatch(
            name="preflight",
            command=preflight_command(targets, conf.producer_args),
            walltime=_WALLTIMES["preflight"],
            project_dir=conf.project_dir,
            account=conf.account,
            logfile=os.path.join(conf.logdir, "%j", "%x.log"),
            prerequisites=conf.prerequisites,
            cpus_per_task=conf.cpus_per_task,
        ),
        "preflight",
        conf.dry_run,
        conf.account,
    )

    producer_ids = []
    for stage, stage_jobs in jobs.items():
        print(f"\n=== {stage} ({len(stage_jobs)} tasks) ===")
        manifest = _write_manifest(conf, stage, stage_jobs)
        print(f"manifest: {manifest}")
        if conf.dry_run:
            for i, job in enumerate(stage_jobs):
                print(f"  [{i}] {job.args}")
        job_id = _submit(
            array_sbatch(
                stage=stage,
                manifest=manifest,
                n_jobs=len(stage_jobs),
                walltime=_WALLTIMES[stage],
                project_dir=conf.project_dir,
                account=conf.account,
                logfile=os.path.join(conf.logdir, "%A_%a", "%x.log"),
                throttle=conf.throttle,
                prerequisites=(preflight_id,) if preflight_id else (),
                cpus_per_task=conf.cpus_per_task,
                gpus=conf.gpus,
                mem_per_gpu=conf.mem_per_gpu,
            ),
            stage,
            conf.dry_run,
            conf.account,
        )
        if job_id:
            producer_ids.append(job_id)

    if conf.with_plot:
        print("\n=== plot ===")
        _submit(
            serial_sbatch(
                name="plot",
                command=(
                    "uv run --no-sync transfer_plot.py"
                    f" --cache_root {conf.producer_args.cache_root}"
                ),
                walltime=_WALLTIMES["plot"],
                project_dir=conf.project_dir,
                account=conf.account,
                logfile=os.path.join(conf.logdir, "%j", "%x.log"),
                prerequisites=tuple(producer_ids),
                cpus_per_task=conf.cpus_per_task,
            ),
            "plot",
            conf.dry_run,
            conf.account,
        )


if __name__ == "__main__":
    main(tyro.cli(TransferSlurmConfig))
