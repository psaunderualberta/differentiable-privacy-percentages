"""Launch the whole transfer-evaluation DAG on SLURM (ADR 0008, ADR 0015).

One entry point submits the whole DAG:

    preflight (1 sequential CPU job)
        └─ curve               (job array, 1 task per source policy × target)
        └─ equation            (job array, 1 task per condition × target)
        └─ reference-candidate (job array, 1 task per reference × target × candidate)
              └─ reference     (job array, 1 selector task per reference × target)
        └─ plot (1 CPU job, only with --with-plot, after every producer array)

The reference stage is the only two-phase producer (ADR 0019): its 20-candidate
random search is split one task per candidate, and a selector then picks the winner
and runs the final evaluation. That edge is the one `afterok` between producers.

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
        --target_datasets chexpert imagenet \\
        --target_eps 10.0 --target_T 2000 5000 7000 --dry-run

    # Submit, including the assembler:
    uv run cc/slurm/transfer-run-starter.py ... --with-plot
"""

import os
import shlex
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
    STAGE_PREREQUISITE_STAGE,
    Job,
    ProducerArgs,
    SourceScope,
    absolute_path,
    array_sbatch,
    condition_grid,
    expand_targets,
    manifest_text,
    plan_jobs,
    preflight_command,
    serial_sbatch,
)

# Per-stage wall clocks, overridable with `--walltimes <stage> <clock>`. Every
# producer stage now fits the <=3h queue, where scheduling priority is best: ADR 0019
# split the reference sweep into per-candidate tasks (it was 11:55 and could not
# finish), and ADR 0018/item 6 split curve and equation tasks down to one cell each.
# The curve clock is measured, not guessed: the slowest probe task (eyepacs, the
# largest input) took 1:14:11, so 2:55 leaves 2.4x for node-to-node jitter. Keep
# that headroom — a curve task has no resumption, so an overrun writes nothing.
# ADR 0020 dropped eyepacs, so the worst remaining case is imagenet at T=7000, an
# order of magnitude under this clock. Left as-is: the clock costs nothing but queue
# priority, and it is the one number that must never be tight.
_WALLTIMES = {
    "preflight": "00-02:55:00",
    "curve": "00-02:55:00",
    "equation": "00-02:55:00",
    "reference-candidate": "00-02:55:00",
    "reference": "00-02:55:00",
    "plot": "00-02:55:00",
}

# Per-task GPU-hours, used only for the --dry-run estimate.
# Per *task*, and every stage is now one unit of work per task: one source policy
# (curve), one condition (equation), one scored sweep candidate at
# SWEEP_SCORING_ITERATIONS inner trainings (reference-candidate), or one final
# evaluation at num_reps (reference). Keep these honest against the wall clocks in
# _WALLTIMES — an estimate below the clock is what hides a stage whose tasks cannot
# finish in time.
#
# The curve figures are measured (probe wave 2026-08-04, jobs 345104-345106: one
# cell each at eps=10, T=5000, num_reps=3). A single scalar cannot stand in for the
# stage: cost tracks the target's input resolution, and eyepacs (256x256) is 46x
# chexpert (64x64) at identical eps and T. So the key is the dataset, and the two
# budget axes are handled separately:
#
#   - eps is dropped: a step costs the same whatever sigma it uses.
#   - T is NOT dropped. It was, while every planned target shared T=5000; ADR 0020's
#     grid spans T=2000..7000, and the inner loop is a scan of exactly T steps, so
#     cost is linear in T. The figures below are per-task at _MEASURED_T and are
#     rescaled by T/_MEASURED_T in `_job_gpu_hours`.
#
# eyepacs is retained here although ADR 0020 dropped it as a column: it is real
# measured data and it is the fallback, so an unknown dataset over-estimates rather
# than under-.
# The other three stages remain analytic (±3×) until they are measured the same way.
_MEASURED_T = 5000
_CURVE_GPU_HOURS = {"eyepacs": 1.24, "imagenet": 0.06, "chexpert": 0.03}
_GPU_HOURS = {
    "curve": max(_CURVE_GPU_HOURS.values()),
    "equation": 1.4,
    "reference-candidate": 1.3,
    "reference": 0.5,
}


def _job_gpu_hours(stage: str, job: Job) -> float:
    """Estimated GPU-hours for one array task.

    Curve tasks are keyed by target dataset (see ``_CURVE_GPU_HOURS``) and scaled
    linearly by ``--target_T``, since the inner loop is a scan of exactly T steps;
    every other stage is flat per task. Both are read back off the manifest line
    rather than threaded through ``Job``, which stays a stdlib-only record of what
    to run.
    """
    if stage != "curve":
        return _GPU_HOURS.get(stage, 1.0)
    args = shlex.split(job.args)
    per_task = _GPU_HOURS["curve"]
    if "--target" in args:
        per_task = _CURVE_GPU_HOURS.get(args[args.index("--target") + 1], per_task)
    if "--target_T" in args:
        per_task *= int(args[args.index("--target_T") + 1]) / _MEASURED_T
    return per_task


@dataclass
class TransferSlurmConfig:
    """Submit the transfer-evaluation DAG for one target cross-product."""

    cache_root: str
    """Where cells are written and the skip filter looks. Required, no default."""

    target_datasets: tuple[str, ...] = ("chexpert", "imagenet")
    """Target columns, in pipeline-validation order (ADR 0020: CheXpert first, then
    ImageNet-32). EyePACS was dropped for having no schedule-resolving power."""
    target_eps: tuple[float, ...] = (10.0,)
    target_T: tuple[int, ...] = (2000, 5000, 7000)
    """The grid is a T-spread at fixed eps: eps is nearly inert across the source
    sweep's 3.3x span, while T is where schedule shape lives. Every value here must
    be ON the source condition grid (eps in {3,5,8,10}, T in {2000,3000,5000,7000}) —
    ``check_on_grid`` makes an off-grid target fatal for the equation stage."""
    target_delta: float = 1e-7

    schedules_parquet: str = ""
    """Source schedules.parquet from compile_results_fetch. Required for the curve stage."""
    eval_dir: str = ""
    """SR eval dir (equations.csv + category_map.json). Required for the equation stage."""

    source_arch: str = SourceScope.arch
    """Curve sources are scoped to this arch axis (ADR 0018); empty keeps every arch."""
    source_min_seeds: int = SourceScope.min_seeds
    """Drop a source regime-arm carrying fewer seeds than this; 0 disables the floor."""
    source_max_seeds: int = SourceScope.max_seeds
    """Transfer at most this many seeds per regime-arm, lowest index first; 0 disables."""

    stages: tuple[str, ...] = ("curve", "equation", "reference")
    """Producer stages to submit."""
    with_plot: bool = False
    """Also submit the assembler, gated behind every producer array."""

    num_reps: int = 3
    """Evaluation seeds per cell, uniform across all three stages.

    3, not 8 (ADR 0018): a cell's ± is now the spread across its regime's four source
    policies, so the reps only have to stabilise a cell mean."""
    seed: int = 0
    batch_size: int = 250
    throttle: int = 8
    """Max concurrently running tasks per array."""
    walltimes: dict[str, str] = field(default_factory=dict)
    """Per-stage wall-clock overrides, e.g. `--walltimes curve 00-01:30:00`."""

    account: str = "aip-nidhih"
    cpus_per_task: int = 2
    gpus: int = 1
    mem_per_gpu: str = "20G"
    """Host RAM per GPU. The 2026-08-04 probe measured eyepacs at 11.33 GiB peak RSS,
    94% of the 12G this used to default to — close enough that a slightly heavier
    policy or a different node would OOM, and an OOM loses a cell exactly like a
    timeout does. 20G restores headroom on the one target that needs it.

    ADR 0020 dropped eyepacs, so this is now sized for a target that no longer runs
    and the remaining two are far lighter. Kept anyway, and deliberately: no probe
    has measured chexpert or imagenet peak RSS, over-requesting costs only queue
    priority, and under-requesting costs a cell. Re-tighten only against a measurement."""
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

    def walltime(self, stage: str) -> str:
        """This launch's wall clock for ``stage`` — the default unless overridden.

        A flag rather than a constant because the clocks are calibrated against
        measured per-task cost, and a stage whose estimate moves (a longer target T,
        a slower node) needs its clock adjusted at launch, not in a source edit.
        """
        unknown = set(self.walltimes) - set(_WALLTIMES)
        if unknown:
            raise SystemExit(
                f"unknown stage(s) in --walltimes: {sorted(unknown)}; "
                f"expected some of {sorted(_WALLTIMES)}"
            )
        return self.walltimes.get(stage, _WALLTIMES[stage])


def build_stage_jobs(conf: TransferSlurmConfig) -> dict[str, list]:
    """The surviving jobs for each requested stage — see ``transfer_launch.plan_jobs``."""
    args = conf.producer_args
    return plan_jobs(
        conf.stages,
        expand_targets(conf.target_datasets, conf.target_eps, conf.target_T, conf.target_delta),
        args,
        condition_grid(Path(args.eval_dir) / "category_map.json") if args.eval_dir else set(),
        SourceScope(
            arch=conf.source_arch,
            min_seeds=conf.source_min_seeds,
            max_seeds=conf.source_max_seeds,
        ),
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

    estimate = sum(_job_gpu_hours(stage, j) for stage, js in jobs.items() for j in js)
    print(f"  estimated {estimate:.0f} GPU-hours total (curve measured, rest analytic)")
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
            walltime=conf.walltime("preflight"),
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
    stage_ids: dict[str, str] = {}
    for stage, stage_jobs in jobs.items():
        print(f"\n=== {stage} ({len(stage_jobs)} tasks) ===")
        manifest = _write_manifest(conf, stage, stage_jobs)
        print(f"manifest: {manifest}")
        if conf.dry_run:
            for i, job in enumerate(stage_jobs):
                print(f"  [{i}] {job.args}")

        # The reference selector must not start until every candidate has scored, or
        # it picks a winner from a partial score set (ADR 0019). plan_jobs emits the
        # phases in dependency order, so the id is already in hand.
        prerequisites = [preflight_id] if preflight_id else []
        upstream = stage_ids.get(STAGE_PREREQUISITE_STAGE.get(stage, ""), "")
        if upstream:
            prerequisites.append(upstream)

        job_id = _submit(
            array_sbatch(
                stage=stage,
                manifest=manifest,
                n_jobs=len(stage_jobs),
                walltime=conf.walltime(stage),
                project_dir=conf.project_dir,
                account=conf.account,
                logfile=os.path.join(conf.logdir, "%A_%a", "%x.log"),
                throttle=conf.throttle,
                prerequisites=tuple(prerequisites),
                cpus_per_task=conf.cpus_per_task,
                gpus=conf.gpus,
                mem_per_gpu=conf.mem_per_gpu,
            ),
            stage,
            conf.dry_run,
            conf.account,
        )
        if job_id:
            stage_ids[stage] = job_id
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
                walltime=conf.walltime("plot"),
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
