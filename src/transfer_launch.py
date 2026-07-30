"""Pure core of the transfer-evaluation SLURM launcher (ADR 0008).

``cc/slurm/transfer-run-starter.py`` submits the whole transfer DAG as one job
array per stage. This module holds the parts of that with behaviour worth
testing — the target cross-product, source-regime and condition enumeration,
on-grid validation, the already-done skip filter, and manifest/sbatch rendering —
so the launcher itself is a thin CLI around them.

Jax-free and importable from ``src/`` (the same trick as ``sr_identity.py``): the
launcher runs off-cluster and must not drag in jax to compute a filename or read a
parquet. ``util/transfer.py`` imports :func:`cell_filename` from here, and
``transfer_equation`` imports :func:`condition_source_id`, so the skip filter and
the producers' writes can never disagree about where a cell lives.
"""

import dataclasses
import pathlib
import shlex

from util.py_launcher import emitted_launcher, job_prologue

# ---------------------------------------------------------------------------
# Targets and cell identity
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, order=True)
class Target:
    """A transfer target dataset and the privacy budget it is evaluated under.

    The launcher-side mirror of ``util.transfer.TargetSpec``, minus the arch (which
    ADR 0007 derives from the dataset) — kept separate so this module stays
    stdlib-only.
    """

    dataset: str
    eps: float
    T: int
    delta: float = 1e-7


def expand_targets(
    datasets: tuple[str, ...],
    epsilons: tuple[float, ...],
    timesteps: tuple[int, ...],
    delta: float = 1e-7,
) -> list[Target]:
    """The full ``datasets × eps × T`` cross-product of targets to evaluate.

    Sorted, so the manifest line order — which *is* the SLURM array task index —
    is a deterministic function of the requested inputs and survives a relaunch.
    """
    return sorted(
        Target(dataset=d, eps=float(e), T=int(t), delta=delta)
        for d in datasets
        for e in epsilons
        for t in timesteps
    )


def cell_filename(source_id: str, target: Target) -> str:
    """The parquet filename one source×target cell is written to.

    The single definition of a cell's on-disk name, shared with
    ``util.transfer.write_transfer_cell``. The launcher's skip filter tests for
    exactly this file, so a divergence here would silently re-run finished cells
    (or, worse, skip unfinished ones).
    """
    return f"{source_id}__{target.dataset}__eps{target.eps:g}_T{int(target.T)}.parquet"


# ---------------------------------------------------------------------------
# What there is to transfer: source regimes (curve) and conditions (equation)
# ---------------------------------------------------------------------------

# The schedules.parquet columns that identify a source regime (CONTEXT.md).
_REGIME_COLUMNS = ["dataset", "eps", "T", "arch_label"]


@dataclasses.dataclass(frozen=True, order=True)
class SourceRegime:
    """One source regime and the seed-policies learned under it.

    The unit of a curve-transfer job: ADR 0008 transfers *every* policy in the
    regime (read off, not selected) and reports their spread as that regime's
    generalization consistency, so they belong to the same job.
    """

    dataset: str
    eps: float
    T: int
    arch: str
    run_ids: tuple[str, ...]


def source_regimes(schedules_parquet) -> list[SourceRegime]:
    """Group ``schedules.parquet``'s runs into the source regimes to transfer.

    Sorted by regime, with each regime's ``run_ids`` sorted, so the curve
    manifest's line order is stable across relaunches.
    """
    import pandas as pd

    df = pd.read_parquet(schedules_parquet, columns=[*_REGIME_COLUMNS, "run_id"])
    keyed = df.drop_duplicates(subset=["run_id"]).groupby(_REGIME_COLUMNS, sort=True)
    return sorted(
        SourceRegime(
            dataset=str(dataset),
            eps=float(eps),
            T=int(T),
            arch=str(arch),
            run_ids=tuple(sorted(str(r) for r in group["run_id"])),
        )
        for (dataset, eps, T, arch), group in keyed
    )


def check_on_grid(targets: list[Target], grid: set, stage: str) -> list[Target]:
    """Validate the requested targets against the trained condition ``(eps, T)`` grid.

    Asymmetric by design (ADR 0008). For the **equation** stage an off-grid target
    is fatal: the template's per-condition constants are indexed by discrete
    condition and are not a function of eps/T, so the closed form is simply
    undefined there and the job would abort on the compute node instead. For
    **curve** and **reference** an off-grid target is the experiment — resampling a
    source curve onto an unseen (eps, T) is the transfer claim — so it is kept, with
    a warning that no equation column will exist beside it.
    """
    off_grid = [t for t in targets if (t.eps, t.T) not in grid]
    if not off_grid:
        return targets
    described = ", ".join(f"(eps={t.eps:g}, T={t.T})" for t in off_grid)
    if stage == "equation":
        raise SystemExit(
            f"equation transfer is on-grid only, but no trained condition exists at "
            f"{described}; drop these from --target_eps/--target_T or distil them first"
        )
    print(
        f"  [warning] {stage}: {described} are off the trained (eps, T) grid — "
        "no equation column will exist beside these targets"
    )
    return targets


def condition_source_id(category: int, condition: dict) -> str:
    """The ``source_id`` an equation-transferred condition is recorded under.

    A distilled condition has no per-seed identity (ADR 0015), so its provenance IS
    the condition ``(dataset, eps, T, arch)`` plus its category index. Shared with
    ``transfer_equation.equation_source`` and fed to :func:`cell_filename`, so the
    launcher's skip filter looks for the file the producer actually writes.
    """
    dataset, arch = condition["dataset"], condition["arch_label"]
    return f"{dataset}_eps{condition['eps']:g}_T{int(condition['T'])}_{arch}_cat{category}"


def conditions_at(category_map, eps: float, T: int) -> list[tuple[int, dict]]:
    """The ``(category, condition)`` pairs trained at exactly ``(eps, T)``.

    Every condition at that budget is transferred — read off, not selected (ADR
    0008) — so one equation job covers all of them. ``category`` is the condition's
    1-indexed position in the map, which is how the distilled template addresses
    its per-condition constants.
    """
    return [
        (i + 1, condition)
        for i, condition in enumerate(category_map)
        if float(condition["eps"]) == float(eps) and int(condition["T"]) == int(T)
    ]


def condition_grid(category_map_path) -> set:
    """The distinct ``(eps, T)`` budgets the SR run has trained conditions for.

    Several conditions can share a budget (they differ in dataset or arch), so this
    is a grid over *budgets*: exactly what an equation target must land on.
    """
    from sr_category import load_category_map

    category_map = load_category_map(category_map_path)
    return {(float(c["eps"]), int(c["T"])) for c in category_map}


# ---------------------------------------------------------------------------
# Jobs: one array task each, plus the skip filter and the manifest
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ProducerArgs:
    """The launch-wide arguments every producer invocation carries.

    One sweep's inputs (``schedules_parquet``, ``eval_dir``) and one evaluation
    protocol (``num_reps``, ``seed``, ``batch_size``), applied uniformly across the
    three stages so the matrix's cells are comparable.
    """

    cache_root: str
    schedules_parquet: str = ""
    eval_dir: str = ""
    num_reps: int = 8
    seed: int = 0
    batch_size: int = 250

    def shared_flags(self) -> str:
        return (
            f"--cache_root {shlex.quote(self.cache_root)}"
            f" --num_reps {self.num_reps}"
            f" --seed {self.seed}"
            f" --batch_size {self.batch_size}"
        )


@dataclasses.dataclass(frozen=True)
class Job:
    """One SLURM array task: a producer invocation and the cells it will write.

    ``args`` is the literal manifest line the array task runs; ``cells`` are the
    cell parquets it is responsible for, relative to ``cache_root``, which is what
    the skip filter tests for.
    """

    stage: str
    args: str
    cells: tuple[str, ...]


def _target_flags(target: Target) -> str:
    return (
        f"--target {shlex.quote(target.dataset)}"
        f" --target_eps {target.eps:g}"
        f" --target_T {target.T}"
        f" --target_delta {target.delta:g}"
    )


def _cells(producer: str, source_ids, target: Target) -> tuple[str, ...]:
    return tuple(f"transfer/{producer}/{cell_filename(s, target)}" for s in source_ids)


def curve_jobs(regimes: list[SourceRegime], targets: list[Target], args: ProducerArgs) -> list[Job]:
    """One curve job per source regime × target.

    The regime is passed as a *filter* rather than a run-id list: the producer
    re-reads ``schedules.parquet`` and transfers every policy matching the regime,
    so the job's identity survives the parquet gaining more seeds.
    """
    return [
        Job(
            stage="curve",
            args=(
                f"{emitted_launcher()} transfer_curve.py"
                f" --schedules_parquet {shlex.quote(args.schedules_parquet)}"
                f" --source_dataset {shlex.quote(regime.dataset)}"
                f" --source_eps {regime.eps:g}"
                f" --source_T {regime.T}"
                f" --source_arch {shlex.quote(regime.arch)}"
                f" {_target_flags(target)}"
                f" {args.shared_flags()}"
            ),
            cells=_cells("curve", regime.run_ids, target),
        )
        for regime in regimes
        for target in targets
    ]


def equation_jobs(category_map, targets: list[Target], args: ProducerArgs) -> list[Job]:
    """One equation job per target budget, covering every condition trained there.

    ADR 0008 transfers every condition at the target's exact ``(eps, T)`` — read
    off, not selected — and the producer already loops them internally, so one
    invocation per target is the natural unit. A target with no matching condition
    yields no job at all: off-grid is fatal at validation time, and an array task
    that provably cannot write a cell should never be submitted.
    """
    jobs = []
    for target in targets:
        conditions = conditions_at(category_map, target.eps, target.T)
        if not conditions:
            continue
        source_ids = [condition_source_id(cat, cond) for cat, cond in conditions]
        jobs.append(
            Job(
                stage="equation",
                args=(
                    f"{emitted_launcher()} transfer_equation.py"
                    f" --eval_dir {shlex.quote(args.eval_dir)}"
                    f" {_target_flags(target)}"
                    f" {args.shared_flags()}"
                ),
                cells=_cells("equation", source_ids, target),
            )
        )
    return jobs


def reference_jobs(
    references: tuple[str, ...], targets: list[Target], args: ProducerArgs
) -> list[Job]:
    """One reference job per native reference × target.

    The three references are swept independently rather than in one invocation so
    they fan out across the cluster: a reference sweep is the longest-running stage,
    and sharing a job would serialise three of them behind one wall clock.
    """
    return [
        Job(
            stage="reference",
            args=(
                f"{emitted_launcher()} transfer_reference.py"
                f" --reference {shlex.quote(reference)}"
                f" {_target_flags(target)}"
                f" {args.shared_flags()}"
            ),
            cells=_cells("reference", [reference], target),
        )
        for reference in references
        for target in targets
    ]


def drop_finished(jobs: list[Job], cache_root) -> list[Job]:
    """Drop jobs whose cells are already on disk.

    The skip is job-level and applied at manifest-build time, so a relaunch after
    a partial run submits a smaller array rather than a full one that no-ops. There
    is deliberately **no** producer-side resumption: a producer re-runs its whole
    unit, so a job with even one missing cell is kept whole. Cell writes are atomic
    (``util.transfer.write_transfer_cell``), so a present file is a finished one.
    """
    root = pathlib.Path(cache_root)
    return [job for job in jobs if not all((root / cell).exists() for cell in job.cells)]


def manifest_text(jobs: list[Job]) -> str:
    """Render the jobs as an array manifest — one command per line.

    Line ``i`` is what array task ``i`` runs, so the file *is* the task index. The
    trailing newline keeps the final line readable by line-addressed shell tools.
    """
    return "".join(f"{job.args}\n" for job in jobs)


# ---------------------------------------------------------------------------
# sbatch rendering
# ---------------------------------------------------------------------------


def _dependency_line(prerequisites: tuple[str, ...]) -> str:
    """The ``#SBATCH --dependency`` line, or nothing when unconstrained.

    ``afterok`` only — never ``singleton``. ``singleton`` is ``run-starter.py``'s
    chain-serialisation device (one job per name at a time); on a job array it
    would serialise the whole stage and destroy the fan-out this launcher exists
    for. The transfer DAG's only real ordering constraint is the preflight.
    """
    if not prerequisites:
        return ""
    return "#SBATCH --dependency=afterok:" + ",".join(prerequisites) + "\n"


def array_sbatch(
    stage: str,
    manifest: str,
    n_jobs: int,
    walltime: str,
    project_dir: str,
    account: str,
    logfile: str,
    throttle: int = 8,
    prerequisites: tuple[str, ...] = (),
    cpus_per_task: int = 2,
    gpus: int = 1,
    mem_per_gpu: str = "12G",
) -> str:
    """The sbatch script submitting one stage as a job array over its manifest.

    Task ``i`` runs manifest line ``i+1`` verbatim, so the manifest is the single
    record of what was launched and the array index means nothing beyond "which
    line". ``%<throttle>`` caps concurrent tasks so a large cross-product does not
    flood the allocation.
    """
    return f"""#!/bin/bash
#SBATCH --array=0-{n_jobs - 1}%{throttle}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gpus={gpus}
#SBATCH --gpus-per-node={gpus}
#SBATCH --mem-per-gpu={mem_per_gpu}
#SBATCH --time={walltime}
#SBATCH --output={logfile}
#SBATCH --job-name=transfer-{stage}
#SBATCH --chdir={project_dir}
#SBATCH --account={account}
{_dependency_line(prerequisites)}
echo "Current working directory: `pwd`"
echo "Starting transfer '{stage}' task $SLURM_ARRAY_TASK_ID at: `date`"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# Python environment setup; must precede the manifest line, whose launcher may be a
# deferred "$PY_LAUNCHER" this resolves. See src/util/py_launcher.py.
{job_prologue()}

# The manifest line IS the task: line (index+1) of the file the launcher wrote.
CMD=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {manifest})
if [ -z "$CMD" ]; then
    echo "no manifest line for task $SLURM_ARRAY_TASK_ID in {manifest}" >&2
    exit 1
fi
echo "cmd: $CMD"

time eval "$CMD"

echo "Job finished with exit code $? at: `date`"
""".strip()


def serial_sbatch(
    name: str,
    command: str,
    walltime: str,
    project_dir: str,
    account: str,
    logfile: str,
    prerequisites: tuple[str, ...] = (),
    cpus_per_task: int = 2,
    mem_per_cpu: str = "8G",
) -> str:
    """The sbatch script for a single CPU job — the preflight or the plot assembler.

    Neither needs a GPU, and the preflight must be *one* sequential job:
    ``util/dataloaders.py`` downloads into the repo with no locking and no
    temp-rename, so letting the producer array first-touch a dataset concurrently
    would race a half-written ``.npy`` cache. Warming every target dataset once,
    up front, is the entire reason this stage exists.
    """
    return f"""#!/bin/bash
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --time={walltime}
#SBATCH --output={logfile}
#SBATCH --job-name=transfer-{name}
#SBATCH --chdir={project_dir}
#SBATCH --account={account}
{_dependency_line(prerequisites)}
echo "Current working directory: `pwd`"
echo "Starting transfer '{name}' at: `date`"

{job_prologue()}

time {command}

echo "Job finished with exit code $? at: `date`"
""".strip()


def preflight_command(targets: list[Target], args: ProducerArgs) -> str:
    """The command the sequential preflight job runs.

    Warms the on-disk cache for every *distinct* target dataset. Deduplication is
    the point: a dataset recurs at every (eps, T) of the cross-product, and
    ``util/dataloaders.py`` downloads into the repo with no locking and no
    temp-rename, so two concurrent first-touches of the same dataset would race a
    half-written ``.npy``.
    """
    datasets = sorted({t.dataset for t in targets})
    return (
        f"{emitted_launcher()} transfer_preflight.py"
        f" --datasets {' '.join(shlex.quote(d) for d in datasets)}"
        f" --batch_size {args.batch_size}"
    )
