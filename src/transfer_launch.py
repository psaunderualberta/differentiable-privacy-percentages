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
import os
import pathlib
import shlex

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


# Producers whose ``source_id`` does NOT determine the arm, so their cell names must
# carry it (ADR 0021). A curve cell's source_id is a W&B run id, learned in exactly one
# arm, so the arm is redundant in its name — and adding it would rename every finished
# cell for no information. The other two share a source_id across the arms:
#   * a reference's is a bare mechanism name (`Constant`/`Dynamic-DPSGD`/`Median`) that
#     both target momenta run;
#   * an equation's is a condition slug, and a *condition* is (dataset, eps, T, arch) with
#     no arm in it — ADR 0016 scopes the arm to the synthesis, not the condition, so the
#     two arm-scoped fits distil the same conditions under the same category indices.
# Without the arm each pair collides on one file, and the skip filter — finding it —
# silently never runs the second arm.
#
# Read by BOTH the launcher's skip filter and ``util.transfer.write_transfer_cell``,
# so the two cannot disagree about which producers get an arm segment.
ARM_IN_CELL_NAME = frozenset({"reference", "equation"})

TARGET_ARMS = ("sgd-m0.0", "sgd-m0.9")
"""The target inner-momentum arms the reference stage is replicated across (ADR 0021).

Curve and equation jobs do *not* consult this: their arm is a property of the source
they transfer (its ``optimizer`` column / its synthesis) and the target inherits it, so
their manifest lines are unchanged. An equation launch covers *one* arm, the one its
``--eval_dir`` synthesis was fitted over (:func:`synthesis_arm`); both arms means two
launches, one per eval dir. Only the references, having no source to inherit from, have
to be told.
"""


def synthesis_arm(eval_dir) -> str:
    """The momentum arm the synthesis under ``eval_dir`` was scoped to (ADR 0016).

    Read off the run's ``manifest.json`` (``config.optimizers``, the ADR 0011 arm
    filter): exactly one entry means the fit is scoped to that arm, while empty or
    several means it pooled them and ``""`` keeps those cells out of the per-arm
    overlay rather than mislabelling them as one arm's.

    Lives here, stdlib-only, rather than in ``transfer_equation``: the launcher needs
    it to predict the cell name its skip filter tests for, and it must be the same
    string the producer writes — so there is one reader of the manifest, re-exported
    by ``transfer_equation``.
    """
    import json

    manifest_path = pathlib.Path(eval_dir) / "manifest.json"
    if not manifest_path.is_file():
        return ""
    optimizers = json.loads(manifest_path.read_text()).get("config", {}).get("optimizers", [])
    return str(optimizers[0]) if len(optimizers) == 1 else ""


def cell_filename(source_id: str, target: Target, arm: str = "") -> str:
    """The parquet filename one source×target cell is written to.

    The single definition of a cell's on-disk name, shared with
    ``util.transfer.write_transfer_cell``. The launcher's skip filter tests for
    exactly this file, so a divergence here would silently re-run finished cells
    (or, worse, skip unfinished ones).

    ``arm`` is appended when non-empty. Callers pass it iff the producer is in
    :data:`ARM_IN_CELL_NAME`; see that constant for why the default is to omit it.
    """
    suffix = f"__{arm}" if arm else ""
    return f"{source_id}__{target.dataset}__eps{target.eps:g}_T{int(target.T)}{suffix}.parquet"


# Where a reference sweep's per-candidate scores live. Deliberately a *sibling* of
# `transfer/`, not a subdirectory of it: `transfer_plot.load_producers` treats every
# subdirectory of `transfer/` as a producer, so a candidate directory nested there
# would surface 19 under-evaluated schedules as extra matrix rows (ADR 0019).
CANDIDATE_DIR = "transfer_candidates"

NUM_SWEEP_CANDIDATES = 20
"""Candidates in a reference's random search. Mirrors ``util.baselines`` so the
launcher can size the candidate array without importing jax (see this module's
docstring); the producer validates the index it is handed against the real sweep."""

# The reference stage is the DAG's only two-phase producer: its selector cannot run
# until every candidate has scored, or it would pick a winner from a partial — or
# empty — score set. Consulted by both launchers when wiring `afterok`.
STAGE_PREREQUISITE_STAGE = {"reference": "reference-candidate"}


def candidate_filename(reference: str, target: Target, candidate: int, arm: str = "") -> str:
    """The file one (reference × target × arm × candidate) score is written to.

    The launcher's skip filter tests for exactly this path, which is what gives the
    reference stage its second resumption granularity: a finished candidate is
    skipped independently of its selector.

    The arm segment sits *before* ``__cand`` so that stripping the candidate index
    still leaves a prefix identifying one arm's sweep — which is how
    ``util.transfer.read_candidate_records`` collects the pool the selector chooses
    from. Put it after, and the selector would pool both arms' scores into one sweep
    (ADR 0021).
    """
    suffix = f"__{arm}" if arm else ""
    return (
        f"{reference}__{target.dataset}__eps{target.eps:g}_T{int(target.T)}{suffix}"
        f"__cand{int(candidate):02d}.json"
    )


# ---------------------------------------------------------------------------
# What there is to transfer: source regimes (curve) and conditions (equation)
# ---------------------------------------------------------------------------

# The schedules.parquet columns that identify a source regime (CONTEXT.md). ADR 0018
# adds `optimizer`: the arm is part of a source regime's identity, and omitting it made
# a "16-seed regime" eight sgd-m0.9 runs pooled with eight sgd-m0.0 ones.
_REGIME_COLUMNS = ["dataset", "eps", "T", "arch_label", "optimizer"]


@dataclasses.dataclass(frozen=True, order=True)
class SourceRegime:
    """One source regime-arm and the seed-policies learned under it.

    The unit of *analysis* for curve transfer: ADR 0008 transfers every policy in
    the regime (read off, not selected) and reports their spread as that regime's
    generalization consistency. Scheduling is per policy (see :func:`curve_jobs`).
    """

    dataset: str
    eps: float
    T: int
    arch: str
    arm: str
    run_ids: tuple[str, ...]
    seeds: tuple[int, ...] = ()
    """The seed index of each entry of ``run_ids``, positionally aligned.

    Carried so :func:`scope_regimes` can apply ADR 0018's seed floor and cap on the
    *seed index* rather than on run-id order, which is a W&B artefact.
    """


def source_regimes(schedules_parquet) -> list[SourceRegime]:
    """Group ``schedules.parquet``'s runs into the source regime-arms to transfer.

    Sorted by regime, with each regime's ``run_ids`` sorted, so the curve
    manifest's line order is stable across relaunches.
    """
    import pandas as pd

    df = pd.read_parquet(schedules_parquet, columns=[*_REGIME_COLUMNS, "run_id", "seed"])
    keyed = df.drop_duplicates(subset=["run_id"]).groupby(_REGIME_COLUMNS, sort=True)
    regimes = []
    for (dataset, eps, T, arch, optimizer), group in keyed:
        ordered = sorted((str(r), int(s)) for r, s in zip(group["run_id"], group["seed"]))
        regimes.append(
            SourceRegime(
                dataset=str(dataset),
                eps=float(eps),
                T=int(T),
                arch=str(arch),
                arm=str(optimizer),
                run_ids=tuple(run_id for run_id, _ in ordered),
                seeds=tuple(seed for _, seed in ordered),
            )
        )
    return sorted(regimes)


@dataclasses.dataclass(frozen=True)
class SourceScope:
    """How much of the source sweep curve transfer runs (ADR 0018).

    Defaults are FirSweep's: the T-sweep axis, both arms, four seeds per regime-arm.
    """

    arch: str = "cnn-16x32-head32"
    """Keep only this arch axis; empty keeps every arch."""
    min_seeds: int = 4
    """Drop a regime-arm carrying fewer seeds than this; 0 disables the floor."""
    max_seeds: int = 4
    """Keep at most this many seeds per regime-arm, lowest index first; 0 disables."""


def scope_regimes(
    regimes: list[SourceRegime],
    arch: str = "",
    min_seeds: int = 0,
    max_seeds: int = 0,
) -> list[SourceRegime]:
    """Narrow the source regime-arms curve transfer will run (ADR 0018).

    ADR 0008 requires the matrix be *read off*, never selected — and every filter
    here is independent of any accuracy number, so it subsamples the pool without
    biasing it:

    * ``arch`` keeps only the scoped axis. The arch axis exists at a single (ε, T)
      point and is out of scope for every synthesis (ADR 0016), so its policies
      could never gain an equation counterpart.
    * ``min_seeds`` is a floor applied **within an arm**. A regime-arm below it is
      dropped whole rather than admitted with a smaller n: its ± would render
      identically to a full one in the same heatmap while being a range rather than
      a consistency measure.
    * ``max_seeds`` caps the survivors at their **lowest seed indices**. The seed
      index is assigned before training and is independent of every accuracy
      number, so this is a subsample, not the best-of-regime selection ADR 0008
      prohibits.

    A zero/empty argument disables that filter.
    """
    scoped = []
    for regime in regimes:
        if arch and regime.arch != arch:
            continue
        if len(regime.run_ids) < min_seeds:
            continue
        kept = sorted(zip(regime.seeds, regime.run_ids))
        if max_seeds:
            kept = kept[:max_seeds]
        scoped.append(
            dataclasses.replace(
                regime,
                run_ids=tuple(run_id for _, run_id in kept),
                seeds=tuple(seed for seed, _ in kept),
            )
        )
    return scoped


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


def absolute_path(path: str) -> str:
    """Resolve a user-supplied path against the invoking cwd, keeping ``""`` as ``""``.

    Load-bearing for both launchers. A producer runs with its cwd set to ``src/``
    (``#SBATCH --chdir`` on the cluster, ``cwd=`` on the local pool) while the
    launcher is invoked from anywhere — usually the repo root. So a relative
    ``--cache_root`` would have the skip filter testing one directory and the
    producer writing another, and a relative ``--schedules_parquet`` that the
    launcher reads fine resolves on the node to ``src/<what you typed>`` and dies
    there, minutes into a submitted array.

    Empty means "stage not requested" to :func:`plan_jobs`, and ``abspath("")``
    would turn that into a real directory, so it is passed through untouched.
    """
    return os.path.abspath(path) if path else ""


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
    num_reps: int = 3
    """Evaluation reps per cell. 3, not 8 (ADR 0018): a cell's ± is now the spread
    across its regime's four source policies, so the reps only have to stabilise a
    cell mean rather than carry the consistency estimate themselves."""
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


def _cells(producer: str, source_ids, target: Target, arm: str = "") -> tuple[str, ...]:
    named = arm if producer in ARM_IN_CELL_NAME else ""
    return tuple(f"transfer/{producer}/{cell_filename(s, target, named)}" for s in source_ids)


def curve_jobs(regimes: list[SourceRegime], targets: list[Target], args: ProducerArgs) -> list[Job]:
    """One curve job per source *policy* × target.

    The regime stays the unit of *analysis* — ADR 0008 transfers every policy in it
    and the assembler reports their spread as the regime's generalization
    consistency — but it is not the unit of *scheduling*. ``transfer_curve`` walks
    its selected policies serially, so a regime-sized job costs (seeds × one
    evaluation) and blows a fixed wall clock as soon as a regime has more seeds than
    the clock has room for. Per-policy tasks bound a task's cost at one evaluation
    and, since a task then owns exactly one cell, let :func:`drop_finished` resume a
    partial run at cell granularity instead of recomputing a regime's finished
    seeds.

    The cost is that the run ids are resolved at launch time: a policy added to
    ``schedules.parquet`` after submission needs a relaunch to be picked up, where a
    regime filter would have swept it in. Cells are keyed on the run id either way,
    so what lands on disk is unchanged.
    """
    return [
        Job(
            stage="curve",
            args=(
                "uv run --no-sync transfer_curve.py"
                f" --schedules_parquet {shlex.quote(args.schedules_parquet)}"
                f" --source_run_id {shlex.quote(run_id)}"
                f" {_target_flags(target)}"
                f" {args.shared_flags()}"
            ),
            cells=_cells("curve", [run_id], target),
        )
        for regime in regimes
        for run_id in regime.run_ids
        for target in targets
    ]


def equation_jobs(
    category_map, targets: list[Target], args: ProducerArgs, arm: str = ""
) -> list[Job]:
    """One equation job per target × distilled condition.

    ADR 0008 transfers every condition at the target's exact ``(eps, T)`` — read
    off, not selected — but they are not the unit of *scheduling*. The producer
    walks its conditions serially, so a target-sized task costs (conditions × one
    evaluation): at roughly 1.4h a condition, the two conditions of a FirSweep target
    leave eight minutes of margin against a 2:55 wall clock. One condition per task
    bounds the cost at a single evaluation and, since a task then owns exactly one
    cell, lets :func:`drop_finished` resume at cell granularity.

    A target with no matching condition yields no job at all: off-grid is fatal at
    validation time, and an array task that provably cannot write a cell should never
    be submitted.

    ``arm`` is the synthesis's arm, and it goes in the cell name rather than on the
    command line: the producer re-derives it from the same ``--eval_dir`` manifest, so
    passing it would be a second, forgeable copy of one fact.
    """
    return [
        Job(
            stage="equation",
            args=(
                "uv run --no-sync transfer_equation.py"
                f" --eval_dir {shlex.quote(args.eval_dir)}"
                f" --category {category}"
                f" {_target_flags(target)}"
                f" {args.shared_flags()}"
            ),
            cells=_cells("equation", [condition_source_id(category, condition)], target, arm),
        )
        for target in targets
        for category, condition in conditions_at(category_map, target.eps, target.T)
    ]


def candidate_jobs(
    references: tuple[str, ...],
    targets: list[Target],
    args: ProducerArgs,
    num_candidates: int = NUM_SWEEP_CANDIDATES,
    arms: tuple[str, ...] = TARGET_ARMS,
) -> list[Job]:
    """One scoring job per native reference × target × arm × sweep candidate (ADR 0019).

    The reference stage's first phase. A reference's search is 20 candidates and, run
    as one blocking sweep, ~87 GPU-hours per target — against an 11:55 wall, and with
    no producer-side checkpointing, every task would have died at ~13% of its work
    having saved nothing. One task per candidate bounds a task at ~1.3h without
    shrinking the search: total compute is unchanged, only its packaging.

    A candidate task owns its score record, not a transfer cell, so :func:`drop_finished`
    resumes a partial reference stage at candidate granularity.

    The sweep is repeated per ``arm`` (ADR 0021): the candidates are scored at the
    target momentum they will be a baseline for, so an m=0.9-tuned reference is never
    reused as the m=0.0 target's bar.
    """
    return [
        Job(
            stage="reference-candidate",
            args=(
                "uv run --no-sync transfer_reference.py"
                f" --reference {shlex.quote(reference)}"
                f" --candidate {candidate}"
                f" --arm {shlex.quote(arm)}"
                f" {_target_flags(target)}"
                f" {args.shared_flags()}"
            ),
            cells=(f"{CANDIDATE_DIR}/{candidate_filename(reference, target, candidate, arm)}",),
        )
        for reference in references
        for target in targets
        for arm in arms
        for candidate in range(num_candidates)
    ]


def reference_jobs(
    references: tuple[str, ...],
    targets: list[Target],
    args: ProducerArgs,
    arms: tuple[str, ...] = TARGET_ARMS,
) -> list[Job]:
    """One selector job per native reference × target × arm — the stage's second phase.

    Reads every candidate's score, picks the winner and runs the final evaluation.
    This is the only reference job that writes a ``producer="reference"`` cell; the
    candidate records it consumes are intermediate and must never reach the assembler.

    The three references are selected independently rather than in one invocation so
    they fan out across the cluster. Each arm selects from its own score pool, which is
    what makes 3 mechanisms × 6 target regimes × 2 arms = 36 reference cells.
    """
    return [
        Job(
            stage="reference",
            args=(
                "uv run --no-sync transfer_reference.py"
                f" --reference {shlex.quote(reference)}"
                f" --arm {shlex.quote(arm)}"
                f" {_target_flags(target)}"
                f" {args.shared_flags()}"
            ),
            cells=_cells("reference", [reference], target, arm),
        )
        for reference in references
        for target in targets
        for arm in arms
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


REFERENCES = ("Constant", "Dynamic-DPSGD", "Median")


def plan_jobs(
    stages: tuple[str, ...],
    targets: list[Target],
    args: ProducerArgs,
    grid: set = frozenset(),
    scope: SourceScope | None = None,
    target_arms: tuple[str, ...] = TARGET_ARMS,
) -> dict[str, list[Job]]:
    """The surviving jobs for each requested stage, after validation and skipping.

    Shared by the SLURM launcher and the local runner, so the two can never disagree
    about what a stage's task list is. Off-grid targets are fatal for the equation
    stage and a warning for the others (curve off-grid is the experiment), and the
    skip filter is applied last, so the printed counts are what will actually run.

    ``target_arms`` replicates the *reference* stage per target momentum (ADR 0021).
    Curve and equation ignore it — their arm comes from the source they transfer.
    """
    jobs: dict[str, list[Job]] = {}
    for stage in stages:
        check_on_grid(targets, set(grid), stage)
        if stage == "reference":
            # Two-phase (ADR 0019): asking for "reference" plans both its candidate
            # array and its selector array, inserted in dependency order so a launcher
            # that iterates the result submits the candidates first.
            for phase, built_phase in (
                (
                    "reference-candidate",
                    candidate_jobs(REFERENCES, targets, args, arms=target_arms),
                ),
                ("reference", reference_jobs(REFERENCES, targets, args, target_arms)),
            ):
                surviving_phase = drop_finished(built_phase, args.cache_root)
                skipped_phase = len(built_phase) - len(surviving_phase)
                print(
                    f"  {phase}: {len(surviving_phase)} task(s) to run "
                    f"({skipped_phase} already done)"
                )
                jobs[phase] = surviving_phase
            continue
        if stage == "curve":
            if not args.schedules_parquet:
                raise SystemExit("--schedules_parquet is required for the curve stage")
            regimes = source_regimes(args.schedules_parquet)
            scoped = scope or SourceScope()
            regimes = scope_regimes(
                regimes,
                arch=scoped.arch,
                min_seeds=scoped.min_seeds,
                max_seeds=scoped.max_seeds,
            )
            built = curve_jobs(regimes, targets, args)
        elif stage == "equation":
            if not args.eval_dir:
                raise SystemExit("--eval_dir is required for the equation stage")
            from sr_category import load_category_map

            category_map = load_category_map(pathlib.Path(args.eval_dir) / "category_map.json")
            built = equation_jobs(category_map, targets, args, synthesis_arm(args.eval_dir))
        else:
            raise SystemExit(f"unknown stage {stage!r}")

        surviving = drop_finished(built, args.cache_root)
        skipped = len(built) - len(surviving)
        print(f"  {stage}: {len(surviving)} task(s) to run ({skipped} already done)")
        jobs[stage] = surviving
    return jobs


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

    Job ids are joined with ``:``, not ``,``. SLURM's grammar is
    ``<type>:<jobid>[:<jobid>...][,<type>:<jobid>...]`` — the comma separates
    dependency *types*, so ``afterok:11,22`` is a parse error rather than "after
    both". Every multi-prerequisite job in this DAG (the reference selector, the
    plot assembler, any chunked stage) depends on the colon form.
    """
    if not prerequisites:
        return ""
    return "#SBATCH --dependency=afterok:" + ":".join(prerequisites) + "\n"


def chunk_ranges(n_jobs: int, max_array_size: int) -> list[tuple[int, int]]:
    """Split ``n_jobs`` array tasks into ``(offset, count)`` chunks SLURM will accept.

    SLURM rejects an array whose largest index reaches ``MaxArraySize`` (commonly
    1001) with "Invalid job array specification", and it is a *cluster* limit, not a
    per-stage one. The curve stage is 248 source policies x 6 target regimes = 1,488
    tasks, so it must be submitted as several arrays over the same manifest, each
    starting its indices at 0 and offset into its own slice.

    Chunking rather than shrinking the stage is deliberate: the manifest stays the
    single record of what was launched, and cell-level resumption via
    :func:`drop_finished` is unaffected, because a task still owns exactly one cell.

    A non-positive ``max_array_size`` disables chunking, for a cluster whose limit is
    unknown or unlimited.
    """
    if max_array_size <= 0:
        return [(0, n_jobs)]
    return [
        (offset, min(max_array_size, n_jobs - offset))
        for offset in range(0, n_jobs, max_array_size)
    ]


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
    offset: int = 0,
) -> str:
    """The sbatch script submitting one stage as a job array over its manifest.

    Task ``i`` runs manifest line ``i + 1 + offset`` verbatim, so the manifest is the
    single record of what was launched and the array index means nothing beyond
    "which line". ``%<throttle>`` caps concurrent tasks so a large cross-product does
    not flood the allocation.

    ``offset`` exists because a stage can exceed SLURM's ``MaxArraySize`` and then
    has to be submitted as several arrays over the *same* manifest (see
    :func:`chunk_ranges`). Every array's indices restart at 0, so the offset is what
    keeps chunk 2 from re-running chunk 1's lines.
    """
    line_offset = f" + {offset}" if offset else ""
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

# The manifest line IS the task: line (index+1+offset) of the file the launcher wrote.
CMD=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1{line_offset}))p" {manifest})
if [ -z "$CMD" ]; then
    echo "no manifest line for task $SLURM_ARRAY_TASK_ID in {manifest}" >&2
    exit 1
fi
echo "cmd: $CMD"

# Exit with the producer's status, not the trailing echo's. The DAG is wired with
# `afterok`, so a task that swallowed a non-zero status would report success and let
# the assembler run over cells that were never written.
time eval "$CMD"
status=$?

echo "Job finished with exit code $status at: `date`"
exit $status
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

# Exit with the command's status — see array_sbatch.
time {command}
status=$?

echo "Job finished with exit code $status at: `date`"
exit $status
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
        "uv run --no-sync transfer_preflight.py"
        f" --datasets {' '.join(shlex.quote(d) for d in datasets)}"
        f" --batch_size {args.batch_size}"
    )
