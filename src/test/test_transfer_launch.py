"""Launcher core for the transfer-evaluation SLURM DAG (cc/slurm/transfer-run-starter.py).

The launcher's job is to turn a target cross-product into a per-stage manifest of
`uv run transfer_*.py ...` lines, one per array task, skipping cells already on disk.
These are the pure parts; `_submit()` is integration glue and is not unit-tested.
"""

import importlib.util
import itertools
import sys
from pathlib import Path

import pytest

from transfer_launch import (
    Job,
    ProducerArgs,
    SourceRegime,
    Target,
    absolute_path,
    array_sbatch,
    cell_filename,
    check_on_grid,
    condition_grid,
    condition_source_id,
    curve_jobs,
    drop_finished,
    equation_jobs,
    expand_targets,
    manifest_text,
    preflight_command,
    reference_jobs,
    serial_sbatch,
    source_regimes,
)


class TestTargetCrossProduct:
    """The targets of a launch are the full `datasets × eps × T` cross-product
    (ADR 0008): every target dataset is evaluated at every requested budget."""

    def test_every_dataset_eps_T_combination_becomes_a_target(self):
        targets = expand_targets(
            datasets=("eyepacs", "chexpert"),
            epsilons=(1.0, 8.0),
            timesteps=(200, 5000),
            delta=1e-7,
        )

        assert len(targets) == 8
        assert Target(dataset="eyepacs", eps=1.0, T=200, delta=1e-7) in targets
        assert Target(dataset="chexpert", eps=8.0, T=5000, delta=1e-7) in targets

    def test_targets_are_ordered_deterministically(self):
        # The manifest line order IS the array task index, so a relaunch must
        # produce the same ordering for the same inputs.
        args = {"datasets": ("b", "a"), "epsilons": (8.0, 1.0), "timesteps": (5000, 200)}
        assert expand_targets(**args) == expand_targets(**args)


class TestCellFilenameMatchesTheProducers:
    """The skip filter decides a job is done by looking for a cell's parquet, so
    the name the launcher predicts must be the name a producer actually writes.
    One shared function, checked against the real writer."""

    def test_predicted_name_is_the_one_write_transfer_cell_produces(self, tmp_path):
        from util.transfer import SourcePolicy, TargetSpec, transfer_rows, write_transfer_cell

        source = SourcePolicy(
            run_id="abc123", dataset="mnist", eps=1.0, delta=1e-7, T=200, p=0.01, arch="cnn"
        )
        target = TargetSpec(name="eyepacs", eps=8.0, delta=1e-7, T=5000, arch="cnn")
        rows = transfer_rows("curve", source, target, [(0, 0.5, 1.0)])

        written = write_transfer_cell(rows, tmp_path)

        predicted = cell_filename("abc123", Target(dataset="eyepacs", eps=8.0, T=5000))
        assert written.name == predicted


def _schedules_parquet(tmp_path, runs):
    """A minimal schedules.parquet: (run_id, regime) pairs, 2 inner steps each."""
    import pandas as pd

    rows = [
        {
            "run_id": run_id,
            "inner_step": step,
            "dataset": dataset,
            "eps": eps,
            "T": T,
            "arch_label": arch,
            "sigma": 1.0,
            "clip": 1.0,
        }
        for run_id, dataset, eps, T, arch in runs
        for step in (0, 1)
    ]
    path = tmp_path / "schedules.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


class TestSourceRegimeEnumeration:
    """A curve job's unit is one source *regime* (CONTEXT.md), not one run: every
    seed-policy learned under the same (dataset, ε, T, arch) is transferred by a
    single job, because the matrix reports their spread as one regime's
    generalization consistency (ADR 0008)."""

    def test_runs_sharing_a_regime_are_grouped_into_one_job_unit(self, tmp_path):
        parquet = _schedules_parquet(
            tmp_path,
            [
                ("runA", "mnist", 1.0, 200, "cnn"),
                ("runB", "mnist", 1.0, 200, "cnn"),
                ("runC", "mnist", 1.0, 200, "mlp"),
            ],
        )

        regimes = source_regimes(parquet)

        assert len(regimes) == 2
        by_arch = {r.arch: r for r in regimes}
        # Both seeds of the cnn regime ride in the same job...
        assert by_arch["cnn"].run_ids == ("runA", "runB")
        # ...and the differing arch is a regime of its own.
        assert by_arch["mlp"].run_ids == ("runC",)
        assert by_arch["mlp"].dataset == "mnist"
        assert by_arch["mlp"].eps == 1.0
        assert by_arch["mlp"].T == 200


class TestOnGridValidation:
    """Equation transfer is on-grid only — the template's per-condition constants
    are indexed by discrete condition, not a function of (ε, T) (ADR 0008), so an
    off-grid equation target is a launch error. Curve transfer off-grid is the
    whole point of the experiment, so there it is only worth a warning."""

    def test_off_grid_equation_target_is_a_launch_error(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            check_on_grid([Target("eyepacs", eps=8.0, T=5000)], grid={(1.0, 200)}, stage="equation")

        assert "8" in str(excinfo.value) and "5000" in str(excinfo.value)

    def test_on_grid_equation_target_passes(self):
        targets = [Target("eyepacs", eps=1.0, T=200)]

        assert check_on_grid(targets, grid={(1.0, 200)}, stage="equation") == targets

    def test_off_grid_curve_target_is_warned_about_but_kept(self, capsys):
        targets = [Target("eyepacs", eps=8.0, T=5000)]

        kept = check_on_grid(targets, grid={(1.0, 200)}, stage="curve")

        assert kept == targets
        assert "warning" in capsys.readouterr().out.lower()


class TestConditionSourceId:
    """An equation cell's source_id is its condition, not a run id (ADR 0015). The
    launcher must predict the same id the producer writes, or the skip filter looks
    for the wrong parquet."""

    def test_predicted_id_is_the_one_equation_source_produces(self):
        from transfer_equation import equation_source

        condition = {"dataset": "fashion-mnist", "eps": 1.0, "T": 200, "arch_label": "cnn"}

        assert condition_source_id(3, condition) == equation_source(3, condition).run_id


class TestConditionGrid:
    """The trained (ε, T) grid that on-grid validation checks against is read off
    the SR run's category map — the same file the equation producer borrows its
    per-condition constants from."""

    def test_grid_is_the_distinct_eps_T_of_the_trained_conditions(self, tmp_path):
        import json

        path = tmp_path / "category_map.json"
        path.write_text(
            json.dumps(
                [
                    {"dataset": "fashion-mnist", "eps": 1.0, "T": 200, "arch_label": "cnn"},
                    {"dataset": "fashion-mnist", "eps": 1.0, "T": 200, "arch_label": "mlp"},
                    {"dataset": "fashion-mnist", "eps": 8.0, "T": 5000, "arch_label": "cnn"},
                ]
            )
        )

        # Two conditions share an (eps, T) — the grid is over budgets, not conditions.
        assert condition_grid(path) == {(1.0, 200), (8.0, 5000)}


def _parse(config_cls, job):
    """Parse a manifest line's args with the producer's own tyro config.

    The manifest line is the launcher's real output — a command the compute node
    runs verbatim — so the binding contract is that the producer accepts it.
    """
    import shlex

    import tyro

    args = shlex.split(job.args)
    assert args[:2] == ["uv", "run"], job.args
    script = next(i for i, a in enumerate(args) if a.endswith(".py"))
    return tyro.cli(config_cls, args=args[script + 1 :])


class TestCurveJobs:
    """A curve job is one source *policy* × one target. The regime remains the unit
    of analysis — the assembler still reports the spread across a regime's seeds —
    but it is not the unit of scheduling: a regime's policies are evaluated serially
    within a task, so a regime-sized job scales its runtime with the seed count and
    overruns the wall clock. One policy per task keeps a task's cost bounded by a
    single evaluation and lets the skip filter resume at cell granularity."""

    def test_one_job_per_policy_target_selecting_that_policy(self):
        from transfer_curve import CurveCellConfig

        regimes = [
            SourceRegime("mnist", 1.0, 200, "cnn", ("runA", "runB")),
            SourceRegime("mnist", 8.0, 200, "mlp", ("runC",)),
        ]
        targets = [Target("eyepacs", eps=4.0, T=5000), Target("chexpert", eps=4.0, T=5000)]

        jobs = curve_jobs(
            regimes, targets, ProducerArgs(cache_root="/c", schedules_parquet="/s.pq")
        )

        # 3 policies x 2 targets, not 2 regimes x 2 targets.
        assert len(jobs) == 6
        conf = _parse(CurveCellConfig, jobs[0])
        assert conf.source_run_id == "runA"
        assert (conf.target, conf.target_eps, conf.target_T) == ("eyepacs", 4.0, 5000)
        assert conf.schedules_parquet == "/s.pq"
        assert conf.cache_root == "/c"

    def test_a_job_owns_exactly_the_cell_its_policy_writes(self):
        regimes = [SourceRegime("mnist", 1.0, 200, "cnn", ("runA", "runB"))]
        targets = [Target("eyepacs", eps=4.0, T=5000)]

        jobs = curve_jobs(regimes, targets, ProducerArgs(cache_root="/c"))

        assert [job.cells for job in jobs] == [
            ("transfer/curve/runA__eyepacs__eps4_T5000.parquet",),
            ("transfer/curve/runB__eyepacs__eps4_T5000.parquet",),
        ]

    def test_a_finished_policy_is_skipped_without_holding_back_its_regime(self, tmp_path):
        # The point of per-policy jobs: one seed timing out no longer forces its
        # regime-mates to be recomputed on the relaunch.
        regimes = [SourceRegime("mnist", 1.0, 200, "cnn", ("runA", "runB"))]
        targets = [Target("eyepacs", eps=4.0, T=5000)]
        done = tmp_path / "transfer" / "curve"
        done.mkdir(parents=True)
        (done / "runA__eyepacs__eps4_T5000.parquet").touch()

        jobs = curve_jobs(regimes, targets, ProducerArgs(cache_root=str(tmp_path)))

        assert [job.cells for job in drop_finished(jobs, tmp_path)] == [
            ("transfer/curve/runB__eyepacs__eps4_T5000.parquet",)
        ]


_CONDITIONS = [
    {"dataset": "fashion-mnist", "eps": 1.0, "T": 200, "arch_label": "cnn"},
    {"dataset": "fashion-mnist", "eps": 1.0, "T": 200, "arch_label": "mlp"},
    {"dataset": "fashion-mnist", "eps": 8.0, "T": 5000, "arch_label": "cnn"},
]


class TestEquationJobs:
    """An equation job is one target (ε, T): it carries *every* distilled condition
    at that budget in a single invocation (read off, not selected — ADR 0008), so
    it owns one cell per matching condition."""

    def test_one_job_per_target_owning_a_cell_per_matching_condition(self):
        from transfer_equation import EquationCellConfig

        targets = [Target("eyepacs", eps=1.0, T=200)]

        (job,) = equation_jobs(_CONDITIONS, targets, ProducerArgs(cache_root="/c", eval_dir="/e"))

        conf = _parse(EquationCellConfig, job)
        assert (conf.target, conf.target_eps, conf.target_T) == ("eyepacs", 1.0, 200)
        assert conf.eval_dir == "/e"
        # Both conditions at (1.0, 200) get a cell; the (8.0, 5000) one does not.
        assert job.cells == (
            "transfer/equation/fashion-mnist_eps1_T200_cnn_cat1__eyepacs__eps1_T200.parquet",
            "transfer/equation/fashion-mnist_eps1_T200_mlp_cat2__eyepacs__eps1_T200.parquet",
        )

    def test_a_target_with_no_matching_condition_produces_no_job(self):
        # Off-grid is fatal at validation time; if a caller skips that, an
        # off-grid target must still never become an array task that cannot work.
        targets = [Target("eyepacs", eps=4.0, T=1000)]

        assert equation_jobs(_CONDITIONS, targets, ProducerArgs(cache_root="/c")) == []


class TestReferenceJobs:
    """A reference job is one native reference × one target, so the three
    references fan out rather than sharing one long serial job."""

    def test_one_job_per_reference_and_target(self):
        from transfer_reference import ReferenceCellConfig

        targets = [Target("eyepacs", eps=1.0, T=200), Target("chexpert", eps=1.0, T=200)]

        jobs = reference_jobs(("Constant", "Median"), targets, ProducerArgs(cache_root="/c"))

        assert len(jobs) == 4
        conf = _parse(ReferenceCellConfig, jobs[0])
        assert conf.reference == "Constant"
        assert (conf.target, conf.target_eps, conf.target_T) == ("eyepacs", 1.0, 200)
        # A reference job writes exactly its own regime's cell.
        assert jobs[0].cells == ("transfer/reference/Constant__eyepacs__eps1_T200.parquet",)


class TestSkipFilter:
    """Already-finished work is dropped at manifest-build time, not resumed inside
    the producer (settled launch design): a relaunch after a partial run submits
    only the jobs whose cells are still missing."""

    def _job(self, *cells):
        return Job(stage="curve", args="uv run --no-sync transfer_curve.py", cells=cells)

    def test_a_job_whose_cells_all_exist_is_dropped(self, tmp_path):
        (tmp_path / "transfer" / "curve").mkdir(parents=True)
        (tmp_path / "transfer" / "curve" / "done.parquet").touch()

        assert drop_finished([self._job("transfer/curve/done.parquet")], tmp_path) == []

    def test_a_job_with_any_missing_cell_is_kept_whole(self, tmp_path):
        # Cells are not independently resumable — the producer re-runs its whole
        # regime — so one missing cell means the job is still outstanding.
        (tmp_path / "transfer" / "curve").mkdir(parents=True)
        (tmp_path / "transfer" / "curve" / "done.parquet").touch()
        job = self._job("transfer/curve/done.parquet", "transfer/curve/todo.parquet")

        assert drop_finished([job], tmp_path) == [job]

    def test_nothing_on_disk_keeps_every_job(self, tmp_path):
        jobs = [self._job("transfer/curve/a.parquet"), self._job("transfer/curve/b.parquet")]

        assert drop_finished(jobs, tmp_path) == jobs


class TestManifest:
    """The manifest is the array's work list: line N is exactly the command array
    task N runs, so line order is the task index and must round-trip."""

    def test_line_n_is_the_command_of_job_n(self):
        jobs = [
            Job("curve", "uv run --no-sync transfer_curve.py --target eyepacs", ()),
            Job("curve", "uv run --no-sync transfer_curve.py --target chexpert", ()),
        ]

        lines = manifest_text(jobs).splitlines()

        assert lines == [job.args for job in jobs]

    def test_manifest_ends_with_a_newline_so_the_last_line_is_readable(self):
        # `sed -n "$((i+1))p"` silently yields nothing for an unterminated last
        # line on some shells; a trailing newline makes the final task safe.
        assert manifest_text([Job("curve", "uv run x.py", ())]).endswith("\n")


class TestArraySbatch:
    """Each stage is submitted as one job array over its manifest, throttled, and
    gated behind the preflight job that warms the dataset cache."""

    def _script(self, n_jobs=4, **kwargs):
        options = {
            "stage": "curve",
            "manifest": "/m/curve.txt",
            "n_jobs": n_jobs,
            "walltime": "00-02:55:00",
            "project_dir": "/proj/src",
            "account": "acct",
            "logfile": "/logs/%A_%a/%x.log",
        }
        options.update(kwargs)
        return array_sbatch(**options)

    def test_array_spans_one_task_per_manifest_line_and_is_throttled(self):
        script = self._script(n_jobs=4, throttle=2)

        assert "#SBATCH --array=0-3%2" in script

    def test_the_task_runs_its_own_manifest_line(self):
        script = self._script()

        # The array task selects line (index+1) of the manifest and runs it — the
        # manifest line order is the task index.
        assert "SLURM_ARRAY_TASK_ID" in script
        assert "/m/curve.txt" in script

    def test_producer_stages_request_a_gpu(self):
        script = self._script(stage="curve")

        assert "--gpus=1" in script
        assert "--mem-per-gpu=12G" in script

    def test_preflight_dependency_is_afterok_never_singleton(self):
        # `singleton` is chain serialization from run-starter.py: it would force the
        # whole array to run one task at a time, defeating the fan-out.
        script = self._script(prerequisites=("12345",))

        assert "#SBATCH --dependency=afterok:12345" in script
        assert "singleton" not in script

    def test_the_task_exits_with_its_command_status(self):
        # The whole DAG is wired with `afterok`, which reads the task's exit status.
        # A trailing `echo` would make every task exit 0 and a failed producer would
        # report success, so the status must be captured and re-raised.
        script = self._script()

        assert script.rstrip().endswith("exit $status")
        assert "status=$?" in script


class TestSerialSbatch:
    """The preflight and the plot assembler are single CPU jobs, not arrays. The
    preflight is deliberately sequential: `util/dataloaders.py` downloads into the
    repo with no locking or temp-rename, so concurrent first-touch of a dataset
    would race. Warming the cache once, up front, is the whole reason it exists."""

    def _script(self, **kwargs):
        options = {
            "name": "preflight",
            "command": "uv run --no-sync -c 'warm caches'",
            "walltime": "00-02:55:00",
            "project_dir": "/proj/src",
            "account": "acct",
            "logfile": "/logs/%j/%x.log",
        }
        options.update(kwargs)
        return serial_sbatch(**options)

    def test_requests_no_gpu_and_is_not_an_array(self):
        script = self._script()

        assert "--gpus" not in script
        assert "--array" not in script
        assert "--mem-per-cpu" in script

    def test_runs_its_command_verbatim(self):
        script = self._script(command="uv run --no-sync transfer_plot.py --cache_root /c")

        assert "uv run --no-sync transfer_plot.py --cache_root /c" in script

    def test_plot_waits_for_every_producer_array(self):
        script = self._script(name="plot", prerequisites=("11", "22", "33"))

        assert "#SBATCH --dependency=afterok:11,22,33" in script

    def test_the_job_exits_with_its_command_status(self):
        # Same contract as the array script: `afterok` is only meaningful if a
        # failing command actually fails the job.
        script = self._script()

        assert script.rstrip().endswith("exit $status")
        assert "status=$?" in script


_LAUNCHERS = {
    "slurm": Path(__file__).resolve().parents[2] / "cc" / "slurm" / "transfer-run-starter.py",
    "local": Path(__file__).resolve().parents[2] / "cc" / "local" / "transfer-local.py",
}


def _load_launcher(path: Path):
    """Load a launcher by path — its filename is not a valid identifier.

    Same device as ``test_run_starter.py``; the parent dir goes on the path for the
    launcher's sibling imports (``_slurm_account``).
    """
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path.parent))


class TestProducerPathsAreAbsolute:
    """Both launchers must absolutise the paths they hand to producers.

    A producer runs with its cwd pinned to ``src/`` (``#SBATCH --chdir`` on the
    cluster, ``cwd=`` for the local pool) while the launcher is invoked from wherever
    the user happens to be — normally the repo root. A relative path therefore names
    one file to the launcher and a different one to the producer. This is not
    hypothetical: a 744-task curve array was submitted with a relative
    ``--schedules_parquet`` that the launcher read fine, and every single task died
    on ``src/./src/cache/.../schedules.parquet`` seconds after starting.
    """

    def test_empty_stays_empty_so_an_unrequested_stage_is_still_unrequested(self):
        # plan_jobs tests `if not args.eval_dir`, and abspath("") is the cwd — which
        # would turn "stage not requested" into a real, wrong directory.
        assert absolute_path("") == ""

    def test_a_relative_path_resolves_against_the_invoking_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        assert absolute_path("./cache/x.parquet") == str(tmp_path / "cache" / "x.parquet")

    @pytest.mark.parametrize("launcher", sorted(_LAUNCHERS))
    def test_launcher_hands_producers_absolute_paths(self, launcher, tmp_path, monkeypatch):
        module = _load_launcher(_LAUNCHERS[launcher])
        config_cls = next(
            getattr(module, name)
            for name in ("TransferSlurmConfig", "TransferLocalConfig")
            if hasattr(module, name)
        )
        monkeypatch.chdir(tmp_path)

        args = config_cls(
            cache_root="out/transfer",
            schedules_parquet="./src/cache/results/sweep/schedules.parquet",
            eval_dir="cache/pysr_eval/slug",
        ).producer_args

        assert args.cache_root == str(tmp_path / "out" / "transfer")
        assert args.schedules_parquet == str(
            tmp_path / "src" / "cache" / "results" / "sweep" / "schedules.parquet"
        )
        assert args.eval_dir == str(tmp_path / "cache" / "pysr_eval" / "slug")

    @pytest.mark.parametrize("launcher", sorted(_LAUNCHERS))
    def test_launcher_leaves_an_unrequested_stage_empty(self, launcher, tmp_path, monkeypatch):
        module = _load_launcher(_LAUNCHERS[launcher])
        config_cls = next(
            getattr(module, name)
            for name in ("TransferSlurmConfig", "TransferLocalConfig")
            if hasattr(module, name)
        )
        monkeypatch.chdir(tmp_path)

        args = config_cls(cache_root="out").producer_args

        assert (args.schedules_parquet, args.eval_dir) == ("", "")


class TestPreflightCommand:
    """One sequential preflight job warms each target dataset's cache before any
    producer runs. Datasets are deduplicated across the target cross-product: the
    same dataset appears at every (ε, T), but downloading it twice concurrently is
    exactly the race the preflight exists to prevent."""

    def test_each_target_dataset_is_warmed_exactly_once(self):
        targets = [
            Target("eyepacs", eps=1.0, T=200),
            Target("eyepacs", eps=8.0, T=5000),
            Target("chexpert", eps=1.0, T=200),
        ]

        command = preflight_command(targets, ProducerArgs(cache_root="/c"))

        import shlex

        tokens = shlex.split(command)
        after = tokens[tokens.index("--datasets") + 1 :]
        named = list(itertools.takewhile(lambda t: not t.startswith("--"), after))
        assert named == ["chexpert", "eyepacs"]
