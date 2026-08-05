"""Tests for what a synthesis is fitted on: the rows sampled from each run and the
operator search space PySR is given (ADR 0016).

Both are fit-defining, so both are also synthesis-identity fields — a synthesis
sampled at a different density, or searched over a different operator set, must not
share a slug (and therefore a warm-start directory) with another (ADR 0005).
"""

import dataclasses
import importlib.util
import sys
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from symbolic_regression import sample_points_per_run

_SR_STARTER = Path(__file__).resolve().parents[2] / "cc" / "slurm" / "sr-run-starter.py"


@pytest.fixture(scope="module")
def sr_starter():
    """``sr-run-starter.py`` is a launcher, not an importable module (its filename is
    not an identifier), so it is loaded by path."""
    sys.path.insert(0, str(_SR_STARTER.parent))  # for `_slurm_account`
    try:
        spec = importlib.util.spec_from_file_location("sr_run_starter", _SR_STARTER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(_SR_STARTER.parent))


def _run_rows(run_id: str, n_steps: int) -> list[dict]:
    """One run's schedule rows as compile_results_fetch writes them."""
    return [
        {
            "run_id": run_id,
            "T": n_steps,
            "inner_step": step,
            "step_norm": step / n_steps,
            "sigma": 1.0 + step,
        }
        for step in range(n_steps)
    ]


def test_every_run_contributes_the_same_number_of_points_regardless_of_T():
    """The fit is over step_norm ∈ [0,1], so a long run must not outweigh a short
    one just by having more inner steps (ADR 0016)."""
    schedules = pd.DataFrame(_run_rows("short", 200) + _run_rows("long", 700))

    sampled = sample_points_per_run(schedules, points_per_run=10)

    counts = sampled.groupby("run_id").size()
    assert counts["short"] == counts["long"] == 10


def test_run_shorter_than_the_budget_keeps_every_step():
    """Asking for more points than a run has must not drop rows or divide by zero."""
    schedules = pd.DataFrame(_run_rows("tiny", 4))

    sampled = sample_points_per_run(schedules, points_per_run=50)

    assert list(sampled["inner_step"]) == [0, 1, 2, 3]


def test_sampling_density_changes_the_synthesis_slug():
    """Two differently-sampled syntheses fit different problems, so they must not
    share a warm-start directory (ADR 0005)."""
    from sr_identity import slug_for

    sparse = {"cache_dir": "sweep", "points_per_run": 20}
    dense = {"cache_dir": "sweep", "points_per_run": 200}

    assert slug_for(sparse) != slug_for(dense)


def test_sampling_density_survives_a_chain_resubmit():
    """A chained job re-emits its identity as CLI flags; the successor must land on
    the same sampling density (and therefore the same slug)."""
    from dataclasses import asdict

    import tyro

    from sr_identity import identity_flags, slug_for
    from symbolic_regression import PySRConfig

    conf = PySRConfig(cache_dir="sweep", points_per_run=20)
    reparsed = tyro.cli(PySRConfig, args=["--cache_dir", "sweep", *identity_flags(asdict(conf))])

    assert reparsed.points_per_run == 20
    assert slug_for(asdict(reparsed)) == slug_for(asdict(conf))


class _RecordingRegressor:
    """Stands in for PySRRegressor — the third-party boundary — so the search space
    handed to it can be inspected without a Julia fit."""

    last_kwargs: ClassVar[dict] = {}

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    def fit(self, X, y, variable_names=None):
        return self


def _search_space(monkeypatch, tmp_path, **conf_kwargs) -> dict:
    """The kwargs a fresh search is constructed with, for a config."""
    import symbolic_regression
    from symbolic_regression import PySRConfig, run_regression

    monkeypatch.setattr(symbolic_regression, "PySRRegressor", _RecordingRegressor)
    df = pd.DataFrame({"step_norm": [0.0, 0.5, 1.0], "sigma": [1.0, 2.0, 3.0]})
    conf = PySRConfig(cache_dir="sweep", **conf_kwargs)
    run_regression(df, "sigma", conf, tmp_path, procs=0)
    return _RecordingRegressor.last_kwargs


def test_configured_operators_reach_the_search(monkeypatch, tmp_path):
    kwargs = _search_space(
        monkeypatch, tmp_path, binary_operators=("+", "*"), unary_operators=("sin",)
    )

    assert list(kwargs["binary_operators"]) == ["+", "*"]
    assert list(kwargs["unary_operators"]) == ["sin"]


def test_default_operators_can_express_an_additive_offset(monkeypatch, tmp_path):
    """A per-condition constant must be able to shift the shape, not only scale it —
    the modulation the FirSweep shape plots show most clearly (ADR 0016)."""
    kwargs = _search_space(monkeypatch, tmp_path)

    assert {"+", "-"} <= set(kwargs["binary_operators"])


def test_the_search_does_not_denoise_its_inputs(monkeypatch, tmp_path):
    """PySR's denoiser fits a GP over EVERY column of X. In template mode X is
    [step_norm, category], so it would smooth across the arbitrary condition index —
    blurring precisely what the per-condition constants exist to capture (ADR 0016)."""
    kwargs = _search_space(monkeypatch, tmp_path)

    assert kwargs["denoise"] is False


def test_operators_survive_a_chain_resubmit():
    """Operator names are also CLI-hostile tokens ("-", "+", "/"): a chained job
    re-emits them as flag values, and must land on the same search space and slug."""
    from dataclasses import asdict

    import tyro

    from sr_identity import identity_flags, slug_for
    from symbolic_regression import PySRConfig

    conf = PySRConfig(cache_dir="sweep")
    reparsed = tyro.cli(PySRConfig, args=["--cache_dir", "sweep", *identity_flags(asdict(conf))])

    # Order is canonicalised (sorted) on the way through, which is why the operator
    # fields are order-insensitive identity fields — the search space is a set.
    assert set(reparsed.binary_operators) == set(conf.binary_operators)
    assert set(reparsed.unary_operators) == set(conf.unary_operators)
    assert slug_for(asdict(reparsed)) == slug_for(asdict(conf))


def test_operator_set_changes_the_synthesis_slug():
    from sr_identity import slug_for

    narrow = {"cache_dir": "sweep", "binary_operators": ("*", "/")}
    wide = {"cache_dir": "sweep", "binary_operators": ("+", "-", "*", "/")}

    assert slug_for(narrow) != slug_for(wide)


def test_operator_order_does_not_change_the_synthesis_slug():
    """The same operators listed in a different order search the same space, so they
    must share a warm-start directory rather than forking one."""
    from sr_identity import slug_for

    a = {"cache_dir": "sweep", "binary_operators": ("+", "*"), "unary_operators": ("exp", "sqrt")}
    b = {"cache_dir": "sweep", "binary_operators": ("*", "+"), "unary_operators": ("sqrt", "exp")}

    assert slug_for(a) == slug_for(b)


def test_sbatch_invocation_survives_shell_expansion(sr_starter, tmp_path):
    """The launcher embeds the identity flags in a bash script that SLURM runs with
    --chdir=src/. `*` is an operator name AND a glob, so an unquoted one would expand
    to every file in that directory and be passed off as a binary operator."""
    import subprocess

    conf = sr_starter.SRSlurmConfig(cache_dir="sweep")
    script = conf.sbatch_file("sigma")

    invocation = next(
        ln for ln in script.splitlines() if ln.strip().startswith("time uv run symbolic_regression")
    )
    args = invocation.split("symbolic_regression.py", 1)[1]
    (tmp_path / "decoy.py").touch()
    shown = subprocess.run(
        ["bash", "-c", f"printf '%s\\n' {args}"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    tokens = shown.stdout.split("\n")

    assert "decoy.py" not in tokens, "a glob expanded against the job's working directory"
    assert "*" in tokens, "the multiplication operator did not reach the script"


def test_launcher_and_script_agree_on_every_identity_field(sr_starter):
    """The launcher names the job by the slug it computes itself; if any identity
    default drifts from PySRConfig's, that name points at a directory the script
    will never write to (ADR 0005)."""
    from sr_identity import IDENTITY_FIELDS
    from symbolic_regression import PySRConfig

    script = {f.name: f for f in dataclasses.fields(PySRConfig)}
    launcher = {f.name: f for f in dataclasses.fields(sr_starter.SRSlurmConfig)}

    for name in IDENTITY_FIELDS:
        if name == "cache_dir":  # always passed explicitly; no default to mirror
            continue
        assert name in script, f"{name} is an identity field but not a PySRConfig field"
        assert name in launcher, f"{name} is an identity field but not on SRSlurmConfig"
        assert launcher[name].default == script[name].default, (
            f"{name} default differs: launcher {launcher[name].default!r} "
            f"vs script {script[name].default!r}"
        )
