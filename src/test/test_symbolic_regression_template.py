"""Tests for per-condition template symbolic regression (ADR 0006).

Covers the template-mode pieces added on top of the pooled scalar fit:
- reading the fitted per-condition constants back off a PySR equation row
  (``extract_template_constants``),
- the ``category_map.json`` round-trip (build → persist → reload → rebuild the
  ``category`` column),
- ``constants.csv`` persistence (constants joined back to condition keys),
- ``template_mode`` / ``n_template_params`` synthesis-identity plumbing.

The extraction unit tests use a faithful fake of PySR's Julia ``NamedTuple``
(attribute access works; string indexing raises, mirroring the real
``MethodError``) so they run without a Julia fit. One ``@pytest.mark.slow``
integration test does a real tiny ``TemplateExpressionSpec`` fit to lock the
actual read-back API path.
"""

import types
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from sr_category import (
    CONDITION_KEYS,
    build_category_map,
    build_constants_table,
    category_series,
    load_category_map,
    save_category_map,
    template_param_names,
)
from symbolic_regression import build_template_spec, extract_template_constants


class _FakeParams:
    """Mimics a Julia NamedTuple: attribute access works, string indexing raises."""

    def __init__(self, **fields):
        for name, value in fields.items():
            setattr(self, name, value)

    def __getitem__(self, key):  # NamedTuple has no string getindex
        raise TypeError(f"NamedTuple has no string getindex for {key!r} (MethodError)")


def _equation_row(**param_arrays):
    params = _FakeParams(**param_arrays)
    julia_expr = types.SimpleNamespace(metadata=types.SimpleNamespace(parameters=params))
    return {"julia_expression": julia_expr}


def test_extract_returns_float64_array_per_param():
    row = _equation_row(
        p1=np.asarray([1.0, 2.0, 0.5], dtype=np.float32),
        p2=np.asarray([0.1, 1.5, -0.5], dtype=np.float32),
    )

    consts = extract_template_constants(row, ["p1", "p2"])

    assert list(consts) == ["p1", "p2"]
    assert consts["p1"].dtype == np.float64
    np.testing.assert_allclose(consts["p1"], [1.0, 2.0, 0.5])
    np.testing.assert_allclose(consts["p2"], [0.1, 1.5, -0.5])


def _match_unordered(recovered: list[np.ndarray], planted: list[np.ndarray], atol: float):
    """Each planted vector matches exactly one recovered vector (slots unordered)."""
    remaining = list(recovered)
    for target in planted:
        hit = next((i for i, r in enumerate(remaining) if np.allclose(r, target, atol=atol)), None)
        assert hit is not None, f"no recovered slot ≈ {target}; got {recovered}"
        remaining.pop(hit)


@pytest.mark.slow
def test_extract_recovers_planted_constants_from_real_fit():
    """Real TemplateExpressionSpec fit → read-back recovers planted per-condition
    constants, order-agnostically across the K parameter slots (ADR 0006)."""
    from pysr import PySRRegressor, TemplateExpressionSpec

    rng = np.random.default_rng(0)
    n_conditions = 3
    scales = np.array([1.0, 2.0, 0.5])
    offsets = np.array([0.1, 1.5, -0.5])

    X = rng.uniform(-3, 3, (600, 2))
    category = rng.integers(0, n_conditions, 600)
    y = scales[category] * np.sin(X[:, 0]) + offsets[category]
    X_with_category = np.column_stack([X, category + 1])  # Julia 1-indexing

    template = TemplateExpressionSpec(
        expressions=["f"],
        variable_names=["x1", "x2", "category"],
        parameters={"p1": n_conditions, "p2": n_conditions},
        combine="f(x1, x2, p1[category], p2[category])",
    )
    model = PySRRegressor(
        expression_spec=template,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["sin"],
        maxsize=10,
        niterations=40,
        parallelism="serial",
        deterministic=True,
        random_state=0,
        verbosity=0,
        progress=False,
    )
    model.fit(X_with_category, y)

    consts = extract_template_constants(model.get_best(), ["p1", "p2"])

    assert consts["p1"].shape == (n_conditions,)
    assert consts["p2"].shape == (n_conditions,)
    _match_unordered([consts["p1"], consts["p2"]], [scales, offsets], atol=1e-2)


# ---------------------------------------------------------------------------
# Slice 2: category_map.json round-trip (condition <-> 1-indexed integer)
# ---------------------------------------------------------------------------


def _conditions_frame():
    """Two conditions, each with 2 seeds × 2 inner steps. Same condition differs
    only by seed/step; the (dataset, eps, T, arch_label) tuple is the category."""
    rows = []
    conditions = [
        ("mnist", 0.5, 100, "mlp"),
        ("cifar-10", 1.0, 200, "cnn"),
    ]
    for ds, eps, T, arch in conditions:
        for seed in (0, 1):
            for step in (0, 1):
                rows.append(
                    {
                        "dataset": ds,
                        "eps": eps,
                        "T": T,
                        "arch_label": arch,
                        "seed": seed,
                        "step_norm": step / 2,
                        "sigma": 1.0 + step,
                    }
                )
    return pd.DataFrame(rows)


def test_category_map_round_trips_and_rebuilds_column(tmp_path):
    df = _conditions_frame()

    cmap = build_category_map(df)
    path = tmp_path / "category_map.json"
    save_category_map(cmap, path)
    reloaded = load_category_map(path)

    # Reload preserves the mapping exactly.
    assert reloaded == cmap

    col = category_series(df, reloaded)
    # 1-indexed, one integer per condition, seeds/steps collapse into it.
    assert set(col) == {1, 2}
    # Every row sharing a condition gets the same category.
    df = df.assign(category=col)
    per_condition = df.groupby(list(CONDITION_KEYS))["category"].nunique()
    assert (per_condition == 1).all()


def test_category_map_is_deterministic_and_seed_independent():
    df = _conditions_frame()
    # Shuffling rows (and dropping a seed) must not change a condition's category.
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)

    cmap_a = build_category_map(df)
    cmap_b = build_category_map(shuffled)
    assert cmap_a == cmap_b
    assert len(cmap_a) == 2  # one entry per condition, seeds collapsed


# ---------------------------------------------------------------------------
# Slice 3: constants.csv persistence (constants joined back to conditions)
# ---------------------------------------------------------------------------


def test_constants_table_joins_constants_to_conditions():
    df = _conditions_frame()
    cmap = build_category_map(df)
    constants = {"p1": np.array([10.0, 20.0]), "p2": np.array([0.1, 0.2])}

    table = build_constants_table(constants, cmap)

    assert len(table) == len(cmap)
    assert set(CONDITION_KEYS) <= set(table.columns)
    assert {"category", "p1", "p2"} <= set(table.columns)
    # Category i (1-indexed) carries constants[name][i-1] and condition cmap[i-1].
    for i, record in enumerate(cmap):
        row = table[table["category"] == i + 1].iloc[0]
        assert row["p1"] == constants["p1"][i]
        assert row["p2"] == constants["p2"][i]
        for key in CONDITION_KEYS:
            assert row[key] == record[key]


def test_constants_table_rejects_length_mismatch():
    cmap = build_category_map(_conditions_frame())  # 2 conditions
    with pytest.raises(ValueError):
        build_constants_table({"p1": np.array([1.0, 2.0, 3.0])}, cmap)


def test_constants_table_csv_round_trips(tmp_path):
    df = _conditions_frame()
    cmap = build_category_map(df)
    constants = {"p1": np.array([10.0, 20.0]), "p2": np.array([0.1, 0.2])}
    table = build_constants_table(constants, cmap)

    path = tmp_path / "constants.csv"
    table.to_csv(path, index=False)
    reloaded = pd.read_csv(path)

    pd.testing.assert_frame_equal(reloaded, table)


# ---------------------------------------------------------------------------
# Slice 4: template_mode / n_template_params synthesis-identity plumbing
# ---------------------------------------------------------------------------


def test_template_fields_are_part_of_synthesis_identity():
    from sr_identity import canonical_identity, identity_hash

    base = {"cache_dir": "x", "template_mode": True, "n_template_params": 3}
    scalar = {**base, "template_mode": False}
    more_params = {**base, "n_template_params": 5}

    h = lambda m: identity_hash(canonical_identity(m))  # noqa: E731
    # A template synthesis and a scalar synthesis with identical filters must NOT
    # share a slug (else they corrupt each other's warm_start; ADR 0006).
    assert h(base) != h(scalar)
    assert h(base) != h(more_params)


def test_identity_flags_omit_template_defaults():
    from sr_identity import identity_flags

    flags = identity_flags({"cache_dir": "x", "template_mode": True, "n_template_params": 3})
    assert "--n_template_params" not in flags
    assert "--no-template_mode" not in flags
    assert "--template_mode" not in flags


def test_identity_flags_emit_nondefault_template_fields():
    from sr_identity import identity_flags

    flags = identity_flags({"cache_dir": "x", "template_mode": False, "n_template_params": 5})
    assert "--no-template_mode" in flags  # bool defaulting True forwards its off-state
    assert "--n_template_params" in flags
    assert flags[flags.index("--n_template_params") + 1] == "5"


def test_identity_flags_reproduce_config_through_cli():
    """identity_flags output is valid CLI that reconstructs the same fit fields."""
    import tyro

    from sr_identity import identity_flags
    from symbolic_regression import PySRConfig

    conf = PySRConfig(cache_dir="x", template_mode=False, n_template_params=5)
    flags = identity_flags(asdict(conf))
    reparsed = tyro.cli(PySRConfig, args=["--cache_dir", "x", *flags])

    assert reparsed.template_mode is False
    assert reparsed.n_template_params == 5


# ---------------------------------------------------------------------------
# Template spec construction (shape f(step_norm) + K per-condition constants)
# ---------------------------------------------------------------------------


def test_template_param_names_are_p1_through_pk():
    assert template_param_names(3) == ["p1", "p2", "p3"]
    assert template_param_names(1) == ["p1"]


def test_build_template_spec_passes_step_norm_and_k_constants_into_shape():
    spec = build_template_spec(n_conditions=4, n_template_params=3)

    # The shape's only real input is step_norm; the K constants are indexed by
    # category and passed into f so PySR discovers the modulation (ADR 0006).
    assert "step_norm" in spec.combine
    for name in ("p1", "p2", "p3"):
        assert f"{name}[category]" in spec.combine
    # Each parameter slot has one constant per condition.
    assert spec.parameters == {"p1": 4, "p2": 4, "p3": 4}


# ---------------------------------------------------------------------------
# Evaluator: rebuild the category column from the persisted map
# ---------------------------------------------------------------------------


def test_evaluator_rebuilds_category_from_map(tmp_path):
    from symbolic_regression_eval import attach_category_column

    df = _conditions_frame()
    cmap = build_category_map(df)
    save_category_map(cmap, tmp_path / "category_map.json")

    out = attach_category_column(df, tmp_path)

    assert "category" in out.columns
    pd.testing.assert_series_equal(out["category"], category_series(df, cmap))


def test_evaluator_skips_category_when_map_absent(tmp_path):
    """Scalar-mode synthesis dirs have no category_map.json — leave the frame as is."""
    from symbolic_regression_eval import attach_category_column

    df = _conditions_frame()
    out = attach_category_column(df, tmp_path)

    assert "category" not in out.columns


# ---------------------------------------------------------------------------
# End-to-end: main() in template mode writes category_map.json + constants.csv
# ---------------------------------------------------------------------------


def _synthetic_cache(tmp_path):
    """A tiny schedules/scalars cache: 2 conditions, 1 seed, learnable σ shape."""
    conditions = [
        ("mnist", 0.5, 20, "mlp", 1.0, 0.2),
        ("cifar-10", 1.0, 20, "cnn", 2.0, 0.5),
    ]
    sched_rows, scalar_rows = [], []
    for i, (ds, eps, T, arch, scale, offset) in enumerate(conditions):
        run_id = f"run{i}"
        for step in range(T):
            sn = step / T
            sigma = scale * sn + offset + 0.5
            sched_rows.append(
                {
                    "run_id": run_id,
                    "run_name": f"name{i}",
                    "dataset": ds,
                    "eps": eps,
                    "T": T,
                    "arch_label": arch,
                    "optimizer": "sgd",
                    "seed": 0,
                    "axis": "sigma",
                    "inner_step": step,
                    "step_norm": sn,
                    "sigma": sigma,
                    "clip": sigma * 2.0,
                }
            )
        scalar_rows.append(
            {
                "run_id": run_id,
                "schedule": "Learned Schedule",
                "mean_acc": 0.9,
                "mean_loss": 0.1,
            }
        )
    cache = tmp_path / "cache"
    cache.mkdir()
    pd.DataFrame(sched_rows).to_parquet(cache / "schedules.parquet", index=False)
    pd.DataFrame(scalar_rows).to_parquet(cache / "scalars.parquet", index=False)
    return cache


@pytest.mark.slow
def test_main_template_mode_writes_category_map_and_constants(tmp_path):
    from symbolic_regression import PySRConfig, main

    cache = _synthetic_cache(tmp_path)
    out_dir = tmp_path / "out"
    conf = PySRConfig(
        cache_dir=str(cache),
        targets=("sigma",),
        datapoint_frequency=2,
        out_dir=str(out_dir),
        niterations=8,
        maxsize=10,
        n_template_params=2,
        procs=1,
    )

    main(conf)

    # main() appends a synthesis slug under out_dir; find it.
    slug_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    assert len(slug_dirs) == 1
    slug = slug_dirs[0]

    cmap = load_category_map(slug / "category_map.json")
    assert len(cmap) == 2  # two conditions

    constants = pd.read_csv(slug / "sigma" / "constants.csv")
    assert len(constants) == 2  # one row per condition
    assert set(CONDITION_KEYS) <= set(constants.columns)
    assert {"category", "p1", "p2"} <= set(constants.columns)
    assert sorted(constants["category"]) == [1, 2]

    # The evaluator loads these three per-target artefacts (symbolic_regression_eval.py).
    target_dir = slug / "sigma"
    assert (target_dir / "model.pkl").exists()
    assert (target_dir / "feature_names.json").read_text()  # non-empty
    equations = pd.read_csv(target_dir / "equations.csv")
    assert "selected" in equations.columns
    assert equations["selected"].sum() >= 1  # exactly one (or PySR's mask) selected
    import json as _json

    assert _json.loads((target_dir / "feature_names.json").read_text()) == [
        "step_norm",
        "category",
    ]
