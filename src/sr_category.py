"""Condition <-> category index for template-mode symbolic regression (ADR 0006).

Template mode fits one schedule shape shared across runs plus per-**condition**
free constants. The *condition* ``(dataset, eps, T, arch_label)`` indexes those
constants; the replicate seeds of a condition collapse into one constant vector.
PySR needs the condition as a 1-indexed integer ``category`` column (Julia
indexing), so this module assigns and persists that mapping.

Kept pandas+stdlib only (no ``pysr``/Julia) so ``symbolic_regression_eval.py``
can rebuild the ``category`` column from ``category_map.json`` without paying the
Julia import. ``symbolic_regression.py`` builds + persists it at fit time; the
evaluator loads + re-applies it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# The grouping key that defines a condition. Order is the JSON record field order.
CONDITION_KEYS: tuple[str, ...] = ("dataset", "eps", "T", "arch_label")

# A category map is an ordered list of condition records; the 1-indexed position
# of a record is its `category` integer (entry 0 -> category 1).
CategoryMap = list[dict]


def template_param_names(n_template_params: int) -> list[str]:
    """The K per-condition constant slot names: ``["p1", "p2", ...]``. Shared by
    the template spec, the read-back, and the ``constants.csv`` columns."""
    return [f"p{k}" for k in range(1, n_template_params + 1)]


def build_category_map(df: pd.DataFrame) -> CategoryMap:
    """Ordered, deterministic condition -> category map from a schedules frame.

    Conditions are sorted by ``CONDITION_KEYS`` so the mapping is independent of
    row order or which seeds are present; entry ``i`` maps to category ``i + 1``.
    """
    conditions = (
        df[list(CONDITION_KEYS)]
        .drop_duplicates()
        .sort_values(list(CONDITION_KEYS))
        .reset_index(drop=True)
    )
    return [
        {key: _jsonable(row[key]) for key in CONDITION_KEYS} for _, row in conditions.iterrows()
    ]


def category_series(df: pd.DataFrame, category_map: CategoryMap) -> pd.Series:
    """1-indexed ``category`` integer for each row of ``df`` via ``category_map``.

    Rows sharing a condition get the same integer; raises if a row's condition is
    absent from the map (the evaluator must reuse the fit-time map).
    """
    lookup = {
        tuple(record[key] for key in CONDITION_KEYS): i + 1 for i, record in enumerate(category_map)
    }
    keys = [
        tuple(_jsonable(v) for v in row)
        for row in df[list(CONDITION_KEYS)].itertuples(index=False, name=None)
    ]
    missing = {k for k in keys if k not in lookup}
    if missing:
        raise KeyError(f"conditions absent from category_map: {sorted(missing)}")
    return pd.Series([lookup[k] for k in keys], index=df.index, name="category")


def build_constants_table(constants: dict[str, object], category_map: CategoryMap) -> pd.DataFrame:
    """Join fitted per-condition constants back onto their condition keys.

    ``constants`` maps each template-parameter name to a length-``n_conditions``
    vector (from :func:`symbolic_regression.extract_template_constants`); entry
    ``i`` belongs to category ``i + 1`` == ``category_map[i]``. Returns one row
    per condition with the ``CONDITION_KEYS``, the 1-indexed ``category``, and a
    column per parameter — ready to persist as ``constants.csv`` (ADR 0006).
    """
    n = len(category_map)
    for name, values in constants.items():
        if len(values) != n:
            raise ValueError(
                f"constant {name!r} has length {len(values)}, expected {n} "
                f"(one per condition in category_map)"
            )
    records = []
    for i, condition in enumerate(category_map):
        row = {**condition, "category": i + 1}
        row.update({name: values[i] for name, values in constants.items()})
        records.append(row)
    return pd.DataFrame(records)


def save_category_map(category_map: CategoryMap, path: str | Path) -> None:
    Path(path).write_text(json.dumps(category_map, indent=2))


def load_category_map(path: str | Path) -> CategoryMap:
    return json.loads(Path(path).read_text())


def _jsonable(value):
    """Coerce a pandas/NumPy scalar to a plain JSON-roundtrippable Python scalar,
    so a built map equals one reloaded from JSON."""
    if isinstance(value, str):
        return value
    if hasattr(value, "item"):  # NumPy scalar
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value) if not isinstance(value, bool) else value
    return value
