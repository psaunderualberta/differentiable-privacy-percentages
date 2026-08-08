"""Re-evaluating a distilled schedule shape, outside the process that fitted it.

Everything here is numpy/pandas only — no ``pysr``, no Julia — because the two
consumers of a synthesis (``symbolic_regression_eval.py`` and
``transfer_equation.py``) both run in fresh processes where a template model
cannot be reconstructed: its ``combine`` is an anonymous Julia closure, so neither
the pickled ``julia_expression`` nor ``from_file`` survives the trip (docs/adr/0006).
Predictions are evaluated from the persisted equation *strings* instead.

Two things live here rather than in the evaluator because the synthesis itself
needs them too:

* **The fit space.** σ and C are strictly positive and span 11–65× within one run,
  so squared error in natural units barely constrains the low end of a curve and
  lets the search propose shapes that go negative or zero. Fitting ``log`` of the
  target makes the error relative and positivity structural. The transform is
  therefore part of the fit, and every prediction has to be mapped back
  (:class:`InvertingPredictor`) before anyone reads it as a σ or a C. See ADR 0025.

* **The dense tripwire** (:func:`front_health`). A fit only ever sees the
  ``points_per_run`` samples it was given; the pole that broke equation transfer
  hid between two of them. Sweeping the finished front on a grid far denser than
  any target T catches that at synthesis time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET_TRANSFORMS: tuple[str, ...] = ("identity", "log")

# Operators symbolic_regression.py allows the search (its builder_kwargs); they map
# 1:1 onto numpy so template equations can be re-evaluated without a Julia backend.
_TEMPLATE_FUNCS = {"sqrt": np.sqrt, "exp": np.exp, "log": np.log}


# ---------------------------------------------------------------------------
# Fit space
# ---------------------------------------------------------------------------


def _validate_transform(transform: str) -> str:
    if transform not in TARGET_TRANSFORMS:
        raise ValueError(
            f"unknown target_transform {transform!r}; expected one of {TARGET_TRANSFORMS}"
        )
    return transform


def to_fit_space(y: np.ndarray, transform: str) -> np.ndarray:
    """Map a target column into the space the search minimises squared error in."""
    _validate_transform(transform)
    y = np.asarray(y, dtype=float)
    if transform == "identity":
        return y
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(y)


def from_fit_space(y: np.ndarray, transform: str) -> np.ndarray:
    """Map a prediction back to natural units (a σ or a C).

    Overflow is left to produce ``inf`` rather than raising: callers already treat
    non-finite predictions as a rejected shape, and an exception here would abort a
    whole-front sweep on one bad row.
    """
    _validate_transform(transform)
    y = np.asarray(y, dtype=float)
    if transform == "identity":
        return y
    with np.errstate(over="ignore"):
        return np.exp(y)


def fittable_mask(y: np.ndarray, transform: str) -> np.ndarray:
    """Which rows the fit space can represent.

    ``log`` has no value at y ≤ 0, and a NaN target is not a datum in any space.
    Such rows must be dropped rather than passed through: PySR's validity check is
    all-or-nothing over the array, so a single NaN target would make every candidate
    equation score ``Inf`` and the search would return nothing.
    """
    _validate_transform(transform)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if transform == "identity":
        return finite
    return finite & (y > 0.0)


# ---------------------------------------------------------------------------
# Template equations
# ---------------------------------------------------------------------------


@dataclass
class TemplateEquation:
    code: object  # compiled Python expression over a0 (step_norm) and a1..aK (constants)
    param_arrays: dict[int, np.ndarray]  # slot j (= template param pj) → per-condition array


def parse_template_equation(equation: str) -> TemplateEquation:
    """Parse one ``f = …; p1 = […]; p2 = […]`` template-equation row into a callable.

    The combine is ``f(step_norm, p1[category], …)``, so in ``f`` the argument
    refs are ``#1`` = step_norm and ``#k`` = the (k-1)-th template constant. Each
    ``pj`` array carries that constant for every 1-indexed condition. Maps to
    Python names ``a0`` (step_norm) and ``aj`` (= ``pj[category]``).
    """
    parts = [p.strip() for p in equation.split(";") if p.strip()]
    expr = parts[0].split("=", 1)[1].strip()
    param_arrays: dict[int, np.ndarray] = {}
    for p in parts[1:]:
        name, rhs = p.split("=", 1)
        slot = int(name.strip().lstrip("p"))
        param_arrays[slot] = np.array(
            [float(x) for x in rhs.strip().strip("[]").split(",")], dtype=float
        )
    py_expr = re.sub(r"#(\d+)", lambda m: f"a{int(m.group(1)) - 1}", expr)
    return TemplateEquation(compile(py_expr, "<template-eq>", "eval"), param_arrays)


class TemplatePredictor:
    """Julia-free stand-in for a template-mode ``PySRRegressor`` (predict only).

    Predictions are evaluated from the persisted equation strings, which embed both
    the universal shape ``f`` and the fitted per-condition constants. The ``index``
    argument mirrors ``PySRRegressor.predict``: row label in ``equations``.

    Values come back in whatever space the synthesis was *fitted* in; wrap this in
    :class:`InvertingPredictor` to read them as σ or C.
    """

    def __init__(self, equations: pd.DataFrame, feature_names: list[str]):
        self.feature_names = list(feature_names)
        self._eqs = {
            int(i): parse_template_equation(row["equation"]) for i, row in equations.iterrows()
        }
        sel = equations.index[equations["selected"]]
        self._selected = int(sel[0]) if len(sel) else int(equations.index[-1])

    def predict(self, X: np.ndarray, index: int | None = None) -> np.ndarray:
        eq = self._eqs[self._selected if index is None else int(index)]
        step = np.asarray(X[:, 0], dtype=float)
        category = np.asarray(X[:, 1]).astype(int)  # 1-indexed condition
        ns: dict[str, object] = {**_TEMPLATE_FUNCS, "a0": step}
        for slot, arr in eq.param_arrays.items():
            ns[f"a{slot}"] = arr[category - 1]
        with np.errstate(all="ignore"):
            out = eval(eq.code, {"__builtins__": {}}, ns)
        return np.broadcast_to(np.asarray(out, dtype=float), step.shape).astype(float)


class InvertingPredictor:
    """A predictor whose output is mapped back out of the fit space.

    The inversion sits at the predictor boundary — not at each call site — so that
    no consumer can read a log-space number as a noise scale. ``inner`` is either a
    :class:`TemplatePredictor` or a pickled ``PySRRegressor`` (pooled scalar fits).
    """

    def __init__(self, inner: object, transform: str):
        self.inner = inner
        self.transform = _validate_transform(transform)
        self.feature_names = getattr(inner, "feature_names", None)

    def predict(self, X: np.ndarray, index: int | None = None) -> np.ndarray:
        raw = self.inner.predict(X) if index is None else self.inner.predict(X, index=index)
        return from_fit_space(raw, self.transform)


# ---------------------------------------------------------------------------
# Dense-grid front health
# ---------------------------------------------------------------------------

_HEALTH_GRID_POINTS = 20_000
"""Well above the largest transfer target T (7000), so the sweep is strictly finer
than any grid a distilled equation will be evaluated on downstream."""

BLOWUP_FACTOR = 1e3
"""How far above the largest fitted target a shape may reach before it is called
implausible. Generous by design — this is a tripwire for shapes that have left the data
behind entirely, not a judgement on extrapolation. It is the only criterion that catches a
*near* miss of a pole: at T=2000 the f152229a clip equation lands 2.2e-4 from its pole and
returns 2.9e13, which is finite and positive and so passes every other check."""


def plausible_bound(observed: np.ndarray) -> float:
    """The largest magnitude a shape may reach, from the targets actually fitted."""
    observed = np.asarray(observed, dtype=float)
    finite = observed[np.isfinite(observed)]
    return BLOWUP_FACTOR * float(np.max(np.abs(finite)))


def dense_step_grid(n_points: int) -> np.ndarray:
    """``step_norm`` values for a length-``n_points`` run.

    Matches the fit's convention — ``step_norm = inner_step / T``
    (compile_results_fetch.py) — so the grid reaches ``1 - 1/n`` and never 1. A grid
    including 1.0 would probe a point no run ever visits.
    """
    return np.arange(n_points, dtype=float) / n_points


def front_health(
    predictor: object,
    equations: pd.DataFrame,
    n_conditions: int,
    n_points: int = _HEALTH_GRID_POINTS,
    max_plausible: float | None = None,
) -> pd.DataFrame:
    """Sweep every front row, over every condition, on a grid denser than any target.

    Returns one row per equation with ``n_nonfinite``, ``n_nonpositive``, ``max_abs``
    and a ``healthy`` verdict. A shape is unhealthy if it is non-finite anywhere (a
    pole), reaches zero or below (not a σ or a C), or — when ``max_plausible`` is
    given — exceeds it. That last bound is the one PySR structurally cannot apply:
    its ``is_valid_array`` is a sum-based NaN/Inf test, so a *finite* 1e15 spike
    passes every check the search makes.

    ``predictor`` must already invert the fit space, since all three criteria are
    statements about σ and C in natural units.
    """
    grid = dense_step_grid(n_points)
    rows: list[dict] = []
    for i in equations.index:
        n_nonfinite = 0
        n_nonpositive = 0
        max_abs = 0.0
        for category in range(1, n_conditions + 1):
            X = np.column_stack([grid, np.full(n_points, float(category))])
            with np.errstate(all="ignore"):
                pred = np.asarray(predictor.predict(X, index=int(i)), dtype=float)
            finite = np.isfinite(pred)
            n_nonfinite += int((~finite).sum())
            n_nonpositive += int((pred[finite] <= 0.0).sum())
            if finite.any():
                max_abs = max(max_abs, float(np.max(np.abs(pred[finite]))))
        healthy = (
            n_nonfinite == 0
            and n_nonpositive == 0
            and (max_plausible is None or max_abs <= max_plausible)
        )
        row = {
            "index": int(i),
            "n_nonfinite": n_nonfinite,
            "n_nonpositive": n_nonpositive,
            "max_abs": max_abs,
            "healthy": healthy,
        }
        for col in ("complexity", "loss", "selected"):
            if col in equations.columns:
                row[col] = equations.loc[i, col]
        rows.append(row)
    return pd.DataFrame(rows)
