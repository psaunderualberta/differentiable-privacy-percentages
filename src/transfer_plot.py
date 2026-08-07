"""Transfer-evaluation assembler (ADR 0008).

Reads whatever producer cells are present under ``cache/transfer/`` (curve,
reference, and equation if it exists) and builds the descriptive source × target
matrix. The matrix is **read off, not selected**: every source policy and every
seed is kept, and each cell reports the spread across its transferred seeds as the
regime's generalization consistency. Mirrors the ``compile_results_fetch`` /
``compile_results_plot`` split — this is the plot side.
"""

import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro

from util.transfer import assemble_transfer

# A matrix cell is keyed by the source REGIME-ARM and the target regime (ADR 0018) —
# not by source_id. The regime is the row unit of analysis, so pooling its policies is
# what makes the printed ± generalization consistency rather than evaluation noise.
_CELL_KEYS = [
    "producer",
    "source_dataset",
    "source_eps",
    "source_T",
    "source_arch",
    "source_arm",
    "source_label",
    "target",
    "target_eps",
    "target_T",
]

# The per-policy view: the same cell split by source policy. Its spread is across one
# policy's evaluation reps, which is a different quantity (CONTEXT.md).
_POLICY_KEYS = ["producer", "source_id", "target", "target_eps", "target_T"]

# The cell identity shared across producers: a curve cell and an equation cell
# overlay iff they agree on the source REGIME and the target (ADR 0015). NOT
# source_id — curve's is a W&B run id and equation's is a condition slug, so they can
# never compare equal; and a distilled condition has no per-seed identity to match
# on anyway. target_arch is excluded: ADR 0007 derives it from the target dataset, so
# it adds nothing but a chance of spurious label mismatch. source_arm IS included
# (ADR 0018): a synthesis is scoped to one arm (ADR 0016), so the other arm's curve
# cells have no equation counterpart and must simply not overlay — without it they
# would be compared against a closed form distilled from different shapes entirely.
_OVERLAY_KEYS = [
    "source_dataset",
    "source_eps",
    "source_T",
    "source_arch",
    "source_arm",
    "target",
    "target_eps",
    "target_T",
]


def _collapse(
    assembled: pd.DataFrame, keys: list[str], unit_key: str | None = None
) -> pd.DataFrame:
    """Mean / spread / count of ``accuracy`` over ``keys``.

    Without ``unit_key`` the spread is over the raw rows, which for a per-policy
    grouping is that policy's evaluation noise.

    With ``unit_key`` the rows are first averaged within each unit and the spread is
    taken over those unit *means*. This matters: a cell pooling P policies of R reps
    each has raw-row variance ``sigma_policy^2 + sigma_eval^2``, so a spread over raw
    rows reports evaluation noise as part of the regime's generalization consistency.
    Averaging first leaves only ``sigma_policy^2 + sigma_eval^2 / R``. With the measured
    per-rep sd of ~0.35pp and ``num_reps=3``, that is the difference between an honest
    consistency figure and one inflated by a third or more.

    A group holding a single unit has no between-unit spread to report — a native
    reference and a distilled condition are both one unit by construction
    (CONTEXT.md) — so it falls back to its rep spread rather than reporting a
    spurious zero. ``spread_of`` names which quantity each row carries, because
    CONTEXT.md requires every ± to say which of the two it is.
    """
    if unit_key is None:
        collapsed = (
            assembled.groupby(keys, dropna=False)["accuracy"]
            .agg(mean_acc="mean", spread=lambda s: float(s.std(ddof=0)), n="count")
            .reset_index()
        )
        collapsed["spread_of"] = "reps"
        collapsed["n_policies"] = 1
        return collapsed

    unit_means = assembled.groupby([*keys, unit_key], dropna=False)["accuracy"].mean().reset_index()
    across_units = (
        unit_means.groupby(keys, dropna=False)["accuracy"]
        .agg(mean_acc="mean", spread=lambda s: float(s.std(ddof=0)), n_policies="count")
        .reset_index()
    )
    over_reps = (
        assembled.groupby(keys, dropna=False)["accuracy"]
        .agg(rep_spread=lambda s: float(s.std(ddof=0)), n="count")
        .reset_index()
    )
    collapsed = across_units.merge(over_reps, on=keys, how="left")
    single = collapsed["n_policies"] <= 1
    collapsed["spread"] = collapsed["spread"].where(~single, collapsed["rep_spread"])
    collapsed["spread_of"] = np.where(single, "reps", "policies")
    return collapsed.drop(columns=["rep_spread"])


def source_labels(assembled: pd.DataFrame) -> pd.Series:
    """The row identity each transfer row pools into.

    Curve rows pool by source **regime-arm**: the regime's seed-policies are what a
    cell averages, so its label names the regime, not one of the run ids inside it.
    Every other producer's ``source_id`` already *is* its row unit — a distilled
    condition has no seeds (ADR 0015), and a native reference is a mechanism rather
    than a regime (CONTEXT.md), so the three references must not merge just because
    their source provenance all mirrors the same target.
    """
    regime = (
        assembled["source_dataset"].astype(str)
        + " ε="
        + assembled["source_eps"].map(lambda e: f"{float(e):g}")
        + " T="
        + assembled["source_T"].map(lambda t: str(int(t)))
        + " "
        + assembled["source_arch"].astype(str)
        + " "
        + assembled["source_arm"].astype(str)
    ).str.strip()
    return regime.where(assembled["producer"] == "curve", assembled["source_id"].astype(str))


def transfer_matrix(assembled: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-seed transfer rows to one row per source regime-arm × target.

    Read off, not selected: every regime-arm is kept, and its cell reports the mean
    accuracy plus the spread over **all** of its rows — every policy in the regime
    crossed with every evaluation rep. That spread is the regime's *generalization
    consistency* (CONTEXT.md).

    Grouping on ``source_id`` instead — as this did before ADR 0018 — makes the ±
    the spread across one policy's reps, i.e. DP-SGD's own run-to-run noise, which
    is a different and much smaller quantity than the one the matrix claims to
    report. That view is still available as :func:`policy_matrix`.
    """
    labelled = assembled.assign(source_label=source_labels(assembled))
    matrix = _collapse(labelled, _CELL_KEYS, unit_key="source_id")
    return matrix.merge(tuning_labels(labelled), on=_CELL_KEYS, how="left")


def tuning_labels(labelled: pd.DataFrame) -> pd.DataFrame:
    """One ``tuned`` label per cell: the knobs its rows were evaluated under (ADR 0024).

    A cell's accuracy is now the accuracy of a *tuned* schedule, so the matrix has to
    carry which knobs won it. Without this a heatmap silently compares schedules
    adapted by different amounts, and "which scale did each target prefer" — a result
    in its own right — is lost in the collapse.

    A cell whose rows disagree is labelled ``"mixed"`` rather than resolved to one of
    them. Stage A is tuned per (target × arm) and *shared* by every source in a cell,
    so agreement is an invariant; if it ever breaks, the cell mean is a mean over
    different schedules and the label should say so rather than quietly pick a winner.

    Rows carry the pre-rendered ``tuned_constants`` string from
    ``util.transfer.describe_knobs``, so the two cannot drift apart.

    Cells written before ADR 0024 carry neither column and read as untuned. That is
    the right default rather than a null: the reference cells already on disk are not
    invalidated by tuned transfer (a native reference was always tuned on its target),
    so the assembler has to keep reading them.
    """
    labelled = labelled.copy()
    for column, identity in (("tuned_scale", 1.0), ("tuned_constants", "")):
        if column not in labelled:
            labelled[column] = identity
    described = labelled.assign(
        tuned=(
            "scale="
            + labelled["tuned_scale"].map(lambda s: f"{float(s):g}")
            + labelled["tuned_constants"].astype(str).map(lambda c: f" {c}" if c else "")
        )
    )
    grouped = described.groupby(_CELL_KEYS, dropna=False)["tuned"].agg(
        lambda values: next(iter(set(values))) if len(set(values)) == 1 else "mixed"
    )
    return grouped.reset_index()


def policy_matrix(assembled: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-seed transfer rows to one row per source *policy* × target.

    The diagnostic companion to :func:`transfer_matrix`: same cells, split by source
    policy, so each row's spread is that policy's **evaluation noise** rather than
    its regime's generalization consistency. Kept distinct because CONTEXT.md
    requires every ± to name which of the two it is.
    """
    return _collapse(assembled, _POLICY_KEYS)


def nearest_source(assembled: pd.DataFrame, target_eps: float, target_T: int) -> str:
    """The ``source_id`` whose source regime is nearest the target in (ε, T).

    Distance is *relative* in each axis — ``|Δε|/target_eps`` and ``|ΔT|/target_T``
    combined in quadrature — so ε and T contribute comparably despite their very
    different scales. Ties break deterministically on ``source_id`` (sorted first).
    """
    sources = (
        assembled[["source_id", "source_eps", "source_T"]]
        .drop_duplicates()
        .sort_values("source_id")
    )
    d_eps = (sources["source_eps"] - target_eps) / target_eps
    d_T = (sources["source_T"] - target_T) / target_T
    dist = (d_eps**2 + d_T**2) ** 0.5
    return str(sources.iloc[dist.to_numpy().argmin()]["source_id"])


def _cell_keys(df: pd.DataFrame) -> set[tuple]:
    """The distinct overlay-cell keys present in one producer's frame."""
    distinct = df[_OVERLAY_KEYS].drop_duplicates()
    return {tuple(row) for row in distinct.itertuples(index=False, name=None)}


def overlay_cells(producers: dict[str, pd.DataFrame]) -> list[tuple]:
    """Cells for which BOTH the curve and equation producers have a record.

    A presence-check join (ADR 0008), keyed on the source *regime* rather than the
    source policy (ADR 0015): the overlay is drawn only where both producers
    evaluated the same source regime × target. Returns the sorted intersection of
    their cell keys; empty if either producer is absent. Reference cells never
    participate.
    """
    curve = producers.get("curve")
    equation = producers.get("equation")
    if curve is None or equation is None:
        return []
    return sorted(_cell_keys(curve) & _cell_keys(equation))


def overlay_stats(assembled: pd.DataFrame, cell: tuple) -> tuple[float, float]:
    """One producer's mean and spread for an overlay cell, pooled over the regime.

    The overlay joins on the source regime (ADR 0015), so the curve side — whose
    row unit is one seed's policy — is pooled across *every* policy in the regime
    before it is compared with the equation side's single distilled condition.
    Consequence: the curve error bar mixes seed noise with across-policy spread
    while the equation's is seed noise alone, so the two bars are not like-for-like
    and must not be read as a significance test.
    """
    mask = pd.Series(True, index=assembled.index)
    for key, value in zip(_OVERLAY_KEYS, cell):
        mask &= assembled[key] == value
    accuracy = assembled.loc[mask, "accuracy"]
    return float(accuracy.mean()), float(accuracy.std(ddof=0))


# ---------------------------------------------------------------------------
# The reported figure: transfer accuracy as a function of source T (ADR 0022)
# ---------------------------------------------------------------------------

# What a source-T profile is grouped by. `source_arm` is in the key rather than
# filtered beforehand so that pooling the arms is structurally impossible: after
# ADR 0021 the two arms differ in their *target* configuration, so their accuracies
# are not even on a common scale (CONTEXT.md: split per arm, never pooled).
_PROFILE_KEYS = ["source_arm", "target", "target_eps", "target_T", "source_T"]


def source_t_profile(matrix: pd.DataFrame, split: tuple[str, ...] = ()) -> pd.DataFrame:
    """Collapse the cell matrix to one row per source T, per target regime, per arm.

    The line and band of ADR 0022's figure. ``mean_acc`` averages the source regimes
    sharing that source T and ``spread`` is the sd **across those regime means** — the
    *source-regime spread* of CONTEXT.md. That is deliberately not the mean of the
    cells' own ± (generalization consistency, ~0.1pp) nor the rep spread (evaluation
    noise, ~0.35pp): it is the quantity that answers "does source provenance matter",
    and at 0.16–0.90pp it is the largest of the three.

    ``split`` adds further grouping columns — ``("source_dataset",)`` gives the
    per-provenance series the figure overplots as marker shapes, whose superposition
    is the visual null result.

    Sorted by ``source_T`` numerically. ADR 0022 records that the alphabetical sort of
    a concatenated label is what hid the source-T diagonal in ``plot_matrix``, so any
    source axis drawn from this frame is explicitly ordered.
    """
    keys = [*_PROFILE_KEYS, *split]
    profile = (
        matrix.groupby(keys, dropna=False)["mean_acc"]
        .agg(mean_acc="mean", spread=lambda s: float(s.std(ddof=0)), n_regimes="count")
        .reset_index()
    )
    return profile.sort_values(keys).reset_index(drop=True)


# Targets excluded from every reported figure. EyePACS floors at 73.982% even
# non-privately, so it separates no two schedules and its column measures nothing
# (ADR 0020). One stray cell survives in the cache from before the decision.
_DROPPED_TARGETS = frozenset({"eyepacs"})


def scope_to_arm(matrix: pd.DataFrame, arm: str, legacy_arm: str = "") -> pd.DataFrame:
    """Narrow a cell matrix to one arm's figure (ADR 0022), dropping what cannot be read.

    Three filters, all of them independent of any accuracy number:

    * **the arm.** One figure per arm and never a pooled one — after ADR 0021 the arms
      differ in their target configuration, so pooling them is a units error.
    * **dropped targets.** See :data:`_DROPPED_TARGETS`.
    * **rows carrying no arm.** From a parquet predating ADR 0011. Nothing records which
      momentum they were learned under, and NaN never compares equal to itself, so left
      in they would neither group nor join — they would draw as a phantom series.

    ``legacy_arm`` is the one exception, and it applies to the reference cells written
    before ADR 0021: those carry no arm, but the bug ADR 0021 fixes pinned *every*
    target to momentum 0.9, so they are provably that arm's references. Naming the arm
    they actually ran at lets the already-matched half of the batch be plotted before
    the re-run lands; the other arm gets no bar rather than a borrowed one.
    """
    kept = matrix[~matrix["target"].isin(_DROPPED_TARGETS)].copy()
    armed = kept["source_arm"].fillna("").astype(str)
    if legacy_arm:
        armed = armed.where(armed != "", legacy_arm)
    return kept[armed == arm].reset_index(drop=True)


# The reference whose rule is drawn on its own, separately from the best-of-three.
_CONSTANT_REFERENCE = "Constant"

# A reference cell is keyed by its target regime and arm; its `source_label` is the
# mechanism (CONTEXT.md: a reference is a mechanism, not a regime). The arm is in the
# key for the same reason as in `_PROFILE_KEYS` — ADR 0021 makes a reference native to
# one target momentum, so one arm's bar is not the other's.
_RULE_KEYS = ["source_arm", "target", "target_eps", "target_T"]


def reference_rules(reference_matrix: pd.DataFrame) -> pd.DataFrame:
    """The two reference bars each panel of ADR 0022's figure is drawn against.

    Per target regime and arm: ``best_acc`` / ``best_mechanism`` — the strongest of the
    three native references, *named* — and ``constant_acc``, the tuned constant
    schedule. ``ref_spread`` is the sd across the three mechanisms, the scale against
    which a win or a tie is read.

    Two rules rather than one because the choice changes the claim (ADR 0022). Against
    Constant alone the transferred schedules win everywhere by up to +11.4pp; against
    the best of three they win clearly on ImageNet-32 and tie on CheXpert. Collapsing
    to best-of-three hides that Constant sits 6–9pp below the adaptive references —
    that the *adaptive* baselines are the hard ones is the interesting statement — and
    collapsing to Constant alone would overstate the result.

    The best-of-three is a maximum over noisy estimates and is therefore upward-biased
    by roughly 0.4pp at these spreads; that makes a tie against it pessimistic, not
    optimistic. Naming the mechanism is not decoration either: it is a different one in
    4 of 6 panels versus 2 of 6.

    An arm with no references yet — the real state of ``sgd-m0.0`` between the ADR 0021
    fix and its re-run — returns an empty frame *with the columns present*, so its panel
    draws without bars instead of taking the other arm's finished figure down with it.
    """
    columns = [*_RULE_KEYS, "best_mechanism", "best_acc", "constant_acc", "ref_spread"]
    if reference_matrix.empty:
        return pd.DataFrame(columns=[*columns, "n_references"])

    grouped = reference_matrix.groupby(_RULE_KEYS, dropna=False)
    rows = []
    for key, group in grouped:
        best = group.loc[group["mean_acc"].idxmax()]
        constant = group[group["source_label"] == _CONSTANT_REFERENCE]["mean_acc"]
        rows.append(
            {
                **dict(zip(_RULE_KEYS, key)),
                "best_mechanism": str(best["source_label"]),
                "best_acc": float(best["mean_acc"]),
                "constant_acc": float(constant.iloc[0]) if len(constant) else float("nan"),
                "ref_spread": float(group["mean_acc"].std(ddof=0)),
                "n_references": len(group),
            }
        )
    return pd.DataFrame(rows).sort_values(_RULE_KEYS).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rendering + IO (integration glue; exercised end-to-end, not unit-tested)
# ---------------------------------------------------------------------------


def _target_label(target: str, eps: float, T: int) -> str:
    """Column label for one target regime."""
    return f"{target}\nε={eps:g} T={int(T)}"


def load_producers(cache_root: Path | str) -> dict[str, pd.DataFrame]:
    """Assemble every producer that has written cells under ``cache/transfer/``.

    Discovers the producer subdirectories (``curve``, ``reference``, ``equation``,
    or any future writer of the shared schema) and assembles each into one frame.
    Whatever is present is returned — the assembler renders what it finds.
    """
    root = Path(cache_root) / "transfer"
    producers: dict[str, pd.DataFrame] = {}
    if not root.is_dir():
        return producers
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if not any(sub.glob("*.parquet")):
            continue
        producers[sub.name] = assemble_transfer(sub.name, cache_root)
    return producers


def plot_matrix(
    assembled: pd.DataFrame,
    producer: str,
    out_stem: Path,
    by_policy: bool = False,
) -> None:
    """Descriptive source × target accuracy matrix for one producer.

    Rows are source regime-arms, columns are target regimes; each cell shows mean
    transfer accuracy with the **generalization-consistency** spread beneath it —
    the spread across the regime's source policy *means*, so it is not inflated by
    each policy's own evaluation noise (see ``_collapse``). Every column is
    annotated (in its title) with the source nearest it in (ε, T) — read off, not
    selected.

    ``by_policy`` draws the diagnostic companion instead: one row per source policy,
    whose ± is that policy's **evaluation noise**. The two are different quantities
    and the title says which is being shown.
    """
    if assembled.empty:
        print(f"  [skip] no cells for producer={producer}")
        return

    if by_policy:
        matrix = policy_matrix(assembled).rename(columns={"source_id": "source_label"})
        spread_name = "policy evaluation noise"
    else:
        matrix = transfer_matrix(assembled)
        spread_name = "regime generalization consistency"
    matrix = matrix[matrix["producer"] == producer]
    if matrix.empty:
        print(f"  [skip] no cells for producer={producer}")
        return

    matrix = matrix.assign(
        _col=[
            _target_label(t, e, tt)
            for t, e, tt in zip(matrix["target"], matrix["target_eps"], matrix["target_T"])
        ]
    )
    sources = sorted(matrix["source_label"].unique())
    columns = sorted(matrix["_col"].unique())
    row_of = {s: i for i, s in enumerate(sources)}
    col_of = {c: j for j, c in enumerate(columns)}

    grid = np.full((len(sources), len(columns)), np.nan)
    text = np.full((len(sources), len(columns)), "", dtype=object)
    for _, r in matrix.iterrows():
        i, j = row_of[r["source_label"]], col_of[r["_col"]]
        grid[i, j] = r["mean_acc"]
        text[i, j] = f"{r['mean_acc']:.3f}\n±{r['spread']:.3f}"

    fig, ax = plt.subplots(figsize=(1.7 * len(columns) + 2.5, 0.55 * len(sources) + 2.0))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="mean transfer accuracy", fraction=0.046, pad=0.04)

    # Annotate each target column with the source nearest it in (ε, T).
    col_titles = []
    for c in columns:
        cell = matrix[matrix["_col"] == c].iloc[0]
        near = nearest_source(assembled, float(cell["target_eps"]), int(cell["target_T"]))
        col_titles.append(f"{c}\n[near: {near}]")

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(col_titles, fontsize=7, rotation=30, ha="right")
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels(sources, fontsize=7)
    ax.set_ylabel("source policy" if by_policy else "source regime-arm")
    for i in range(len(sources)):
        for j in range(len(columns)):
            if text[i, j]:
                ax.text(j, i, text[i, j], ha="center", va="center", fontsize=6, color="white")
    ax.set_title(f"Transfer matrix — {producer} (mean ± {spread_name})", fontsize=10)
    fig.tight_layout()
    _save(fig, out_stem)


_TARGET_TITLES = {"imagenet": "ImageNet-100 (32x32)", "chexpert": "CheXpert", "eyepacs": "EyePACS"}

# Target T is an *ordered* quantity, so its three series are steps of one hue rather
# than arbitrary categorical colours — light to dark reads as short to long without
# consulting the legend. Single-hue by construction, so the ordering survives CVD.
_TARGET_T_COLOURS = ["#9ecae1", "#4292c6", "#08306b"]

# Source dataset is the null-result axis: same colour, different marker, drawn on top
# of each other so their superposition IS the evidence that provenance does not matter.
_SOURCE_MARKERS = ["o", "s", "^", "D"]

_INK = "#222222"
_MUTED = "#777777"


def _facet_targets(profile: pd.DataFrame) -> list[str]:
    """Target datasets in a stable, reported order (CheXpert first — ADR 0020)."""
    order = [t for t in ("chexpert", "imagenet") if t in set(profile["target"])]
    return order + sorted(set(profile["target"]) - set(order))


def plot_source_t(
    curve: pd.DataFrame,
    reference: pd.DataFrame,
    arm: str,
    out_stem: Path,
    legacy_reference_arm: str = "",
) -> None:
    """ADR 0022's reported figure: transfer accuracy against **source T**, one arm.

    Small multiples, one facet per target dataset with independent y — CheXpert spans
    ~4pp and ImageNet-32 ~16pp, and a shared scale would render "CheXpert numbers are
    larger" (a fact about the datasets) while collapsing every within-panel difference
    to one hue. That is the failure that retired the heatmap.

    Within a facet: x is source T, one line per target T, band = source-regime spread,
    and the source datasets overplotted as marker shapes. Two reference rules per
    target T (ADR 0022) — best-of-three solid and annotated with its mechanism, tuned
    Constant faint and dashed — because which rule is drawn changes the claim.

    Never draws two arms together: the arms differ in their target configuration after
    ADR 0021, so their accuracies are not on a common scale.
    """
    curve = scope_to_arm(curve, arm)
    rules = reference_rules(scope_to_arm(reference, arm, legacy_reference_arm))
    if curve.empty:
        print(f"  [skip] no curve cells for arm={arm}")
        return

    profile = source_t_profile(curve)
    by_dataset = source_t_profile(curve, split=("source_dataset",))
    targets = _facet_targets(profile)
    source_ts = sorted(profile["source_T"].unique())
    x_of = {t: i for i, t in enumerate(source_ts)}
    datasets = sorted(by_dataset["source_dataset"].unique())
    marker_of = dict(zip(datasets, _SOURCE_MARKERS))

    fig, axes = plt.subplots(1, len(targets), figsize=(5.4 * len(targets), 4.4), squeeze=False)
    for ax, target in zip(axes[0], targets):
        panel = profile[profile["target"] == target]
        target_ts = sorted(panel["target_T"].unique())
        colour_of = dict(zip(target_ts, _TARGET_T_COLOURS))

        for target_T in target_ts:
            series = panel[panel["target_T"] == target_T].sort_values("source_T")
            colour = colour_of[target_T]
            x = [x_of[t] for t in series["source_T"]]
            ax.fill_between(
                x,
                series["mean_acc"] - series["spread"],
                series["mean_acc"] + series["spread"],
                color=colour,
                alpha=0.18,
                linewidth=0,
            )
            ax.plot(x, series["mean_acc"], color=colour, linewidth=2.0, zorder=3, label=None)
            # Each source dataset as its own marker at the same x: the overlap is the
            # null result, so they are deliberately not offset apart.
            for dataset in datasets:
                points = by_dataset[
                    (by_dataset["target"] == target)
                    & (by_dataset["target_T"] == target_T)
                    & (by_dataset["source_dataset"] == dataset)
                ].sort_values("source_T")
                ax.plot(
                    [x_of[t] for t in points["source_T"]],
                    points["mean_acc"],
                    linestyle="none",
                    marker=marker_of[dataset],
                    markersize=6,
                    markerfacecolor="none",
                    markeredgecolor=colour,
                    markeredgewidth=1.3,
                    zorder=4,
                )
        _draw_rules(ax, rules, target, target_ts, colour_of, len(source_ts))

        ax.set_xticks(range(len(source_ts)))
        ax.set_xticklabels([str(t) for t in source_ts])
        ax.set_xlim(-0.35, len(source_ts) - 1 + 0.45)
        ax.set_xlabel("source T", color=_INK)
        ax.set_title(_TARGET_TITLES.get(target, target), fontsize=11, color=_INK)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=_MUTED)

    axes[0][0].set_ylabel("transfer accuracy (%)", color=_INK)
    # One shared legend below the panels rather than one per facet: with three lines,
    # two marker shapes and two rule styles, an in-axes legend covers the very data it
    # explains — and every facet's key is identical anyway.
    all_target_ts = sorted(profile["target_T"].unique())
    handles = [
        plt.Line2D([], [], color=_TARGET_T_COLOURS[i], linewidth=2.0, label=f"target T={t}")
        for i, t in enumerate(all_target_ts)
    ]
    handles += [
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker=marker_of[d],
            markerfacecolor="none",
            markeredgecolor=_MUTED,
            markersize=6,
            label=f"source: {d}",
        )
        for d in datasets
    ]
    handles += [
        plt.Line2D([], [], color=_MUTED, linewidth=1.4, label="best-of-3 reference"),
        plt.Line2D(
            [], [], color=_MUTED, linewidth=1.0, linestyle="--", label="tuned Constant reference"
        ),
    ]
    fig.legend(
        handles=handles,
        fontsize=8,
        frameon=False,
        loc="lower center",
        # One row: matplotlib fills a multi-row legend column-major, which interleaves
        # the target-T series with the marker and rule keys.
        ncol=len(handles),
        labelcolor=_INK,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"Transferred schedules vs native references — arm {arm}\n"
        "band = source-regime spread (sd across the source regimes at that source T)",
        fontsize=10,
        color=_INK,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    _save(fig, out_stem)


def _stagger(values: list[float], min_gap: float) -> list[float]:
    """Push label positions apart so near-coincident rules stay separately readable.

    Only the *labels* move; the rules themselves are drawn at their true accuracy. Two
    reference bars can sit 0.5pp apart on a 16pp axis (Dynamic at 15.17 against 14.68),
    which is a real and interesting closeness — the fix is to keep both legible, not to
    drop one.
    """
    placed: list[float] = []
    for value in sorted(values):
        floor = placed[-1] + min_gap if placed else value
        placed.append(max(value, floor))
    return placed


def _draw_rules(ax, rules: pd.DataFrame, target: str, target_ts, colour_of, n_x: int) -> None:
    """The two reference bars per target T, each in its series' colour.

    The best-of-three is annotated with the mechanism that won it: it is Dynamic-DPSGD
    in 4 of the 6 panels and Median in the other 2, so an unlabelled line would quietly
    mean something different from facet to facet.

    Labels sit *inside* the axes, right-aligned. Outside, they overflow into the
    neighbouring facet, and matplotlib does not clip annotations by default.
    """
    panel = rules[rules["target"] == target].set_index("target_T")
    present = [t for t in target_ts if t in panel.index]
    if not present:
        return

    for target_T in present:
        row = panel.loc[target_T]
        ax.axhline(row["best_acc"], color=colour_of[target_T], linewidth=1.4, alpha=0.85, zorder=2)
        if np.isfinite(row["constant_acc"]):
            ax.axhline(
                row["constant_acc"],
                color=colour_of[target_T],
                linewidth=1.0,
                linestyle="--",
                alpha=0.5,
                zorder=2,
            )

    span = float(np.diff(ax.get_ylim()))
    ordered = sorted(present, key=lambda t: float(panel.loc[t, "best_acc"]))
    label_y = _stagger([float(panel.loc[t, "best_acc"]) for t in ordered], 0.045 * span)
    for target_T, y in zip(ordered, label_y):
        ax.annotate(
            f"{panel.loc[target_T, 'best_mechanism']}, T={int(target_T)}",
            xy=(n_x - 1 + 0.40, y),
            fontsize=6,
            color=_MUTED,
            va="center",
            ha="right",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
            annotation_clip=True,
        )


def plot_source_target_heatmap(
    curve: pd.DataFrame,
    reference: pd.DataFrame,
    arm: str,
    out_stem: Path,
    legacy_reference_arm: str = "",
) -> None:
    """Companion to :func:`plot_source_t`: source T × target T, one panel per target.

    Small (4×3 cells) and diverging about the best-of-three reference, so hue reads
    directly as beats-the-bar / loses-to-it rather than as an absolute level.

    It earns its place beside the line plot because the ImageNet-32 structure is
    genuinely two-dimensional — the argmax sits on the source-T = target-T diagonal —
    and a line plot renders a diagonal as crossing lines. CheXpert has no diagonal at
    all (shorter source T is monotonically better), so the two panels showing different
    shapes is itself the finding.

    Cells carry **Δ against that column's own best-of-three reference**, not raw
    accuracy, so hue means exactly "beats the bar" and nothing else. Every target T
    faces a different bar (70.01 / 70.71 / 71.40 on CheXpert), so a single panel-wide
    centre would colour a column by which bar it happened to be near. Subtracting a
    per-column constant leaves the within-column ordering untouched, so the diagonal
    the panel exists to show survives exactly; absolute accuracy is on the line plot
    and in the CSV.
    """
    curve = scope_to_arm(curve, arm)
    rules = reference_rules(scope_to_arm(reference, arm, legacy_reference_arm))
    if curve.empty:
        print(f"  [skip] no curve cells for arm={arm}")
        return

    profile = source_t_profile(curve)
    targets = _facet_targets(profile)
    source_ts = sorted(profile["source_T"].unique())

    fig, axes = plt.subplots(
        1, len(targets), figsize=(3.1 * len(targets) + 1.6, 3.4), squeeze=False
    )
    drawn = 0
    for ax, target in zip(axes[0], targets):
        panel = profile[profile["target"] == target]
        target_ts = sorted(panel["target_T"].unique())
        bar = rules[rules["target"] == target].set_index("target_T")["best_acc"].to_dict()
        grid = np.full((len(source_ts), len(target_ts)), np.nan)
        for _, row in panel.iterrows():
            centre = bar.get(row["target_T"])
            if centre is None:
                continue
            grid[source_ts.index(row["source_T"]), target_ts.index(row["target_T"])] = (
                row["mean_acc"] - centre
            )
        # Every cell is a Δ against a reference, so a target with no reference bar has
        # nothing this panel can express — and an arm whose references have not been
        # swept yet has none at all (the sgd-m0.0 state before the ADR 0021 re-run).
        if not np.isfinite(grid).any():
            print(f"  [skip] no reference bar for target={target}")
            continue
        drawn += 1

        reach = float(np.nanmax(np.abs(grid)))
        im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-reach, vmax=reach, origin="upper")
        for i in range(len(source_ts)):
            for j in range(len(target_ts)):
                if not np.isfinite(grid[i, j]):
                    continue
                # Ink flips to white on the saturated ends of the ramp, where the
                # default dark text is unreadable.
                strong = abs(grid[i, j]) > 0.55 * reach
                ax.text(
                    j,
                    i,
                    f"{grid[i, j]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if strong else _INK,
                )
        ax.set_xticks(range(len(target_ts)))
        ax.set_xticklabels(
            [f"{t}\n(bar {bar[t]:.2f})" if t in bar else str(t) for t in target_ts], fontsize=7
        )
        ax.set_yticks(range(len(source_ts)))
        ax.set_yticklabels([str(t) for t in source_ts], fontsize=8)
        ax.set_xlabel("target T", color=_INK)
        ax.set_title(_TARGET_TITLES.get(target, target), fontsize=9, color=_INK)
        ax.tick_params(colors=_MUTED)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
            "pp above (red) / below (blue) the best-of-3 reference", fontsize=7
        )

    if not drawn:
        plt.close(fig)
        print(f"  [skip] no reference bars at all for arm={arm}; nothing to centre on")
        return

    axes[0][0].set_ylabel("source T", color=_INK)
    fig.suptitle(
        f"Source T × target T, relative to the best-of-3 reference — arm {arm}",
        fontsize=10,
        color=_INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, out_stem)


def plot_overlay(producers: dict[str, pd.DataFrame], out_stem: Path) -> None:
    """Per-regime curve-vs-equation accuracy comparison, only where both exist.

    Draws one grouped point per shared source-regime × target cell (ADR 0015):
    curve-transfer mean±spread beside equation-transfer mean±spread. Skips entirely
    when the two producers share no regime (the compare-only-when-both-exist rule).
    The curve side is pooled across the regime's seed-policies, so its error bar is
    wider by construction than the equation's — see ``overlay_stats``.
    """
    cells = overlay_cells(producers)
    if not cells:
        print("  [skip] no source regimes present in both curve and equation")
        return

    labels = [
        f"{s_ds} ε={s_eps:g} T={int(s_T)} {s_arch} {s_arm}\n→ {_target_label(t, e, tt)}"
        for (s_ds, s_eps, s_T, s_arch, s_arm, t, e, tt) in cells
    ]
    x = np.arange(len(cells))
    curve_mu, curve_sd = zip(*(overlay_stats(producers["curve"], c) for c in cells))
    eqn_mu, eqn_sd = zip(*(overlay_stats(producers["equation"], c) for c in cells))

    fig, ax = plt.subplots(figsize=(1.4 * len(cells) + 2.5, 4.0))
    ax.errorbar(
        x - 0.1, curve_mu, yerr=curve_sd, fmt="o", capsize=3, label="curve", color="#1f77b4"
    )
    ax.errorbar(x + 0.1, eqn_mu, yerr=eqn_sd, fmt="s", capsize=3, label="equation", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_xlim(-0.5, len(cells) - 0.5)
    ax.set_ylabel("transfer accuracy")
    ax.set_title("Curve vs equation transfer (shared source regimes)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, out_stem)


def _save(fig: plt.Figure, stem: Path) -> None:
    # The extension is appended, not substituted: an arm label is `sgd-m0.9`, and
    # `with_suffix` reads its `.9` as an existing suffix and silently writes
    # `..._sgd-m0.pdf` — two arms' figures landing on one filename.
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  ✓ {stem.name}.pdf")


@dataclasses.dataclass
class PlotConfig:
    """Assemble and plot the transfer cells under ``<cache_root>/transfer/``."""

    cache_root: str = "cache"
    out_dir: str = ""
    """Output dir. Defaults to <cache_root>/transfer/plots/."""
    legacy_reference_arm: str = ""
    """Attribute arm-less reference cells to this arm (ADR 0021 compatibility).

    The 18 references written before ADR 0021 carry no arm, but the bug it fixes pinned
    every target to momentum 0.9 — so ``sgd-m0.9`` names the arm they actually ran at
    and makes the matched half of the batch plottable before the re-run lands. Leave
    empty once the re-run has replaced them, at which point every reference records its
    own arm."""


def main(conf: PlotConfig) -> None:
    producers = load_producers(conf.cache_root)
    if not producers:
        raise SystemExit(f"no transfer cells under {Path(conf.cache_root) / 'transfer'}")

    out_root = Path(conf.out_dir) if conf.out_dir else Path(conf.cache_root) / "transfer" / "plots"
    out_root.mkdir(parents=True, exist_ok=True)

    for producer, assembled in producers.items():
        print(f"\n=== producer: {producer} ===")
        plot_matrix(assembled, producer, out_root / f"matrix_{producer}")
        transfer_matrix(assembled).to_csv(out_root / f"matrix_{producer}.csv", index=False)
        # The per-policy companion: same cells, but its ± is evaluation noise.
        plot_matrix(assembled, producer, out_root / f"matrix_{producer}_by_policy", by_policy=True)
        policy_matrix(assembled).to_csv(out_root / f"matrix_{producer}_by_policy.csv", index=False)

    print("\n=== overlay ===")
    plot_overlay(producers, out_root / "curve_vs_equation")

    # The reported figures (ADR 0022). One per arm, never pooled: after ADR 0021 the
    # arms differ in their target configuration, so their accuracies are not on a common
    # scale. The heatmaps above stay as the appendix/completeness artifact.
    curve = transfer_matrix(producers["curve"]) if "curve" in producers else pd.DataFrame()
    reference = (
        transfer_matrix(producers["reference"]) if "reference" in producers else pd.DataFrame()
    )
    if curve.empty:
        print("\n[skip] no curve cells; the reported source-T figure needs them")
        return
    for arm in sorted(set(curve["source_arm"].dropna()) - {""}):
        print(f"\n=== reported figure: arm {arm} ===")
        plot_source_t(
            curve, reference, arm, out_root / f"source_t_{arm}", conf.legacy_reference_arm
        )
        plot_source_target_heatmap(
            curve,
            reference,
            arm,
            out_root / f"source_t_by_target_t_{arm}",
            conf.legacy_reference_arm,
        )
        source_t_profile(scope_to_arm(curve, arm)).to_csv(
            out_root / f"source_t_{arm}.csv", index=False
        )


if __name__ == "__main__":
    main(tyro.cli(PlotConfig))
