# The reported transfer figure is a source-T line plot, not the source × target matrix

ADR 0008's deliverable is a descriptive source × target-dataset **matrix**, and
`transfer_plot.plot_matrix` renders it as a heatmap — 63 source regime-arms × 6 target
regimes, 2860×7306 px. That figure is replaced in the write-up by **small-multiples line
plots over source T**, one figure per arm, with the transfer references drawn as rules in
the same coordinate space. The full grid is retained as the appendix CSV that
`plot_matrix` already writes. A compact source-T × target-T heatmap survives as a
companion.

## Status

accepted (supersedes the *presentation* of ADR 0008's matrix; the computed cells, the
read-off-not-selected rule, and the row schema are all unchanged)

## Why the heatmap does not work

Three independent failures, in increasing order of seriousness.

1. **It does not fit on a page.** 63 rows is not a size problem to be solved with a
   smaller font.
2. **The colour scale encodes the wrong difference.** CheXpert spans 65–71.5% and
   ImageNet-32 spans 1–17%; one shared viridis scale across both renders "CheXpert numbers
   are larger than ImageNet numbers" — a fact about the datasets — while every cell within
   a column collapses to nearly one hue.
3. **The row axis is the variable that does nothing.** Within an arm, source **T** explains
   80–91% of the variance in transfer accuracy while source **dataset** explains 0–1% and
   source ε 2–10%. The matrix is named after, and spatially organised by, source dataset.

Worse than any of these: `plot_matrix` orders rows by `sorted(source_label.unique())`, an
alphabetical sort of a concatenated label string, which scatters source T across the axis.
That is why the structure below was invisible in a figure built specifically to show it.

## What the data is actually shaped like

Matched arm (`sgd-m0.9`), mean accuracy by source T × target T:

| | ImageNet-32 | | | CheXpert | | |
|---|---|---|---|---|---|---|
| source T ↓ / target T → | 2000 | 5000 | 7000 | 2000 | 5000 | 7000 |
| 2000 | **14.13** | 13.19 | 12.78 | **69.89** | **71.19** | **71.41** |
| 3000 | **14.22** | 15.04 | 15.03 | 69.30 | 70.95 | 71.35 |
| 5000 | 13.18 | **16.11** | 16.78 | 68.28 | 70.20 | 70.86 |
| 7000 | 12.43 | 15.92 | **16.98** | 67.55 | 69.51 | 70.28 |

On ImageNet-32 the argmax sits on the **diagonal** — source T matching target T is the
best transfer. On CheXpert there is no diagonal: shorter source T is monotonically better
at every target T. Two different two-dimensional structures in one variable pair, and
neither is a property of the source dataset.

## The layout

**Body figure, one per arm** (the `sgd-m0.0` arm's goes in the appendix):

- Facets: one per target dataset. Independent y per facet — the two dataset scales are 4pp
  and 16pp wide.
- x: source T (ordinal, 4 values). y: transfer accuracy. One line per target T.
- **Source dataset is drawn as marker shape, plotted overlapping.** Their superposition
  *is* the evidence that provenance does not matter; it replaces 63 rows with a visual
  null result. Source ε becomes the within-x scatter, for the same reason.
- Band: **source-regime spread** (CONTEXT.md) — sd across the 8 source regimes sharing that
  source T, computed over regime means. Not decoration: on ImageNet-32 at target T=7000 it
  runs 0.16 (matched) → 0.25 → 0.66 → 0.90 (source T=2000), so near the diagonal a transfer
  is both most accurate *and* most provenance-independent.
- **Two reference rules**: best-of-three (solid, annotated with which mechanism it is) and
  Constant (faint, dashed).

**Companion**: source T × target T heatmap, one small panel per target dataset, 4×3 cells,
diverging colormap centred on that panel's best-of-three reference so hue reads directly as
beats/loses-to-bar. It earns its place because the ImageNet-32 diagonal is a genuinely
two-dimensional pattern that a line plot renders as crossing lines.

## Why two reference rules rather than one

The choice of rule changes the claim, so it is recorded rather than left to the plotting
code. Matched arm, best source T per target regime:

| target | tgt T | learned | Constant | Dynamic | Median | vs Constant | vs best-of-3 | ref sd |
|---|---|---|---|---|---|---|---|---|
| CheXpert | 2000 | 69.89 | 69.50 | 70.01 | 69.12 | +0.39 | −0.13 | 0.73 |
| CheXpert | 5000 | 71.19 | 69.62 | 70.71 | 70.62 | +1.57 | +0.48 | 0.43 |
| CheXpert | 7000 | 71.41 | 69.30 | 71.02 | 71.40 | +2.11 | +0.01 | 0.21 |
| ImageNet-32 | 2000 | 14.22 | 5.97 | 12.27 | 13.07 | +8.25 | +1.16 | 0.45 |
| ImageNet-32 | 5000 | 16.11 | 6.54 | 14.68 | 13.58 | +9.57 | +1.43 | 1.05 |
| ImageNet-32 | 7000 | 16.98 | 5.57 | 15.17 | 12.17 | +11.41 | +1.82 | 0.54 |

Against Constant alone the method wins everywhere by up to +11.4pp; against the best of the
three it wins clearly on ImageNet-32 (2–11× the reference sd) and **ties on CheXpert**
(−0.13 / +0.48 / +0.01, all inside a reference sd of 0.21–0.73). A single best-of-3 rule
hides that Constant sits 6–9pp below the adaptive references on ImageNet-32 — which is
itself the interesting statement, that the *adaptive* baselines are what is hard to beat.
A single Constant rule would overstate the result.

The supportable sentence is therefore: *transferred learned schedules beat a tuned constant
schedule at every target regime, and beat the strongest adaptive baseline on ImageNet-100
while matching it on CheXpert.* "Beats all baselines" is not supportable.

Two further caveats the figure must not be read against: best-of-three over noisy estimates
is upward-biased by roughly 0.4pp at these sds, which makes the CheXpert tie if anything
*pessimistic*; and the best reference is a different mechanism per panel (Dynamic at 4 of 6,
Median at 2 of 6), which is why the rule is annotated rather than merely drawn.

## Considered and rejected

- **Keeping one figure that is both the completeness artifact and the evidence.** Rejected:
  attempting both is what produced a figure that achieves neither. ADR 0008's
  "read off, not selected" is a constraint on *what is computed*, not on what is plotted —
  a figure summarising all cells has selected nothing, provided the full grid ships as CSV.
- **A forest plot over source regimes**, matching the `arch_forest_delta` convention.
  Rejected: it reintroduces the 63-row axis, and CONTEXT.md reserves the forest plot for the
  robustness-across-architectures claim, which this is not.
- **One 2×2 figure with arms as rows and independent y per row.** Rejected as the worst
  option available: after ADR 0021 the arms have different *target* configurations, and
  independent y-axes on stacked facets is precisely how two incomparable quantities are made
  to look comparable. A caption disclaimer does not undo it.
- **A Δ(learned − Constant) figure to put both arms on a common scale.** Not rejected, but
  not adopted as the primary: normalising to each arm's own Constant baseline *is* a real
  common scale, but it changes the quantity from accuracy to improvement-over-baseline, and
  so answers "does mismatch cost you?" rather than "how well does this transfer?" Add it
  only if the text asks the first question — and note ADR 0021 gives up the ability to
  answer it cleanly.

## Consequences

- `plot_matrix` and its CSVs stay as they are; the heatmap becomes an appendix/diagnostic
  artifact rather than the reported figure. The new plots are additional functions in
  `transfer_plot.py`, not a rewrite.
- **Row ordering must become explicit wherever a source axis is drawn.** The alphabetical
  sort of a concatenated label is the specific defect that hid the diagonal, and it will hide
  the next structure too.
- The **source-regime spread** term this figure depends on is defined in CONTEXT.md
  alongside the two spreads it must not be confused with (generalization consistency,
  evaluation noise) — at ~0.1pp and ~0.35pp respectively, both are smaller than it, and all
  three appear in this analysis.
- The stray `eyepacs` cell (1) and the three rows carrying no arm, from a parquet predating
  ADR 0011, are filtered rather than plotted.
- **The `sgd-m0.0` panel's numbers are provisional** until the ADR 0021 re-run. Every
  statement above about the matched arm is not.
