# Fit syntheses in log space, with denominators capped at one node

A symbolic-regression synthesis now fits `log σ` / `log C` rather than σ / C, caps what
`/` may divide by at complexity 1, samples 500 points per run rather than 50, and sweeps
its finished Pareto front on a 20,000-point grid before shipping it. Equation transfer
evaluates on `inner_step / T` — the grid the fit actually used — and refuses a shape that
is non-finite, non-positive, or more than 1000× the largest target the synthesis was
fitted on.

Together these make an **interior pole structurally impossible** rather than merely
unlikely, and make **positivity** a property of the representation rather than a hope.

## Status

accepted

## Why

Equation transfer died in `seat_on_budget`:

```
ValueError: seat_on_budget did not bind the budget: spent 40258 of 1.11441e+06
(3.6125%). The bracketed bisection returned c=10
```

The seater was innocent. The `selected` clip equation of synthesis `f152229a` (the
`sgd-m0.9` arm, front index 16, complexity 24) contains

```
-0.004737322 / (#1 - 0.9917169)          # #1 == step_norm
```

— a **pole at step_norm = 0.991719958599793**, nested inside `sqrt(sqrt(exp(...)))`. On
the target grid the clip shape spans 1.6e-20 … 1.3e15, the multiplier σ/C spans
1.9e-16 … 1.6e19, and the GDP sum `Σ exp(1/s²)` is `inf` for every trial `c`, so the
bisection saturates at its bracket ceiling. All 32 categories of `f152229a` are affected.

**Both arms' selected clip equations are unusable, for different reasons.** Running the
new dense sweep (below) over the existing fronts:

| synthesis | target | unusable front rows | selected row | why |
|---|---|---|---|---|
| f152229a | sigma | 2 / 18 | **healthy** | — |
| f152229a | clip | 6 / 17 | **broken** | pole at step_norm 0.9917–0.9918, all 32 conditions, max \|C\| 1.19e75 |
| 7b40dedd | sigma | 9 / 20 | **healthy** | — |
| 7b40dedd | clip | 12 / 17 | **broken** | C ≤ 0 over `step_norm ∈ [0, 0.014]` in 14 of 32 conditions |

`7b40dedd` has no pole — its selected clip equation is finite everywhere and peaks at a
plausible 11.2 against a fitted maximum of 11.5. It is broken the *other* way: it returns
a non-positive clip over a leading band of up to 1.6 % of training, in every
fashion-mnist condition at the two shorter T. A C of 0 kills the gradient and makes the
multiplier σ/C undefined, so this would have failed transfer too — quietly rather than
loudly. Equation transfer was never viable for either arm's clip.

**Nothing upstream could have seen it.** The synthesis ran at `points_per_run=50`, so for
T=2000 the fit only ever evaluated `step_norm ∈ {0, 0.02, …, 0.98}`. The nearest sample
sits 0.0117 from the pole, where the term is a tame `+0.405`. Transfer evaluates the full
T-point grid and lands 2.8e-4 away. Recomputing the front's MSE against
`features_full.parquet` at increasing density shows the equation being selected precisely
*because* the fit was too coarse to see it:

| points_per_run | idx 16 (c=24, the pole) | idx 13 (c=20) | outcome |
|---|---|---|---|
| 50 (as run) | 0.005191 | 0.005862 | pole wins, selected |
| 100 | 0.005178 | 0.005695 | pole wins |
| 200 | 0.1064 | 0.005635 | pole rejected |
| 500 | 1.281e+26 | 0.005607 | pole rejected |

**And no finiteness check in PySR could ever have caught it.** `is_valid_array(x)` is
`is_valid(sum(x))` — a sum-based NaN/Inf test. It catches a division that lands exactly on
the pole; a *finite* 1e15 spike passes every check the search makes.

**Whether a pole is even *hit* depends on the target T**, which is why finiteness is not a
sufficient criterion anywhere. The same f152229a clip equation, evaluated on the grids
transfer actually uses: at **T=2000** it lands 2.2e-4 from the pole and returns a maximum
of **2.885e13** — finite, positive, and ruinous once `seat_on_budget` squares the
multiplier. A guard checking only finiteness and positivity would have passed it. Hence
the third criterion, a magnitude bound taken from the synthesis's own fitted targets
(1000× the largest, `sr_predict.plausible_bound`) — wide enough that no sane shape
approaches it, and cleared by ten orders of magnitude by a near-missed pole.

Two further defect families are independent of the pole and were found while measuring it.
**Non-positivity is everywhere**: σ row 6 reaches **−0.0077** (a negative noise scale), σ
row 7 touches −0, clip rows 5/6/11/16 hit **exactly 0**, and in `7b40dedd` it takes out
the selected clip row as above. **`safe_sqrt` NaNs**: clip rows 14/15 produce NaN — 15
points at 500/run, 669 and 620 on a dense grid. Unlike the pole, those NaN rows are
defeated by density alone; no front row is clean at 500/run and broken at dense.

### Log space

Fitting σ and C directly is a poor use of squared error. Within-curve dynamic range is
11–65× (median 15×), and the selected equation's RMSE is **3.0×** the smallest σ target
and **8.1×** the smallest clip target — absolute MSE places essentially no constraint on
the low end of any curve. Fitting `log` of the target makes the error relative, so the
trough is weighted like the peak, and makes positivity structural: the prediction is
`exp(f)`, which cannot be ≤ 0 whatever `f` does.

This needs **no operator change**. The existing `(+ - * / sqrt exp)` set predicts the
transformed target directly; adding `log` to the vocabulary is a separate question, left
open.

The transform is a property of the fit, so it is a **synthesis-identity field**
(`sr_identity.IDENTITY_FIELDS`): a log-space synthesis gets its own slug and can never
warm-start from natural-units PySR state. It is recorded in `manifest.json` and undone at
the predictor boundary (`sr_predict.InvertingPredictor`), so no consumer can read a
log-space number as a noise scale, and syntheses predating this ADR — which have no such
field, and the oldest of which have no manifest — are read as `identity`.

Note that σ's absolute scale is discarded downstream anyway: `seat_on_budget` is a pure
scalar rescale when the budget binds (`project_inverse_sigmas` is a documented no-op
then), so the final schedule is `(σ = c·σ̂, C = Ĉ)`. Clip's scale *does* matter — it is a
PSAC scale. The two targets have different downstream contracts, and log space serves
both.

### The denominator cap

`constraints={"/": (-1, 1)}` caps what `/` may divide by at one node, leaving the
numerator unlimited.

In template mode `#1` is `step_norm` — the only real variable — and `#2/#3/#4` are
`p1/p2/p3[category]`, per-condition fitted constants. Dividing by those cannot make an
interior pole: they are fixed for a whole run. The dominant σ idiom `expr / #4` is
complexity 1 and is **preserved**. The only `#1`-containing denominators on either front
are σ rows 2, 3, 15–17 and clip rows 3, 16 — all killed by the cap.

The residual case is a bare `/ #1`, a pole at step_norm = 0. It self-eliminates:
step_norm = 0 is present in all 255 runs, so such a candidate evaluates to `Inf`,
`is_valid_array` fails, and it is rejected during search.

**The guarantee:** with denominators capped at complexity 1, `+ - * /const sqrt exp`
generates only continuous, bounded functions on [0,1]. Interior poles stop being something
to detect and become something the search cannot express — strictly stronger than any
probe, which only ever certifies its own grid points.

**Known cost.** The cap removes σ front rows 15–17 (loss 0.00244 / 0.00210 / 0.00208),
which are verified clean (max |pred| ≤ 0.91 on a 20k grid, zero NaNs). The observed front
would drop to row 14 at 0.002734, **+31 % MSE**. That overstates it: a re-run under the
constraint spends complexity 20–24 differently, and in log space `/const` becomes an
additive offset, so division should be less load-bearing than 31 % suggests.

Like the transform, the cap restricts the search space and is therefore an identity field.

### The dense sweep is a tripwire, not the mechanism

`sr_predict.front_health` evaluates every front row, over every condition, on 20,000
points — well above the largest transfer target T (7000) — and flags non-finite,
non-positive, or implausibly large (>1000× the largest fitted target) shapes. It runs
*after* the artefacts are persisted, so a failure leaves the whole front on disk rather
than discarding hours of search, and it raises only when the **selected** row is the
broken one.

It exists to catch a future operator or constraint change that reopens the door. It is not
what makes the current failure impossible — the cap and the log space are.

The failure modes map one-to-one onto the fixes: **poles → the `/` cap**; **negatives and
zeros → log space**; **`safe_sqrt` NaNs → density**.

### The step grid was also wrong

`transfer_equation.evaluate_equation_shape` evaluated on `np.linspace(0, 1, target_T)`,
i.e. `i/(T-1)`, but the fit's `step_norm` is `i/T` (`compile_results_fetch.py`). That
stretches the shape by `T/(T-1)` and reads every fitted feature off by up to one step —
worst at small T. Unrelated to the pole, found alongside it, fixed to
`np.arange(T) / T`.

## Considered and rejected

- **Post-hoc filtering of the front.** It picks among equations the search was already
  misled into producing, and leaves the search itself uncorrected. The whole point is that
  the coarse fit *preferred* the pole (table above).
- **An in-loss probe grid via `loss_function_expression`.** Smoke-tested and working, but
  wrong on two counts. A 32-condition × 4096-point probe is 131,072 rows — **2,600× the
  rows per loss call**. And pole detection needs probe density ≥ the largest target T
  (7000): at spacing 1/500 the offending term evaluates to an unremarkable ≈2.4, and only
  at 1/4096 does it reach ≈38 and overflow. The `constraints` route gives a stronger
  guarantee for one config line.
- **Self-normalised TV (`TV/range`) as a pole detector.** Does not discriminate: the pole
  scores exactly 2.0, identical to a clean single-humped shape, because it inflates range
  as much as it inflates total variation. Only the data-referenced `TV_pred/TV_data` form
  separates them (legitimate rows 0.86–1.25, pole `inf`). Moot under the cap.
- **Fitting the full dataset instead of subsampling.** Mechanically fine and cheaper than
  the 85× it looks like (only `finalize_costs` is O(n)), but it re-breaks ADR 0016's
  per-run balance: T-share goes 25.1/24.7/25.1/25.1 at 50/run to 11.8/17.4/29.5/41.3 at
  full resolution, and would need `fit(weights=)` to compensate. 500/run keeps the balance.
- **`nested_constraints`.** `{"exp": {"exp": 0}}` cost 3× MSE in a smoke test, and σ row 13
  uses `exp(exp(#4))` legitimately. It restricts shape without targeting the failure mode,
  which is division.
- **Constraining `exp`.** Self-limiting already: towers over parameters are constant in
  step_norm, and towers over step_norm overflow at the training rows and get rejected.
- **Per-condition MSE reweighting.** Measured and not worth it — the largest condition
  carries only 6.4 % (σ) / 8.7 % (clip) of pooled SSE. The imbalance that matters is
  *within* a curve, not across conditions, and log space is what addresses it.

## Consequences

- **All four FirSweep syntheses must be re-run.** The new identity fields change every
  slug, so they start clean rather than warm-starting from the old fronts — which is the
  desired behaviour, not an inconvenience. Each is one long job (ADR 0017).
- **All existing equation-transfer cells are invalid** — both arms', not just
  `f152229a`'s — and must be regenerated. This compounds with the separate
  `seat_on_budget` units bug fixed in `f44d39a`: cells written before that commit were
  already invalid for an unrelated reason.
- **`equations.csv` and `constants.csv` are now in log space.** Anything that reads an
  equation string and does not go through `InvertingPredictor` — a plot axis label, a
  thesis table of coefficients — is now reporting `log σ`, and must say so.
- **A new per-target artefact, `front_health.csv`**, records the dense sweep for every
  front row. It is the first place to look when a distilled equation misbehaves.
- **PySR's `batch_size` remains at its default 50**, against 96 free template constants
  (32 conditions × 3 params). Constant optimisation is badly under-determined per batch;
  `batching=True` batches both evolution and constant optimisation, while `finalize_costs`
  and hall-of-fame comparison use the full dataset. Raising it is cheap and may be a larger
  quality win than the row count. **Open, deliberately not changed here.**
