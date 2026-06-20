# Symbolic Regression: per-condition template constant read-back

**Scope:** `src/symbolic_regression.py :: build_template_spec, extract_template_constants`,
`src/sr_category.py :: template_param_names`, `docs/adr/0006-per-condition-template-symbolic-regression.md`.

## Design decisions & rationale

- **Template mode is the default SR fit (ADR 0006).** Instead of one pooled equation over
  `(eps, T, step_norm, arch_param_count, …)`, PySR fits a single *universal schedule shape*
  `f(step_norm)` shared across all runs plus a few free *per-condition constants*
  `p1…pK` (default K=3). The indexing category is the **condition**
  `(dataset, eps, T, arch_label)`; the replicate seeds of a condition collapse into one
  constant vector. Within a run the only varying input is `step_norm`, so a pooled equation
  must contort one form to cover every condition — splitting shape from constants lets `f`
  capture the transferable shape while the constants absorb each condition's scale/offset.

- **The constants are passed *into* `f`, not applied outside it.** `build_template_spec`
  builds `combine = "f(step_norm, p1[category], p2[category], …)"` with
  `variable_names=["step_norm", "category"]`. So although `f`'s only real *feature* input is
  `step_norm`, the K constants are extra arguments and PySR discovers the modulation algebra
  itself (e.g. a complexity-6 winner `f = #4 + sin(#1)*#3` = `offset + sin(step_norm)*scale`).
  `category` is **1-indexed** because it indexes a Julia vector (`p1[category]`); the fit-time
  column is built by `sr_category.category_series`, which emits `position + 1`.

- **Read-back path (verified, do not re-derive).** Fitted per-condition constants are read
  off any equation row — `model.get_best()` or `model.equations_.iloc[i]` — via:

  ```python
  params = equation_row["julia_expression"].metadata.parameters  # Julia NamedTuple
  p1 = np.asarray(params.p1, dtype=np.float64)                    # field == declared name
  ```

  `extract_template_constants` iterates the declared names with `getattr(params, name)` to
  build a `{name: ndarray}` of shape K × n_conditions. Because it works off `equations_`
  rows too, it is compatible with reloading a pickled `PySRRegressor`.

- **Gotcha — NamedTuple has no string `getindex`.** `params["p1"]` raises a Julia
  `MethodError`; fields must be reached by **attribute** (`getattr(params, "p1")`). The unit
  tests model this with a fake whose `__getitem__` raises, to lock the attribute-access
  contract.

- **Gotcha — values come back float32.** `extract_template_constants` casts to float64, since
  downstream `constants.csv` / stage-2 may need the wider type.

- **Gotcha — parameter slots are unordered / symmetric.** PySR assigns offset↔scale to `p1`
  or `p2` interchangeably across runs (both orderings observed). `p1…pK` is an *unordered
  set*; meaning is fixed by the fitted `combine` expression, not by slot name. Any assertion
  on recovered values must be order-agnostic across the K slots (see
  `test/test_symbolic_regression_template.py :: _match_unordered`).

- **Persistence (stage-1 only).** `main` persists `category_map.json` (ordered condition →
  1-indexed integer) at the slug dir and a per-target `constants.csv` (the K × n_conditions
  matrix joined back onto condition keys, via `sr_category.build_constants_table`). Generalising
  these constants to unseen conditions (`p[condition] ~ g(eps, T, arch)`) is the deferred
  stage-2 — only the read-back + persistence exist now.
