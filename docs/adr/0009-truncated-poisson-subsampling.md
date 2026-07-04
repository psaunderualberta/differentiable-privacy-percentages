# True Poisson subsampling via a truncated fixed-size buffer

The DP-SGD sampler drew *exactly* `batch_size` records without replacement
(`approx_max_k` over a uniform vector) while the GDP accountant assumed **Poisson
subsampling** — an unaccounted mismatch. We close it by making the *sampler* truly
Poisson (leaving the accountant untouched), realized under static-shape JAX as a
**fixed-size buffer of `B > batch_size` slots**: draw independent per-record
inclusions `mask = (u < p)`, pull the included records into the buffer with an
**exact `lax.top_k`**, and mask out the unused slots. The buffer cannot represent
the vanishingly rare event that more than `B` records are drawn, so the realized
mechanism is Poisson *truncated at `B`*, with `B` sized so that truncation costs
negligible privacy.

## Status

accepted

## Why

- **Direction — fix the sampler, not the accountant.** We had already built (and
  reverted, commit `22d2b0a`) the alternative: re-account the fixed-size
  without-replacement sampler under an RDP/WOR model with a Riemannian projection.
  It works, but it drags the whole project out of the GDP framework where the
  literature review, baselines, tuning, and every existing comparison already live
  — a large methodological change to defend months before the thesis defense. Fixing
  the sampler keeps the trusted GDP accountant byte-for-byte unchanged.
- **Truncated Poisson is honestly negligible, not hand-waved.** Couple the truncated
  run to a true-Poisson run on shared coin-flips: they differ only if some step
  overflows, so the length-`T` runs are `≤ T·q` apart in total variation
  (`q = P(Binom(N,p) > B)`). Through the standard approximate-DP lemma this yields
  `(ε, δ_μ(ε) + (1+eᵉ)·T·q)`-DP — a purely **additive** δ-inflation at the operating
  point. The `(1+eᵉ)` factor (easy to miss; `≈2.2×10⁴` at ε=10) is carried, and `B`
  is chosen per-run as the smallest integer with `(1+eᵉ)·T·q ≤ c·δ`, `c = 1e-3`,
  from the **exact** Binomial tail (host-side scipy at startup, like `approx_to_gdp`),
  capped at `N`. Each run logs the analytic `q` and the empirical overflow rate, so
  negligibility is *certified per run*, not asserted.
- **`batch_size` stays the expected batch `L = pN`** (the accountant's `p·N` and the
  gradient-mean / noise divisor), so the entire privacy path — sensitivity `C`, noise
  scale, the `(μ/p)²+T` constraint — is unchanged. `B` is a *separate* derived
  quantity on `DPTrainingParams`.

## Considered and rejected

- **RDP / fixed-size-WOR re-accounting** (the reverted Riemannian direction). Sound,
  arguably more elegant, but leaves the GDP framework and demands fresh literature
  review, baseline re-accounting, and tuning for no expected accuracy gain — wrong
  trade so close to the defense.
- **Fixed buffer `B = 2pN`** (constant multiple). Simple, but not certified against
  `δ` (still a hand-wave), and — because inner-loop compute is **linear in `B`**
  (paid `T` × outer-steps × `schedule_batch_size` × any ES population) — it roughly
  doubles sweep GPU-hours to buy an added `δ ≈ 10⁻⁴⁸` when `~10⁻⁹` suffices
  (worst-case certified `B ≈ 390 < 500 = 2pN`). Certified per-run sizing spends
  compute only where the tail demands it.
- **Keeping `approx_max_k` for the buffer fill.** Its approximation can drop a
  genuinely-included record and admit an excluded one, reintroducing a
  data-dependent deviation that muddies the "true Poisson" claim. Exact `lax.top_k`
  over `{0,1}` scores places all included records above all excluded ones, so with
  no overflow the buffer holds *exactly* the Poisson sample; on overflow, index-order
  tie-breaking gives the deterministic "keep-first-`B`" rule the δ analysis assumes.
  `B` is small (~400), so exact `top_k` is cheap.

## Consequences

- **Every existing result must be re-run.** All prior W&B runs were trained under
  fixed-size WOR while accounted as Poisson; the full `create_experiments.py` matrix
  and all baselines have to be regenerated to be self-consistent. Committed to.
- **Both inner-loop paths change identically** — `train_with_noise` and
  `train_with_stateful_noise` — so learned schedules and the `StatefulMedianGradient`
  baseline share one sampler; otherwise the matched-privacy box-plots would compare
  Poisson against WOR *between conditions*. The stateful path's per-step gradient
  medians are taken over the `m` valid rows (same `valid` mask), not the `B` slots.
- **The `L`-vs-`B` split** disambiguates a previously-overloaded `batch_size`:
  physical-count sites (buffer/index shapes, microbatch reshape) use `B`; divisor
  sites (`dp.py:142`, `util.py:144`) use `L`. Padding slots hold *real* records
  (masked to zero), so there is no dummy/NaN data — but train-loss reporting is fixed
  to mask and divide by `m` so logged curves are not polluted by masked rows.
- Derivations live in `thesis/notes/privacy-accountant-and-projection.md`
  ("Sampling: reconciling the accountant with the implementation").
