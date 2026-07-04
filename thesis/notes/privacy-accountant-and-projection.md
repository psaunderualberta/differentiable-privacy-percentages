# Privacy Accountant and Projection

**Scope:** `src/privacy/gdp_privacy.py` — `GDPPrivacyParameters`, `approx_to_gdp`, `project_weights`, `project_sigma_and_clip`, helpers `_sc_*`, and weight↔(σ, clip) conversion methods.

## Design decisions & rationale

- **GDP (Gaussian DP) as the accountant**, converted once from (ε, δ) via Brent's method at startup outside JAX — `approx_to_gdp` (L139). Keeps the JAX hot path free of scipy and treats μ as a fixed constant for the rest of training.
- **All immutable privacy quantities are wrapped in `lax.stop_gradient` via `@property`** (`mu`, `p`, `T`, `mu_0`, `w_min`, `w_max`, L192–214). Prevents the schedule's gradient updates from leaking into accounting parameters even though they live on the same `eqx.Module`.
- **Two projection variants coexist**: `project_weights` (1D bisection over weight-space, used by weight-parameterized schedules) and `project_sigma_and_clip` (2D Euclidean projection in (σ, clip) space, used when σ and clip are independently learned). The TODO at L326 ("Move projection into the gradient computation") flags that current placement after the GD step is a known compromise — *(rationale for not yet doing so: not in repo — confirm with author)*.
- **Negative (σ, clip) values are accepted then `abs()`-ed at the end of `project_sigma_and_clip`** (L472–473): the constraint is symmetric in sign, so allowing the Newton iterates to cross zero would waste iterations — instead, post-hoc absolute value is taken.

## Numerical stability

- **`_MAX_MU2 = 80.0`** cap on `(clip/σ)²` (`_mu_two`, L23) keeps `exp(mu²)` below float32 overflow (~3.4e38).
- **Log-space evaluation of overflow-prone products** in the Newton hot path (`_sc_logQ`, `_sc_residual`, `_sc_jacobian`): every `Q · aⁿ / bᵐ` term is computed as `exp(min(log_Q + n·log a − m·log b, _MAX_LOG=80))`.
- **Row-scaled Cramer's rule** in `_sc_newton_step` (L83–95): each Jacobian row is normalized by its ∞-norm before solving the 2×2 system, with a 1e-30 floor on row scale and determinant.
- **Newton step damping**: per-component `alpha ≤ 1` chosen so neither `a + α·da` nor `b + α·db` falls below 90% of the way to 0; post-step values are floored at 1e-12 (L126–131).
- **Runtime `eqx.error_if` guards** in `_validated_mu_schedule` (L281–293) reject zero/Inf weights or μ-schedule entries.

## JAX/Equinox patterns

- **`project_weights` is `@eqx.filter_jit`**; entire bisection + nested Newton runs as one compiled call (L327).
- **Nested `lax.while_loop`s**: outer loop bisects μ until `hi - lo ≤ tol`; inner loop runs Newton on the per-component equation until residual ≤ tol (L362–395). Pure JAX, JIT-compatible, differentiable through.
- **`project_sigma_and_clip`** uses `while_loop` to find `lam_max` by 1.05× doubling (L443–449), then a fixed-iteration `fori_loop` of 60 bisection steps for ~1e-18 relative precision (L460).
- **All static fields (mu, p, T, …) stop-gradient'd**, so the projection's outputs only carry gradients w.r.t. the input weights / (σ, clip) — clean for backprop into the schedule.

## Privacy accounting math

### From (ε, δ)-DP to GDP μ

`approx_to_gdp(eps, delta)` (L139–161) solves for μ in the standard GDP↔(ε,δ) identity

$$\delta = \Phi\!\left(-\tfrac{\varepsilon}{\mu} + \tfrac{\mu}{2}\right) - e^{\varepsilon}\,\Phi\!\left(-\tfrac{\varepsilon}{\mu} - \tfrac{\mu}{2}\right),$$

using `scipy.optimize.root_scalar(method="brentq")` with bracket `[1e-12, 100]`. Runs once at startup; the result is stored as `self.__mu` and frozen via `stop_gradient`.

### Per-step GDP (μ₀) under Poisson subsampling

`compute_mu_0` (L216):

$$\mu_0 = \sqrt{\log\!\left(\frac{\mu^2}{p^2 T} + 1\right)}.$$

This is the uniform per-step μ such that T identical subsampled-Gaussian steps with subsampling probability *p* compose (under the GDP central-limit / subsampling amplification used in this repo) to a total budget of μ. Inverting it gives the constraint that *any* non-uniform schedule must satisfy.

### Composition / budget constraint

The total μ-expenditure for a non-uniform schedule of (σᵢ, Cᵢ) is

$$\mu_{\text{total}} = p\,\sqrt{\sum_{i=1}^{T}\bigl(e^{(C_i / \sigma_i)^2} - 1\bigr)}$$

(implemented as `compute_expenditure`, L234–236). Requiring `μ_total ≤ μ` gives the master constraint

$$\sum_{i=1}^{T} e^{(C_i/\sigma_i)^2} \le \Bigl(\tfrac{\mu}{p}\Bigr)^2 + T \;\;=\!:\; B,$$

the bound `B` appearing as `bound` in `project_weights` (L337) and `B` in `project_sigma_and_clip` (L425).

### Weights → μ-schedule (reparameterization)

The codebase reparameterizes the per-step μ schedule by *weights* `wᵢ ≥ 0` with `∑ wᵢ² = (μ/p)² + T` (uniform case `wᵢ ≡ 1` yields μᵢ ≡ μ₀). `weights_to_mu_schedule` (L238–249):

$$\mu_i = \sqrt{\log\!\bigl(w_i^2\,(e^{\mu_0^2} - 1) + 1\bigr)}.$$

This is the unique map sending `wᵢ = 1 ∀i` to `μᵢ = μ₀ ∀i` while preserving the budget identity `∑ e^{μᵢ²} = (μ/p)² + T`. Inverse `mu_schedule_to_weights` (L251–265):

$$w_i = \sqrt{\frac{e^{\mu_i^2} - 1}{e^{\mu_0^2} - 1}}.$$

### Weights → σ, weights → clip

Given a per-component clip `C` and weights:

- `weights_to_sigma_schedule(C, weights)` → `σᵢ = C / μᵢ` (L295–297, via `gdp_to_sigma` L267).
- `weights_to_clip_schedule(sigmas, weights)` → `clipᵢ = σᵢ · μᵢ` (L299–301).

So at any (σᵢ, μᵢ) pair, `clipᵢ / σᵢ = μᵢ`, and `e^{(clipᵢ/σᵢ)²} = e^{μᵢ²} = wᵢ²(e^{μ₀²} − 1) + 1`. Summing recovers the budget identity exactly.

### Inverting σ → weights

`sigma_schedule_to_weights(C, sigmas)` (L303–324): clips σ to `max_sigma` (config), then computes `C/σ = μ` and feeds through `mu_schedule_to_weights`. The pre-clip prevents `C/σ → 0` (which would map to weight 0 and break the constraint).

### Projection in weight space — `project_weights`

Given post-GD weights `wᵢ` (possibly violating `∑ e^{wᵢ²} = B`), solve the *Lagrangian* of the equality-constrained projection

$$\min_{\tilde w} \tfrac12 \sum (\tilde w_i - w_i)^2 \;\text{s.t.}\; \sum e^{\tilde w_i^2} = B.$$

KKT: `tilde wᵢ · (1 + 2μ·e^{tilde wᵢ²}) = wᵢ`, where μ is the (scalar) dual. The implementation (L328–400):

1. **Inner Newton** (`c_i_tildes_body`, L348–360) on each component `tilde wᵢ` with
   - residual `f = c̃·(1 + 2μ·e^{c̃²}) − w`,
   - derivative `f' = 1 + 2μ·e^{c̃²}·(1 + 2c̃²)`.

   ```python
   f_ = c_i_tildes * (1 + 2 * mu * jnp.exp(c_i_tildes**2)) - weights
   f_prime = 1 + 2 * mu * jnp.exp(c_i_tildes**2) * (1 + 2 * c_i_tildes**2)
   ```

2. **Outer bisection** on μ ∈ `[0, B]` (L391–395) using monotonicity of `h(μ) = ∑ e^{c̃ᵢ(μ)²} − B`. Negative `h` → shrink upper bound; positive → raise lower.

3. Returns `c̃ᵢ(μ*)` at the converged dual μ*.

### Projection in (σ, clip) space — `project_sigma_and_clip`

For schedules that learn σ and clip independently, the budget is inequality `∑(e^{(Cᵢ/σᵢ)²} − 1) ≤ (μ/p)²`. Lagrangian KKT (in (X=clip, Y=σ)):

$$\begin{aligned}
X_i &= x_i - 2\lambda \cdot e^{\mu_i^2} \cdot X_i / Y_i^2 \\
Y_i &= y_i + 2\lambda \cdot e^{\mu_i^2} \cdot X_i^2 / Y_i^3
\end{aligned}$$

where `(xᵢ, yᵢ)` are the post-GD inputs and `μᵢ² = (Xᵢ/Yᵢ)²` (clipped to `_MAX_MU2`). The repo's `_sc_residual` (L46–55) and `_sc_jacobian` (L58–71) implement exactly these in log-space:

```python
t0 = exp(log_Q + log_a - 2 log_b)   # 2λ·e^{μ²}·X/Y²
t1 = exp(log_Q + 2 log_a - 3 log_b) # 2λ·e^{μ²}·X²/Y³
f0 = a - x + t0                     # stationarity for X
f1 = b - y - t1                     # stationarity for Y
```

`_sc_inner_solve_all` (L99–136) runs a *shared* `while_loop` of damped Newton across all T components in parallel until `max(f0² + f1²) ≤ 1e-10`, capped at 2000 iterations. The outer code (L429–476):

1. Quick feasibility check (`S = ∑ e^{(C/σ)²} ≤ B?`); short-circuits.
2. Find `lam_max` by 1.05× doubling until the Newton solve yields `h(λ) ≤ 0`.
3. 60-step `fori_loop` bisection on λ ∈ `[1e-6, lam_max]`.
4. Final Newton solve at `λ* = (lo + hi)/2`, then `abs(·)` to recover sign-canonical (σ, clip).

### Effective-epsilon diagnostic

`compute_eps(max_sigma)` (L220–232) reports the ratio

$$\frac{e^{1/\sigma_{\max}^2} - 1}{e^{\mu_0^2} - 1},$$

which measures how much of the per-step budget a single step at `max_sigma` consumes relative to the uniform baseline `μ₀`. Used as a logged metric, not as a constraint.

## Sampling: reconciling the accountant with the implementation

### The gap

The accountant above assumes **Poisson subsampling**: each of the `N` training
records is included in a step's minibatch independently with probability
`p = batch_size / N`. This is what the `μ/p` amplification and the master
constraint `∑ e^{(Cᵢ/σᵢ)²} ≤ (μ/p)² + T` encode.

The sampler in `environments/dp.py` does *not* do this. `_gen_indices`
(L250–253) draws a uniform vector over `N` and takes its top-`batch_size` via
`approx_max_k`, i.e. it selects **exactly `batch_size` records without
replacement** — *fixed-size sampling without replacement (WOR)*, not Poisson.
Fixed-size WOR and Poisson subsampling have genuinely different amplification,
so the code as written is accounted under the wrong subsampling scheme (the
in-code comment claiming it "matches Poisson semantics" is incorrect).

Two ways to close the gap: (A) re-account under fixed-size WOR, or (B) make the
sampler truly Poisson. We take **(B)** — it leaves the trusted GDP accountant
byte-for-byte unchanged, keeps all baselines and literature comparisons inside
the GDP framework we have already built up, and avoids introducing an RDP/WOR
re-accounting so close to the defense.

### Why "true" Poisson needs a truncation

Under static-shape JAX we cannot materialize a variable-size batch. The fix is a
**fixed-size buffer** of size `B`: draw the realized batch size
`m ~ Binomial(N, p)`, load `m` records into the buffer, zero-pad the remaining
`B − m` rows, forward/backward-pass all `B`, and **mask (zero) the padded rows'
per-sample gradients** before summing. Only the buffer is ever passed through the
model — the `N` Bernoulli coin-flips are cheap index generation, so the naive
"run the whole dataset through the model" concern never actually arises.

The one thing a size-`B` buffer *cannot* represent is the (astronomically
unlikely) event `{m > B}`. So the realized mechanism is not Poisson but
**Poisson subsampling truncated at `B`**: on overflow we keep the first `B`
sampled records and drop the rest. This is a deliberate, quantified
approximation, and the point of the analysis below is to certify it costs
negligible privacy.

**Invariants that must not change** when moving to the buffer:
- `batch_size` remains the Poisson *expected* batch `L = pN`; it stays the
  accountant's `p·N`. The buffer size `B > L` is a *separate* quantity.
- The summed clipped gradients are divided by the public constant `L`
  (`= batch_size`), **never** by the buffer size `B` nor the realized `m`. The
  Gaussian sensitivity is still `C` (one record moves the sum by `≤ C`) and the
  noise scale is unchanged, so σ, clip, and the entire constraint set are
  untouched.

### Privacy cost of the truncation

Let `q = P(\mathrm{Binom}(N, p) > B)` be the per-step overflow probability.
Couple the truncated run to a true-Poisson run on shared coin-flips: they emit
identical outputs unless some step overflows, so the total-variation distance
between the two length-`T` runs is `≤ T q`.

Feeding TV-closeness through the standard approximate-DP lemma, for neighbours
`x, x'` and any measurable `S` (with `M'` the true-Poisson mechanism, which is
`(ε, δ)`-DP by the GDP guarantee):

$$
\begin{aligned}
\Pr[M(x)\in S]
  &\le \Pr[M'(x)\in S] + Tq \\
  &\le e^{\varepsilon}\Pr[M'(x')\in S] + \delta + Tq \\
  &\le e^{\varepsilon}\Pr[M(x')\in S] + \delta + (1 + e^{\varepsilon})\,Tq.
\end{aligned}
$$

So the truncated mechanism is

$$
\bigl(\varepsilon,\; \delta_\mu(\varepsilon) + (1 + e^{\varepsilon})\,T\,q\bigr)\text{-DP}
\quad\text{for every }\varepsilon,
$$

i.e. truncation costs a purely **additive** `δ`-inflation of
`(1 + e^{ε}) T q` at the operating point, and nothing else about the GDP
guarantee changes. The `(1 + e^{ε})` factor — easy to overlook — arises because
the comparison is made at fixed `ε`; at `ε = 10` it is `≈ 2.2×10⁴`, so it is not
negligible and must be carried.

### Sizing the buffer `B`

Choose the smallest integer `B` satisfying

$$
(1 + e^{\varepsilon})\, T \cdot P\!\bigl(\mathrm{Binom}(N, p) > B\bigr) \;\le\; c\,\delta,
$$

with a margin `c = 10^{-3}` (added `δ` three orders below the target). The tail
is an **exact Binomial upper tail**, computed once per run on the host — the same
"solve a scalar at startup, freeze it" pattern as `approx_to_gdp`. Both `B` and
the realized added-`δ` are logged to W&B so each run *certifies* its own
truncation is negligible, rather than arguing it by hand.

For the current experiment ranges (`N = 60000`, `batch_size = L = 250`,
`p = 1/240`, so `\mathrm{std} = \sqrt{Np(1-p)} \approx 15.8`) the binding worst
case is `ε = 10, T = 7000, δ = 10^{-6}`, which needs `q \lesssim 6×10^{-18}` and
lands at `B ≈ L + 8.6\,\mathrm{std} \approx 390` (≈ `1.6 L`). Note this is *below*
the naïve `2 p N = 500`: because inner-loop compute is **linear in `B`** (paid
`T` × outer-steps × `schedule_batch_size` × any ES population), a fixed `2pN`
would roughly double the sweep's GPU-hours to buy an added `δ ≈ 10^{-48}` — far
past the point of diminishing returns. Per-run `δ`-certified sizing spends
compute only where the tail bound demands it.

**Status:** planned change. The accountant already assumes Poisson; the sampler
and the buffer/masking path in `train_with_noise` still need to be updated to
realize it.
