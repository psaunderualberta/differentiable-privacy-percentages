# Riemannian gradient descent under Rényi DP — the as-built procedure

*This is the authoritative, as-built reference for the schedule outer loop that
optimises a DP-PSAC noise schedule on the RDP budget manifold. It supersedes the
"planned design" Riemannian section of
[`privacy-accountant-and-projection.md`](privacy-accountant-and-projection.md)
(which is GDP-framed and still describes momentum/vector transport — **not used**;
see §7). Design rationale lives in `docs/adr/0007-rdp-accounting-model.md` and
`docs/adr/0008-riemannian-schedule-projection.md`.*

The outer loop is projected gradient descent reframed as **optimisation on the
budget manifold** $M=\{\theta : g(\theta)=0\}$: tangent-project the loss gradient
(Riemannian gradient), take an SGD step, then retract back onto $M$. Fixed points
satisfy KKT ($\nabla L \parallel \nabla g$) rather than the biased $\nabla L
\parallel \text{ray}$ of a plain scaling projection.

---

## 1. Setup and notation

| Symbol | Meaning | Code |
|---|---|---|
| $\theta$ | learnable BSpline control points (noise side) | `noise_schedule.control_points` |
| $\sigma^{\text{mult}}_i$ | per-step noise multiplier, $i=1..T$ | `_get_private_sigmas()` |
| $w_i = 1/\sigma^{\text{mult}}_i$ | accountant noise weight | `get_private_weights()` |
| $C_i$ | per-step clip (free Euclidean param) | `get_private_clips()` |
| $p = m/n$ | subsampling ratio | `privacy_params.p` |
| $T$ | number of DP-SGD steps | `privacy_params.T` |
| $(\varepsilon,\delta)$ | target DP budget | `privacy_params.eps/.delta` |
| $\alpha^\ast$ | binding integer Rényi order (adaptive) | `privacy_params.alpha_star` |
| $\mathcal A$ | candidate integer orders | `privacy_params.alphas` |

**Schedule parameterisation (BSpline, positivity via softplus):**
$$\sigma^{\text{mult}} = B\,\operatorname{softplus}(\theta), \qquad
w = 1 \big/ \big(B\,\operatorname{softplus}(\theta)\big),$$
where $B \in \mathbb R^{T\times k}$ is the fixed clamped-uniform basis
(`bspline.py::get_valid_schedule = basis @ softplus(cp)`).

**Decoupled DP-PSAC noise (why the constraint is $\sigma$-only).** The noise std
handed to `dp.py` is $C_i\,\sigma^{\text{mult}}_i$, so the clip $C_i$ **cancels**
from the noise-to-sensitivity ratio and the privacy cost depends on $w$ only.
Clips are free, learning-rate-like utility parameters optimised by plain
Euclidean SGD; the manifold, tangent projection and retraction touch the noise
leaves only.

**Adjacency / sensitivity.** Fixed-size sampling **without replacement** under
**replace-one** (substitution) adjacency ⇒ per-step Gaussian sensitivity $2C$
(drop one $\lVert\cdot\rVert\le C$ contribution, add another). The
noise-to-sensitivity multiplier is $s = \sigma^{\text{mult}}/2 = 1/(2w)$.

---

## 2. RDP accountant (`privacy/rdp_accountant.py`)

**Base (un-subsampled) Gaussian RDP** at integer order $\alpha$, under $2C$
sensitivity ($s=1/(2w)$, so $\alpha/(2s^2) = 2\alpha w^2$):
$$\operatorname{func}(\alpha) = \frac{\alpha}{2 s^2} = 2\,\alpha\, w^2.$$

**Amplified per-step RDP** — without-replacement subsampled Gaussian,
Wang–Balle–Kasiviswanathan Thm 9 (arXiv:1808.00087), closed form for integer
$\alpha\ge 2$. With $\operatorname{cgf}(x) = x\cdot\operatorname{func}(x{+}1)$:

- $j=2$ term (the $\operatorname{func}(\infty)=\infty$ Gaussian collapse fixes the
  inner $\min$ to $\log 2$ in the second branch):
$$M_2 = 2\log p + \log\binom{\alpha}{2}
      + \min\!\Big(\log 4 + \operatorname{func}(2) + \log\big(1-e^{-\operatorname{func}(2)}\big),\;
      \operatorname{func}(2) + \log 2\Big).$$
- $j = 3,\dots,\alpha$:
$$M_j = \log 2 + \operatorname{cgf}(j{-}1) + j\log p + \log\binom{\alpha}{j}.$$
- Log-moment sum (the "$+1$" is folded into a `softplus` for float32 accuracy in
  the small regime):
$$L = \operatorname{softplus}\!\Big(\operatorname{logsumexp}_{j=2}^{\alpha} M_j\Big).$$
- Per-step RDP, capped by the un-amplified bound:
$$\rho_{\text{step}}(\alpha; w) = \frac{1}{\alpha-1}\,
      \min\big((\alpha-1)\operatorname{func}(\alpha),\; L\big).$$

$\alpha$ is a **static Python int**, so the moment sum unrolls at trace time and
$\binom{\alpha}{j}$ are compile-time constants; $\rho_{\text{step}}$ is JIT- and
grad-compatible in $w$.

**Composition** (RDP is additive over the $T$ independent steps):
$$\rho_{\text{total}}(\alpha) = \sum_{i=1}^{T}\rho_{\text{step}}(\alpha; w_i).$$

**RDP → $(\varepsilon,\delta)$ conversion** (tightest order):
$$\varepsilon(\delta) = \min_{\alpha\in\mathcal A}\Big[\,\rho_{\text{total}}(\alpha)
      + \tfrac{\log(1/\delta)}{\alpha-1}\,\Big].$$

**Binding order** (recomputed between outer steps, in Python):
$$\alpha^\ast = \operatorname*{arg\,min}_{\alpha\in\mathcal A}\Big[\,\rho_{\text{total}}(\alpha)
      + \tfrac{\log(1/\delta)}{\alpha-1}\,\Big].$$

Validated against `autodp` (`GaussianMechanism(sigma=s).RenyiDP`,
`AmplificationBySampling(PoissonSampling=False, improved_bound_flag=False)`),
rtol $10^{-5}$, in `test/test_rdp_accountant.py`.

---

## 3. The budget manifold and the constraint scalar

At the binding order, the on-budget condition $\varepsilon(\delta)=\varepsilon$
rearranges to a target on $\rho_{\text{total}}$:
$$\rho_{\text{total}}(\alpha^\ast) = \underbrace{\varepsilon - \tfrac{\log(1/\delta)}{\alpha^\ast-1}}_{c(\alpha^\ast)\ =\ \texttt{rho\_budget()}}.$$

The **constraint scalar** (`RDPPrivacyParameters.constraint`, exposed on the
schedule as `constraint_value()`):
$$\boxed{\,g(\theta) = \rho_{\text{total}}\big(\alpha^\ast; w(\theta)\big) - c(\alpha^\ast)\,}$$
$$M = \{\theta : g(\theta)=0\},\qquad
g>0 \Rightarrow \text{over budget (too little noise)}.$$

$g$ is a cheap, closed-form, differentiable scalar; it is what we autodiff for the
manifold normal (§4). It is a function of $\theta$ through
$w(\theta)=1/(B\,\operatorname{softplus}(\theta))$, subsuming both $\partial
w/\partial\theta$ and $\partial\rho/\partial w$.

---

## 4. Riemannian gradient (tangent projection) — `policy/riemannian.py`

$M$ is a level set of $g$, so its normal is $\nabla g$ and its tangent space is
$T_\theta M = \{v : \langle \nabla g, v\rangle = 0\}$.

**Normal by autodiff** (never hand-derived, never the retraction):
$$n_\theta = \nabla_\theta g(\theta) = \texttt{eqx.filter\_grad}(\texttt{constraint\_value})(\text{schedule}).$$
Autodiff flows through `softplus` and the fixed `basis` in one shot and lands the
normal in exactly the `es_filter`/`eqx.partition` differentiable leaf space. $n$
is **zero on every clip leaf** (clips are absent from $g$), so those pass through
untouched.

**Riemannian (tangent) gradient** = orthogonal projection of the ambient update
$v$ onto $T_\theta M$:
$$\xi = \operatorname{proj}_{T_\theta M}(v) = v - \frac{\langle n, v\rangle}{\langle n, n\rangle}\, n.$$

Delivered as a **stateless** `optax.GradientTransformation`
(`riemannian_tangent_projection()`): it reads `params` (the schedule), computes
$n$, and returns $\xi$. No momentum buffer, no vector transport (§7). Inner
products $\langle\cdot,\cdot\rangle$ are summed over inexact-array leaves; the
$\langle n,n\rangle=0$ case is guarded to a no-op.

---

## 5. Equality scaling retraction — `project_scale` / `schedule.project()`

The budget always binds, so the retraction $R_\theta$ drives the schedule back to
the boundary by scaling $\sigma^{\text{mult}}$ by a single common factor $s>0$.
This is **exact in the BSpline family**: scaling the output vector by $s$ equals
scaling the positive control points by $s$,
$$s\cdot\sigma^{\text{mult}} = B\,\big(s\,\operatorname{softplus}(\theta)\big),$$
so there is no least-squares round-trip error
(`from_projection`: $\text{cp} = \operatorname{softplus}^{-1}(B^{+}(s\,\sigma^{\text{mult}}))$).

**Target $\varepsilon$ directly, via the full $\min_\alpha$ conversion** — *not* a
single fixed order:
$$\text{find } s \text{ s.t. } \; r(s) := \varepsilon\!\left(\delta;\; w=\tfrac{1}{s\,\sigma^{\text{mult}}}\right) - \varepsilon = 0,$$
$$\varepsilon(\delta;\,\cdot) = \min_{\alpha\in\mathcal A}\big[\rho_{\text{total}}(\alpha) + \tfrac{\log(1/\delta)}{\alpha-1}\big].$$
$r(s)$ is monotone **decreasing** in $s$ (more noise ⇒ smaller $\varepsilon$), so
$s^\ast$ is found by **60 iterations of bisection** on $[10^{-4}, 10^4]$
(`fori_loop`): if $r(\text{mid})>0$ the schedule is still under-noised ⇒ raise the
lower bracket, else lower the upper. Return $s^\ast\,\sigma^{\text{mult}}$.

**Why target $\varepsilon$ and not $g$ (a single-$\alpha$ constraint).** A single
order's budget $c(\alpha) = \varepsilon - \log(1/\delta)/(\alpha-1)$ can be
**negative** off-manifold (e.g. small $\alpha$ with large $\log(1/\delta)$),
making $\rho_{\text{total}}(\alpha)=c(\alpha)$ unreachable by any positive
scaling — the single-$\alpha$ bisection then has no root. $\varepsilon(\delta)$ is
always reachable. Because the retraction is post-step and **never
differentiated**, the $\min_\alpha$ kinks are harmless. $\alpha^\ast$ is needed
only for the tangent normal (§4), evaluated *at* the on-budget point where
$c(\alpha^\ast)=\rho_{\text{total}}(\alpha^\ast)\ge 0$ by construction — hence the
$\alpha^\ast$ refresh runs on the freshly-retracted schedule (§6).

The retraction leaves the clip side $C$ bit-exact unchanged.

---

## 6. The full outer-loop algorithm (`main.py`)

**Initialisation.**
1. Build the schedule; `schedule = schedule.project()` (land on $M$).
2. `schedule = schedule.refresh_alpha_star()` — select $\alpha^\ast$ on the
   on-budget $w$.

**Optimizer chain** (order matters — neutralise/clip first, they preserve
direction; then tangent-project; then scale by lr):
```
optax.chain(
    clip_by_global_norm(max_grad_norm),   # rescales / poisons a bad step
    zero_nans(),                          # corrupt step -> no-op (not a crash)
    riemannian_tangent_projection(),      # v -> xi = v - <n,v>/<n,n> n
    scale_by_learning_rate(lr),           # -lr * xi   (no outer momentum)
)
```

**Per outer step $k$** (schedule is on $M$ with a fixed $\alpha^\ast$ at entry):
$$\begin{aligned}
&v = \nabla_\theta L(\theta_k) && \text{(analytic loss grad, whole schedule)}\\
&\hat v = \texttt{zero\_nans}(\texttt{clip}(v)) && \text{robustness}\\
&\xi = \hat v - \tfrac{\langle n,\hat v\rangle}{\langle n,n\rangle} n,\quad n=\nabla_\theta g(\theta_k) && \text{Riemannian gradient}\\
&\theta^- = \theta_k - \eta\,\xi && \texttt{apply\_updates}\\
&\theta_{k+1} = R_{\theta^-}\ (=\ \texttt{project()}) && \text{scaling retraction} \to M\\
&\alpha^\ast \leftarrow \operatorname*{arg\,min}_\alpha[\dots] && \texttt{refresh\_alpha\_star()} \text{ on } \theta_{k+1}
\end{aligned}$$
Clip leaves flow through $\xi=\hat v$ unchanged (normal is zero there) and are
carried through the retraction untouched — they descend $L$ as ordinary Euclidean
parameters.

Convergence: fixed point $\xi=0 \iff \nabla_\theta L \in \operatorname{span}
\nabla_\theta g \iff$ KKT-in-$\theta$; descent lemma ⇒ $O(1/K)$ to a stationary
point (nonconvex ⇒ stationary, not global).

---

## 7. Correctness notes / gotchas

- **No outer-loop momentum ⇒ stateless transform, no vector transport.**
  "Momentum" in this project is the *inner* DP-SGD optimiser's momentum only. The
  outer Riemannian descent is plain tangent-projected SGD, so the transform keeps
  no buffer and needs no projection vector transport. The outer `optax.sgd`
  momentum is forced to $0$ on this path. *(This supersedes the heavy-ball /
  vector-transport section in the companion notes file.)*
- **$\alpha^\ast$ timing.** Refresh $\alpha^\ast$ **after** the retraction, on the
  on-budget schedule — that is where $c(\alpha^\ast)\ge 0$ and where the manifold
  the next step lives on is defined. Each projection/step uses **one** fixed
  integer order so the feasible surface is smooth ($\min_\alpha$ has kinks).
- **Scope.** Only `DecoupledSigmaAndClipSchedule` is on RDP + the Riemannian path
  (`use_riemannian = isinstance(schedule, DecoupledSigmaAndClipSchedule)` in
  `main.py`); other schedules stay on the GDP projection. Its `from_config`
  converts the shared GDP object → `RDPPrivacyParameters` internally, so factory
  signatures are unchanged.
- **`dp.py` unchanged.** The decoupled schedule already emits $C\sigma^{\text{mult}}$
  as the noise std; RDP accounting needs no inner-loop rewrite.
- **ES path not covered.** ES calls `project()` inside the black-box eval, so it
  estimates descent of $L\circ R$ ⇒ biased shape unless the ES estimator is itself
  made Riemannian. Deferred; analytic path only.

---

## 8. Code and test map

| Concern | Module | Tests |
|---|---|---|
| RDP formulae ($\rho_{\text{step}}$, $\rho_{\text{total}}$, $\varepsilon(\delta)$, $\alpha^\ast$) | `privacy/rdp_accountant.py` | `test/test_rdp_accountant.py` (autodp oracle) |
| Budget object, $g$, $c(\alpha^\ast)$, scaling retraction, $\alpha^\ast$ reselect | `privacy/rdp_privacy.py` | `test/test_rdp_privacy.py` |
| $\sigma$-only schedule, `constraint_value`, `project`, `refresh_alpha_star` | `policy/schedules/decoupled_sigma_and_clip.py` | `test/test_decoupled_schedule.py` |
| Stateless tangent-projection transform | `policy/riemannian.py` | `test/test_riemannian.py` |
| BSpline $B\,\operatorname{softplus}(\theta)$ + `from_projection` | `policy/base_schedules/bspline.py` | `test/test_bspline_jax.py` |
| Outer-loop wiring (chain, $\alpha^\ast$ refresh) | `main.py` | end-to-end smoke |

**References.** Wang–Balle–Kasiviswanathan (2019) / Balle–Barthe–Gaboardi (2018)
for WOR amplification; Mironov (2017) for RDP composition & conversion; Boumal,
*Introduction to Optimization on Smooth Manifolds* (2023) Ch. 4–5, 10 and
Absil–Mahony–Sepulchre (2008) Ch. 4, §8.1 for Riemannian gradient + retraction.
