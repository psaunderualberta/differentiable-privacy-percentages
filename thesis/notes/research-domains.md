# Research Domains

**Scope:** Survey of academic research areas that the thesis touches on or builds upon. Organized into four groups: (1) core privacy domains, (2) optimization & learning machinery that powers the outer-loop, (3) adjacent/supporting domains (autodiff, JAX, learned optimizers), and (4) directly comparable prior work on adaptive/dynamic DP-SGD schedules.

A note on framing: the codebase contains residual reinforcement-learning nomenclature (e.g. `policy/`, `environments/`, "policy loss") from an earlier iteration of this work. The current method is **not** framed as RL — it is bilevel/hyper-gradient optimization with a privacy constraint. RL is therefore listed only as historical context, not as an active domain.

Citations marked **(verify citation)** are ones I am not fully confident about and should be checked before citing.

---

## 1. Core domains

### 1.1 Differential Privacy (foundations)
The formal privacy notion underlying the entire thesis.

- **Dwork, McSherry, Nissim, Smith (2006).** *Calibrating Noise to Sensitivity in Private Data Analysis.* TCC 2006. — The original (ε, δ)-DP definition and Laplace/Gaussian mechanism foundations.
- **Dwork, Kenthapadi, McSherry, Mironov, Naor (2006).** *Our Data, Ourselves: Privacy via Distributed Noise Generation.* EUROCRYPT 2006. — Introduces (ε, δ)-DP (approximate DP) explicitly.
- **Dwork & Roth (2014).** *The Algorithmic Foundations of Differential Privacy.* Foundations and Trends in Theoretical Computer Science. — The standard monograph; cite for general DP background, composition theorems, post-processing, group privacy.
- **Dwork, Rothblum, Vadhan (2010).** *Boosting and Differential Privacy.* FOCS 2010. — Advanced composition theorem, used to motivate why naive ε-summation overestimates privacy loss.

### 1.2 DP-SGD (the inner-loop algorithm being optimized)
- **Song, Chaudhuri, Sarwate (2013).** *Stochastic gradient descent with differentially private updates.* GlobalSIP 2013. — Earliest DP-SGD formulation.
- **Bassily, Smith, Thakurta (2014).** *Private Empirical Risk Minimization: Efficient Algorithms and Tight Error Bounds.* FOCS 2014. — Theoretical DP-ERM; tight bounds.
- **Abadi, Chu, Goodfellow, McMahan, Mironov, Talwar, Zhang (2016).** *Deep Learning with Differential Privacy.* CCS 2016. — The canonical DP-SGD paper: per-example clipping (the "Abadi clipping" used in `clip_grads_abadi`), Gaussian noise addition, and the **moments accountant**. This is the algorithm whose σ and clip schedules are being learned.
- **McMahan, Andrew, Erlingsson, Chien, Mironov, Papernot, Kairouz (2018).** *A General Approach to Adding Differential Privacy to Iterative Training Procedures.* arXiv:1812.06210. — Generalizes DP-SGD; underpins TF-Privacy/Opacus accountants. *(verify citation)*

### 1.3 Privacy accounting — Rényi DP, zCDP, GDP
The accounting machinery used in `privacy/gdp_privacy.py`.

- **Mironov (2017).** *Rényi Differential Privacy.* CSF 2017. — RDP definition; tighter composition than (ε, δ)-DP.
- **Mironov, Talwar, Zhang (2019).** *Rényi Differential Privacy of the Sampled Gaussian Mechanism.* arXiv:1908.10530. — RDP of subsampled Gaussian, used in modern accountants.
- **Wang, Balle, Kasiviswanathan (2019).** *Subsampled Rényi Differential Privacy and Analytical Moments Accountant.* AISTATS 2019. — Closed-form RDP of subsampled mechanisms.
- **Bun & Steinke (2016).** *Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds.* TCC 2016. — zCDP; provides cleaner Gaussian mechanism analysis.
- **Dwork & Rothblum (2016).** *Concentrated Differential Privacy.* arXiv:1603.01887. — The original CDP definition.
- **Dong, Roth, Su (2022).** *Gaussian Differential Privacy.* Journal of the Royal Statistical Society: Series B, 84(1). — **The f-DP / GDP framework underlying this thesis.** Defines μ-GDP, the central object in `GDPPrivacyParameters`.
- **Bu, Dong, Long, Su (2020).** *Deep Learning with Gaussian Differential Privacy.* Harvard Data Science Review, 2(3). — Applies GDP to DP-SGD; gives the CLT-based composition that motivates per-step μ scheduling.
- **Koskela, Jälkö, Honkela (2020).** *Computing Tight Differential Privacy Guarantees Using FFT.* AISTATS 2020. — Numerical privacy accountants; comparison baseline.
- **Gopi, Lee, Wutschitz (2021).** *Numerical Composition of Differential Privacy.* NeurIPS 2021. — PRV accountant; current state-of-the-art numerical accounting. *(verify citation)*

### 1.4 Privacy amplification by subsampling
The subsampling rate `p` in `GDPPrivacyParameters` is the lever that subsampling amplification acts on.

- **Kasiviswanathan, Lee, Nissim, Raskhodnikova, Smith (2011).** *What Can We Learn Privately?* SIAM J. Computing. — Early subsampling amplification result.
- **Balle, Barthe, Gaboardi (2018).** *Privacy Amplification by Subsampling: Tight Analyses via Couplings and Divergences.* NeurIPS 2018. — Tight subsampling amplification for many mechanisms.

---

## 2. Optimization & learning machinery (the outer loop)

### 2.1 Bilevel optimization
The outer-loop optimization of (σ, C) schedules over an inner DP-SGD trajectory is a bilevel problem.

- **Colson, Marcotte, Savard (2007).** *An overview of bilevel optimization.* Annals of Operations Research, 153(1). — Survey of the bilevel optimization literature.
- **Franceschi, Frasconi, Salzo, Grazzi, Pontil (2018).** *Bilevel Programming for Hyperparameter Optimization and Meta-Learning.* ICML 2018. — Bilevel framing of hyperparameter optimization that closely parallels this thesis's setup.
- **Pedregosa (2016).** *Hyperparameter optimization with approximate gradient.* ICML 2016. — Approximate hypergradients; relevant when truncating backprop through long inner trajectories.

### 2.2 Hypergradient methods (gradient-based hyperparameter optimization)
Differentiating through an inner training loop to update outer ("hyper") parameters — exactly the mechanism used here for (σ, C).

- **Maclaurin, Duvenaud, Adams (2015).** *Gradient-based Hyperparameter Optimization through Reversible Learning.* ICML 2015. — Foundational paper on backpropagating through SGD trajectories.
- **Franceschi, Donini, Frasconi, Pontil (2017).** *Forward and Reverse Gradient-Based Hyperparameter Optimization.* ICML 2017. — Forward-mode vs. reverse-mode hypergradients; memory tradeoffs directly relevant to the `jax.lax.scan` + `jax.checkpoint` design.
- **Lorraine, Vicol, Duvenaud (2020).** *Optimizing Millions of Hyperparameters by Implicit Differentiation.* AISTATS 2020. — Implicit-function-theorem alternative to unrolled differentiation; relevant for scaling.
- **Baydin, Cornish, Rubio, Schmidt, Wood (2018).** *Online Learning Rate Adaptation with Hypergradient Descent.* ICLR 2018. — Online (per-step) hypergradients for learning rates; methodologically adjacent.
- **Shaban, Cheng, Hatch, Boots (2019).** *Truncated Back-propagation for Bilevel Optimization.* AISTATS 2019. — Truncated unrolling; relevant when T is too large to backprop through fully. *(verify citation)*

### 2.3 Meta-learning / learning to learn
Conceptually adjacent: learning some component of an optimization process rather than hand-designing it.

- **Schmidhuber (1987).** *Evolutionary Principles in Self-Referential Learning.* Diploma thesis, TU Munich. — Earliest formulation of "learning to learn." *(verify exact citation form)*
- **Thrun & Pratt (1998).** *Learning to Learn.* Springer. — Foundational edited volume.
- **Andrychowicz, Denil, Gomez, Hoffman, Pfau, Schaul, Shillingford, de Freitas (2016).** *Learning to learn by gradient descent by gradient descent.* NeurIPS 2016. — RNN-based learned optimizers; shares the "differentiate through training" structure.
- **Finn, Abbeel, Levine (2017).** *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.* ICML 2017. — MAML; canonical meta-learning paper, mention for completeness.
- **Metz, Maheswaranathan, Nixon, Freeman, Sohl-Dickstein (2019).** *Understanding and correcting pathologies in the training of learned optimizers.* ICML 2019. — Documents pitfalls (chaotic gradients, exploding meta-loss) that this thesis must contend with when unrolling DP-SGD.
- **Metz et al. (2022).** *VeLO: Training Versatile Learned Optimizers by Scaling Up.* arXiv:2211.09760. — Modern large-scale learned optimizers.

### 2.4 Evolutionary Strategies (black-box gradient estimation)
The `es` branch adds an **antithetic OpenAI-ES** estimator as a drop-in replacement for the analytic hypergradient on ES-opted-in leaves (selected per-schedule via `es_filter()`). It is motivated by the well-documented pathologies of long unrolled gradients (Section 2.3) and by the need to differentiate through schedule parameters that may not have a tractable analytic gradient path (e.g. discrete or non-smooth reparameterizations).

- **Rechenberg (1973).** *Evolutionsstrategie: Optimierung technischer Systeme nach Prinzipien der biologischen Evolution.* Frommann-Holzboog. — Original Evolution Strategies; cite for historical lineage. *(verify citation form)*
- **Schwefel (1977).** *Numerische Optimierung von Computermodellen mittels der Evolutionsstrategie.* Birkhäuser. — Co-founding ES reference. *(verify citation form)*
- **Hansen & Ostermeier (2001).** *Completely Derandomized Self-Adaptation in Evolution Strategies.* Evolutionary Computation, 9(2). — CMA-ES; the dominant ES variant in classical black-box optimization. Cite as background for why the simpler OpenAI-ES form was preferred here (covariance adaptation is unnecessary when perturbations target a low-dimensional schedule).
- **Wierstra, Schaul, Glasmachers, Sun, Peters, Schmidhuber (2014).** *Natural Evolution Strategies.* JMLR 15(27). — **NES; the source of the log-utility rank shaping used in `_nes_log_utilities` in `src/environments/outer_loop.py`.** Defines fitness shaping as ranks → log-utility weights summing to zero, which is what stabilizes the ES gradient estimate against loss scale and outliers.
- **Salimans, Ho, Chen, Sidor, Sutskever (2017).** *Evolution Strategies as a Scalable Alternative to Reinforcement Learning.* arXiv:1703.03864. — **The canonical "OpenAI-ES" reference; introduces antithetic sampling and the population-parallel formulation that the implementation mirrors.** Cite for both the estimator form and the antithetic CRN variance-reduction trick.
- **Mania, Guy, Recht (2018).** *Simple Random Search of Static Linear Policies is Competitive for Reinforcement Learning.* NeurIPS 2018. — ARS; demonstrates that very small ES populations suffice for low-dimensional parameterizations — directly relevant to learning short B-spline control-point vectors.
- **Nesterov & Spokoiny (2017).** *Random Gradient-Free Minimization of Convex Functions.* Foundations of Computational Mathematics, 17(2). — Theoretical analysis of Gaussian-smoothed finite-difference gradient estimators; provides convergence guarantees and variance bounds for the family of estimators OpenAI-ES belongs to.
- **Vicol, Metz, Sohl-Dickstein (2021).** *Unbiased Gradient Estimation in Unrolled Computation Graphs with Persistent Evolution Strategies.* ICML 2021. — **The most directly comparable methodological reference: PES uses ES specifically to estimate hypergradients through long unrolled training trajectories, exactly the bilevel setting of this thesis.** The current implementation is non-persistent (vanilla ES per outer step), but PES is the natural extension if variance becomes the binding constraint.
- **Metz, Freeman, Schoenholz, Kachman (2021).** *Gradients are Not All You Need.* arXiv:2111.05803. — **Catalogues the failure modes of analytic hypergradients on long unrolled computations (chaotic loss landscapes, exploding gradients) and motivates ES as a robust alternative.** This is the primary methodological justification for offering ES as an option in this codebase.
- **Choromanski, Rowland, Sindhwani, Turner, Weller (2018).** *Structured Evolution with Compact Architectures for Scalable Policy Optimization.* ICML 2018. — Structured perturbation directions for ES; relevant if perturbation dimensionality grows. *(verify citation)*
- **Lehman, Chen, Clune, Stanley (2018).** *ES Is More Than Just a Traditional Finite-Difference Approximator.* GECCO 2018. — Clarifies the relationship between ES and finite differences; useful background for explaining what the estimator actually computes.
- **Glasserman & Yao (1992).** *Some Guidelines and Guarantees for Common Random Numbers.* Management Science, 38(6). — Theoretical foundation for **Common Random Numbers (CRN)** variance reduction. The implementation shares the spherical-noise key across each antithetic pair, and (by necessity — `train_with_noise` contains a `jax.pure_callback` for the batch fetcher that cannot be vmapped) shares the minibatch and init keys across the entire population. Both choices are CRN: paired samples see correlated inner-loop noise so the finite-difference signal isolates the schedule perturbation.

### 2.5 Constrained optimization / projection
The `project_weights` algorithm is a projection onto a non-convex constraint set defined by ∑e^(wᵢ²) = (μ/p)² + T.

- **Boyd & Vandenberghe (2004).** *Convex Optimization.* Cambridge University Press. — Reference for projected-gradient methods (background only — the constraint here is not convex).
- **Nocedal & Wright (2006).** *Numerical Optimization* (2nd ed.). Springer. — Reference for Newton's method; bisection complementarity.
- **Brent (1973).** *Algorithms for Minimization Without Derivatives.* Prentice-Hall. — Brent's method, used in `approx_to_gdp` to invert ε(μ) at startup.

---

## 3. Adjacent / supporting domains

### 3.1 Automatic differentiation
- **Griewank & Walther (2008).** *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation* (2nd ed.). SIAM. — The standard AD textbook.
- **Baydin, Pearlmutter, Radul, Siskind (2018).** *Automatic Differentiation in Machine Learning: A Survey.* JMLR 18(153). — Modern ML-focused AD survey.
- **Griewank (1992).** *Achieving logarithmic growth of temporal and spatial complexity in reverse automatic differentiation.* Optimization Methods and Software, 1(1). — Original gradient checkpointing analysis; underpins `jax.checkpoint` use in `train_with_noise`.
- **Chen, Xu, Zhang, Guestrin (2016).** *Training Deep Nets with Sublinear Memory Cost.* arXiv:1604.06174. — ML-style gradient checkpointing; the directly cited justification for the scan-with-checkpoint pattern.

### 3.2 JAX, Equinox, and the software stack
- **Bradbury, Frostig, Hawkins, Johnson, Leary, Maclaurin, Necula, Paszke, VanderPlas, Wanderman-Milne, Zhang (2018).** *JAX: composable transformations of Python+NumPy programs.* http://github.com/google/jax. — Cite for JAX itself.
- **Kidger & Garcia (2021).** *Equinox: neural networks in JAX via callable PyTrees and filtered transformations.* Differentiable Programming Workshop at NeurIPS 2021 / arXiv:2111.00254. — Cite for `eqx.Module`, `eqx.filter_jit`, `eqx.partition`/`combine`, `eqx.error_if`.
- **DeepMind et al. (2020).** *The DeepMind JAX Ecosystem.* http://github.com/deepmind. — Optax, Chex, etc.

### 3.3 Differentiable programming / differentiating through algorithms
The thesis differentiates through (i) DP-SGD training, (ii) noise/clip schedules, and (iii) a constrained projection — all instances of "differentiable programming."

- **Amos & Kolter (2017).** *OptNet: Differentiable Optimization as a Layer in Neural Networks.* ICML 2017. — Differentiating through optimization problems.
- **Agrawal, Amos, Barratt, Boyd, Diamond, Kolter (2019).** *Differentiable Convex Optimization Layers.* NeurIPS 2019. — General framework for differentiating through cone programs.
- **Blondel et al. (2022).** *Efficient and Modular Implicit Differentiation.* NeurIPS 2022. — JAXopt; implicit differentiation in JAX. *(verify citation)*

### 3.4 Reinforcement learning (legacy framing only)
The codebase's `policy/`, `environments/`, "policy loss" terminology is residual from when this work was framed as RL. The thesis should briefly explain this naming and pivot to the bilevel/hypergradient framing.

- **Sutton & Barto (2018).** *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. — General RL reference; cite only when discussing the legacy framing.
- **Williams (1992).** *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning.* Machine Learning 8. — REINFORCE / policy-gradient origin; cite if explaining why "policy gradient" was the original mental model.

---

## 4. Directly comparable prior work (adaptive / dynamic DP-SGD schedules)

This is the most important section for positioning the thesis's contribution: it must distinguish the **gradient-based, end-to-end-learned** schedule from prior **heuristic** or **runtime-statistic-driven** schedules.

### 4.1 Adaptive clipping (learning C, possibly online)
- **Andrew, Thakkar, McMahan, Ramaswamy (2021).** *Differentially Private Learning with Adaptive Clipping.* NeurIPS 2021. — Adapts clipping norm online via a quantile estimator; the closest prior work to the "learn the clip schedule" half. Important to contrast: theirs adapts at runtime via a private estimator; this thesis pre-computes a schedule via outer-loop optimization.
- **Pichapati, Suresh, Yu, Reddi, Kumar (2019).** *AdaCliP: Adaptive Clipping for Private SGD.* arXiv:1908.07643. — Per-coordinate adaptive clipping.
- **Bu, Wang, Zha, Karypis (2023).** *Automatic Clipping: Differentially Private Deep Learning Made Easier and Stronger.* NeurIPS 2023. — Removes manual clip tuning. *(verify citation year/venue)*

### 4.2 Dynamic / scheduled noise (learning σ over time)
- **Lee & Kifer (2018).** *Concentrated Differentially Private Gradient Descent with Adaptive per-Iteration Privacy Budget.* KDD 2018. — Allocates privacy budget non-uniformly across iterations based on a public validation signal. The most directly comparable σ-scheduling work; the thesis's contribution is doing this end-to-end via gradients rather than via a hand-designed allocation rule.
- **Yu, Zhang, Kim, Hassan, Wang (2019).** *Differentially Private Model Publishing for Deep Learning.* IEEE S&P 2019. — Dynamic privacy budget allocation. *(verify citation)*
- **Wei, Liu et al. (2020).** *Federated Learning with Differential Privacy: Algorithms and Performance Analysis.* IEEE TIFS. — Time-varying noise in federated DP. *(verify exact authors)*
- **Du, Mireshghallah, Berg-Kirkpatrick, Shokri (2021).** *Dynamic Differential-Privacy Preserving SGD.* arXiv. *(verify citation — exists in some form, check exact title/authors)*

### 4.3 Privacy-utility frontier and tuning
- **Papernot, Thakurta (2021).** *Hyperparameter Tuning with Renyi Differential Privacy.* ICLR 2022. — Private hyperparameter selection; relevant for honest accounting if any hyper-tuning is itself done on private data.
- **Mohapatra, Sasy, He, Kamath, Thakkar (2022).** *The Role of Adaptive Optimizers for Honest Private Hyperparameter Selection.* AAAI 2022. — Adaptive optimizers under DP. *(verify citation)*

### 4.4 Scaling DP-SGD (context for why utility matters)
- **Li, Tramèr, Liang, Hashimoto (2022).** *Large Language Models Can Be Strong Differentially Private Learners.* ICLR 2022. — DP fine-tuning of LMs at scale.
- **De, Berrada, Hayes, Smith, Balle (2022).** *Unlocking High-Accuracy Differentially Private Image Classification through Scale.* arXiv:2204.13650. — Demonstrates that careful tuning closes much of the privacy-utility gap; motivates the value of optimal scheduling.
- **Sander, Stock, Sablayrolles (2023).** *TAN Without a Burn: Scaling Laws of DP-SGD.* ICML 2023. — Scaling laws for DP-SGD hyperparameters. *(verify citation)*

### 4.5 Federated learning (where σ scheduling matters most in practice)
- **McMahan, Moore, Ramage, Hampson, Agüera y Arcas (2017).** *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017. — FedAvg.
- **McMahan, Ramage, Talwar, Zhang (2018).** *Learning Differentially Private Recurrent Language Models.* ICLR 2018. — DP-FedAvg; the production setting where this scheduling matters most.

---

## How these connect to the thesis's contribution

- **Sections 1.1–1.3** define the privacy guarantee being respected.
- **Section 1.3 (GDP specifically)** is the accounting framework that makes the per-step μ schedule a natural learnable object — without GDP's clean per-step composition, the projection onto the constraint set would not have a tractable JIT-compatible form.
- **Sections 2.1–2.2** are the methodological core: this work is bilevel optimization with hypergradients.
- **Section 2.4** justifies the alternative ES-based gradient estimator on the `es` branch — used when analytic hypergradients are unreliable (Metz 2021) or when schedule leaves are not amenable to backprop. The implementation specifically borrows the antithetic estimator from Salimans (2017) and NES rank shaping from Wierstra (2014); PES (Vicol 2021) is the unbiased extension if outer-step variance becomes the bottleneck.
- **Section 2.5** justifies the projection algorithm in `project_weights`.
- **Sections 3.1–3.3** justify the implementation: AD theory + JAX/Equinox + differentiable programming.
- **Section 4** is the comparison surface — every paper there should be situated against the thesis's claim that schedules can be **learned end-to-end via the privacy-aware hypergradient**, not approximated by heuristics or estimated online from private statistics.

---

_Appended 2026-05-17: Added Section 2.4 on Evolutionary Strategies (Rechenberg, Schwefel, Hansen, Wierstra/NES, Salimans/OpenAI-ES, Mania/ARS, Nesterov-Spokoiny, Vicol/PES, Metz "Gradients are Not All You Need", Choromanski, Lehman, Glasserman/CRN) to cover the ES estimator landed on the `es` branch. Renumbered the previous 2.4 "Constrained optimization" to 2.5; updated the connecting summary to reflect both changes._

---

_Appended 2026-05-18: Extended the ES estimator with two further mechanisms from the NES family — a natural-gradient log-σ update (sNES, single scalar σ) and adaptation sampling for its learning rate η_σ (Wierstra 2014, §6.2 / Algorithm 7, importance-weighted Mann–Whitney U). Code in `src/environments/nes.py` (`nes_log_sigma_gradient`, `nes_es_step`, `adaptation_sampling_update`); knobs added to `ESConfig`. The existing Wierstra (2014) citation in Section 2.4 now covers three uses (rank-shaped utilities, sNES σ-update, adaptation sampling) rather than one; the additions below expand the methodological lineage and add the U-test primary source._

**Additions to Section 2.4 — NES extensions and adaptation sampling:**

- **Glasmachers, Schaul, Yi, Wierstra, Schmidhuber (2010).** *Exponential Natural Evolution Strategies.* GECCO 2010. — **xNES; the lineage that introduced the multiplicative log-parameter update for the search-distribution scale that `nes_es_step` implements as `new_log_sigma = log_sigma + η_σ · ∇_{log σ} J`.** Cite for the choice to parameterise σ in log space (keeps σ > 0 without projection and makes the natural gradient an additive update).
- **Schaul, Glasmachers, Schmidhuber (2011).** *High Dimensions and Heavy Tails for Natural Evolution Strategies.* GECCO 2011. — **Introduces sNES (separable NES), the variant the implementation tracks: a single scalar σ instead of a full covariance.** Cite for why we sidestep CMA-ES-style covariance adaptation when the perturbed leaves (e.g. B-spline control points) are already low-dimensional.
- **Mann & Whitney (1947).** *On a Test of Whether One of Two Random Variables is Stochastically Larger than the Other.* Annals of Mathematical Statistics, 18(1). — **Primary source for the U-test used inside `adaptation_sampling_update`.** The implementation uses the *weighted* variant (importance weights `w_i ∝ exp(‖ε_i‖² · (1 − 1/c²)/2) / c^d`) to compare the current σ against a hypothetical σ′ = cσ from the same sample set, per Wierstra (2014) §6.2.
- **Wierstra et al. (2014), §6.2 / Algorithm 7 specifically.** Already cited above for rank shaping; reaffirm here for the **adaptation-sampling rule** that grows η_σ multiplicatively when the U-statistic crosses a threshold ρ and decays it toward an init baseline otherwise. The choice of a single hypothetical direction (σ′ = cσ, c > 1) rather than a two-sided test is taken directly from Algorithm 7; the gradient itself decides the direction of σ-motion, so adaptation sampling only modulates the rate.

**Connection to the thesis:** the σ-update and its meta-adaptation matter mainly when the ES estimator is run for many outer steps with a poorly-chosen initial σ — exactly the regime of long bilevel runs on the schedule. Without sNES, σ is a fixed hyperparameter that the user must tune; with sNES + adaptation sampling, σ is auto-tuned online from the same population draws used for the mean-gradient estimate, at no extra inner-loop cost.
