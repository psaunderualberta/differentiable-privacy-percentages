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

### 2.6 Riemannian / manifold optimization (optimizing on the privacy-budget surface)
The schedule outer loop is being reframed (ADR-0008) as optimization on the submanifold `M = {g = B}` where the privacy budget binds. The **Riemannian gradient** (orthogonal projection of the ambient ∇L onto the tangent space `T_θM`) followed by a cheap **retraction** recovers KKT points at scaling-retraction cost — avoiding the systematically-biased schedule *shape* that a cheap but wrongly-directed projection (correcting along a ray ≠ ∇g) would converge to. Fixed-β heavy-ball momentum is added via **projection-based vector transport** of the buffer. This is the constrained-optimization pillar of the method distinct from the unconstrained hypergradient (§2.1–2.2) and the Euclidean projection of §2.5.

- **Absil, Mahony, Sepulchre (2008).** *Optimization Algorithms on Matrix Manifolds.* Princeton University Press. — Foundational text; defines **retractions** and **vector transport**, the operations the scaling retraction and projection-based momentum transport instantiate. Cite Ch. 4 / §8.1 for the tangent/normal decomposition and retraction conditions.
- **Boumal (2023).** *An Introduction to Optimization on Smooth Manifolds.* Cambridge University Press. — Modern textbook and the **primary reference for the derivation**: Riemannian gradient as tangent projection on a level-set submanifold, first-order retractions, and O(1/K) convergence to stationary points (Ch. 4–5, 10).
- **Absil & Malick (2012).** *Projection-like Retractions on Matrix Manifolds.* SIAM J. Optimization, 22(1). — Establishes that Euclidean nearest-point projection is a (second-order) retraction and that cheaper maps qualify as valid first-order retractions — **the licence for replacing the exact nearest-point projection with the cheap scaling retraction** while keeping first-order convergence.
- **Bonnabel (2013).** *Stochastic Gradient Descent on Riemannian Manifolds.* IEEE Transactions on Automatic Control, 58(9). — Foundational almost-sure convergence of Riemannian SGD; cite for the stochastic outer loop (fresh network each step ⇒ stochastic hypergradient).
- **Boumal, Absil, Cartis (2019).** *Global Rates of Convergence for Nonconvex Optimization on Manifolds.* IMA J. Numerical Analysis, 39(1). — O(1/K) to a **stationary** point without convexity — matches the honest caveat that the guarantee is stationary/KKT, not global.
- **Zhang & Sra (2016).** *First-order Methods for Geodesically Convex Optimization.* COLT 2016. — Convergence theory for Riemannian first-order methods; background for the descent-lemma argument.
- **Sato, Kasai, Mishra (2019).** *Riemannian Stochastic Variance Reduced Gradient Algorithm with Retraction and Vector Transport.* SIAM J. Optimization, 29(2):1444–1472. — Riemannian stochastic methods combining **retraction + vector transport**; the convergence setting that justifies transporting the momentum buffer by re-projection rather than parallel transport. Citation verified 2026-07-02; in thesis `ref.bib` as `sato2019riemannian`.
- **Alimisis, Orvieto, Bécigneul, Lucchi (2020).** *A Continuous-time Perspective for Modeling Acceleration in Riemannian Optimization.* AISTATS 2020 (PMLR v108), pp. 1297–1307. — Riemannian momentum/acceleration; the assumptions under which fixed-β heavy-ball with projection-based transport is convergent on a compact embedded submanifold. Citation verified 2026-07-02; in thesis `ref.bib` as `alimisis2020continuous`.

Connection: the DP-PSAC decoupling (§4.1 / appended below) collapses the constraint to a **σ-only** manifold, and the manifold normal `n_θ = ∇_θ g` is obtained by **autodiff of the closed-form RDP budget** — we differentiate the *constraint*, never the (non-differentiable, post-step) retraction, which is what makes cheap retractions legal here.

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

### 3.5 Symbolic regression (the schedule-distillation stage)
A **post-hoc** stage of the method fits closed-form laws to the learned σ/C/μ schedules so that the result reads as an interpretable schedule equation rather than a black-box array. The implementation uses **PySR** with a `TemplateExpressionSpec` (`src/symbolic_regression.py :: PySRConfig`, `src/sr_category.py`): it learns one **universal shape** `f(t/T)` shared across runs plus a small set of free **per-condition constants** `p1…pK` (default K=3), indexed by the condition `(dataset, ε, T, arch_label)` — see `docs/adr/0006`. Fit quality is assessed with position/scale-aware metrics (NRMSE-by-mean, RMSLE, extremum-location error) and a privacy-budget validity check, deliberately *not* correlation — see `docs/adr/0001`.

- **Koza (1992).** *Genetic Programming: On the Programming of Computers by Means of Natural Selection.* MIT Press. — Foundational tree-based genetic programming; the search paradigm PySR implements. Cite for the GP lineage of the symbolic search.
- **Schmidt & Lipson (2009).** *Distilling Free-Form Natural Laws from Experimental Data.* Science, 324(5923). — The canonical "discover an equation from data" paper (Eureqa); establishes the **Pareto accuracy-vs-parsimony** framing that motivates reporting an equation at a chosen complexity. The closest spiritual antecedent to distilling a schedule law from learned curves.
- **Cranmer (2023).** *Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl.* arXiv:2305.01582. — **The tool actually used.** Cite for PySR itself, the multi-population evolutionary search, the complexity/loss Pareto front, and the `TemplateExpressionSpec` mechanism that encodes the shared-shape-plus-constants structure used here.
- **Kommenda, Burlacu, Kronberger, Affenzeller (2020).** *Parameter identification for symbolic regression using nonlinear least squares.* Genetic Programming and Evolvable Machines, 21. — Optimising the numeric constants embedded in an evolved expression via inner least-squares. Directly relevant: the per-condition constants `p1…pK` are exactly such embedded parameters that PySR must fit, not just discover symbolically.
- **Udrescu & Tegmark (2020).** *AI Feynman: A physics-inspired method for symbolic regression.* Science Advances, 6(16). — Dimensional-analysis- and symmetry-guided SR; background for how structural priors (here, a fixed template shape) reduce the search space.
- **Udrescu, Tan, Feng, Neto, Wu, Tegmark (2020).** *AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity.* NeurIPS 2020. — Pareto-optimal equation selection; reinforces the complexity-vs-accuracy tradeoff used to pick the reported equation. *(verify citation)*
- **Petersen, Larma, Mundhenk, Santiago, Kim, Kim (2021).** *Deep Symbolic Regression: Recovering Mathematical Expressions from Data via Risk-Seeking Policy Gradients.* ICLR 2021. — Neural/RL-based SR; cite as the contrasting (non-GP) branch of the SR landscape, to justify why an evolutionary tool (PySR) was chosen over an autoregressive generator for a low-dimensional, low-data fitting problem.
- **La Cava, Orzechowski, Burlacu, de França, Virgolin, Jin, Kommenda, Moore (2021).** *Contemporary Symbolic Regression Methods and their Relative Performance (SRBench).* NeurIPS 2021 Datasets & Benchmarks. — Standard SR benchmark suite; cite when justifying PySR's competitiveness and when situating the fit-metric choices against community practice. *(verify citation)*

---

## 4. Directly comparable prior work (adaptive / dynamic DP-SGD schedules)

This is the most important section for positioning the thesis's contribution: it must distinguish the **gradient-based, end-to-end-learned** schedule from prior **heuristic** or **runtime-statistic-driven** schedules.

### 4.1 Adaptive clipping (learning C, possibly online)
- **Andrew, Thakkar, McMahan, Ramaswamy (2021).** *Differentially Private Learning with Adaptive Clipping.* NeurIPS 2021. — Adapts clipping norm online via a quantile estimator; the closest prior work to the "learn the clip schedule" half. Important to contrast: theirs adapts at runtime via a private estimator; this thesis pre-computes a schedule via outer-loop optimization.
- **Pichapati, Suresh, Yu, Reddi, Kumar (2019).** *AdaCliP: Adaptive Clipping for Private SGD.* arXiv:1908.07643. — Per-coordinate adaptive clipping.
- **Bu, Wang, Zha, Karypis (2023).** *Automatic Clipping: Differentially Private Deep Learning Made Easier and Stronger.* NeurIPS 2023 (Advances in Neural Information Processing Systems 36). — Removes manual clip tuning. In thesis `ref.bib` as `bu2023automatic`.

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

### 4.6 Symbolic distillation of learned schedules into closed-form laws
The contribution is not only the learned σ/C schedule but its **distillation into a transferable closed-form law** (the SR stage, §3.5). This section positions that against prior work that distils a learned/black-box model into a symbolic surrogate.

- **Cranmer, Sanchez-Gonzalez, Battaglia, Xu, Cranmer, Spergel, Ho (2020).** *Discovering Symbolic Models from Deep Learning with Inductive Biases.* NeurIPS 2020. — **The most direct methodological analogue:** train a flexible model (a GNN), then run symbolic regression on its internal functions to extract a human-readable equation. This thesis does the same one level up — it symbolically distils a *learned hyperparameter schedule* (itself the output of bilevel optimisation) rather than a network's message function. Cite to frame the SR stage as "symbolic distillation of a learned object."
- **Schmidt & Lipson (2009).** *Distilling Free-Form Natural Laws from Experimental Data.* Science, 324(5923). — Already in §3.5; reaffirm here as the lineage for "recover the law behind observed curves." The contrast is that the curves here are *optimised artefacts under a privacy constraint*, not natural-system measurements, so the distilled law must also satisfy the GDP budget (validity check, `docs/adr/0001`).
- **Distinguishing claim.** Prior adaptive/dynamic DP-SGD work (§4.1–4.2) produces schedules either by hand-design or by online private statistics; none yields a closed-form, condition-parameterised schedule *law*. The novelty surface for §4.6 is the two-stage pipeline — learn the schedule via the privacy-aware hypergradient (§2.1–2.2), then symbolically distil a `f(t/T)`-plus-per-condition-constants law that is interpretable, transferable, and privacy-budget-valid.

---

## How these connect to the thesis's contribution

- **Sections 1.1–1.3** define the privacy guarantee being respected.
- **Section 1.3 (GDP specifically)** is the accounting framework that makes the per-step μ schedule a natural learnable object — without GDP's clean per-step composition, the projection onto the constraint set would not have a tractable JIT-compatible form.
- **Sections 2.1–2.2** are the methodological core: this work is bilevel optimization with hypergradients.
- **Section 2.4** justifies the alternative ES-based gradient estimator on the `es` branch — used when analytic hypergradients are unreliable (Metz 2021) or when schedule leaves are not amenable to backprop. The implementation specifically borrows the antithetic estimator from Salimans (2017) and NES rank shaping from Wierstra (2014); PES (Vicol 2021) is the unbiased extension if outer-step variance becomes the bottleneck.
- **Section 2.5** justifies the projection algorithm in `project_weights`.
- **Section 2.6** justifies the planned Riemannian-gradient + retraction outer loop (ADR-0008): tangent-projecting the hypergradient onto the privacy-budget manifold and retracting cheaply recovers KKT points that a cheap scaling projection alone would miss (biased schedule shape). The σ-only manifold and autodiff-computed normal follow from the DP-PSAC decoupling.
- **Sections 3.5 + 4.6** cover the symbolic-distillation stage: §3.5 is the SR tooling/lineage (Koza GP → Schmidt-Lipson/Eureqa → Cranmer/PySR, with Kommenda for constant fitting), and §4.6 positions distilling a *learned schedule* into a closed-form law against Cranmer (2020)'s symbolic distillation of deep models — the part of the contribution that turns a learned array into a transferable, privacy-budget-valid equation.
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

---

_Appended 2026-06-29: Added Section 3.5 (symbolic regression as the schedule-distillation stage — Koza GP, Schmidt-Lipson/Eureqa, Cranmer/PySR + `TemplateExpressionSpec`, Kommenda nonlinear-LSQ constant fitting, Udrescu-Tegmark AI Feynman 1.0/2.0, Petersen Deep SR, La Cava SRBench) and Section 4.6 (symbolic distillation of learned schedules — Cranmer 2020 "Discovering Symbolic Models from Deep Learning", positioned as the closest analogue, with the distinguishing claim against §4.1–4.2). Grounded in `src/symbolic_regression.py`, `src/sr_category.py`, and `docs/adr/0001` & `0006`. Updated the connecting summary with a §3.5/§4.6 bullet._

---

_Appended 2026-07-02: Added Section 2.6 (Riemannian / manifold optimization — Absil-Mahony-Sepulchre, Boumal, Absil-Malick projection-like retractions, Bonnabel Riemannian SGD, Boumal-Absil-Cartis nonconvex rates, Zhang-Sra, Sato-Kasai-Mishra retraction+transport, Alimisis Riemannian acceleration) for the planned Riemannian-gradient + retraction outer loop. Grounded in `docs/adr/0007` (accounting model) & `docs/adr/0008` (projection) and the Riemannian section of `thesis/notes/privacy-accountant-and-projection.md`. Below are additions folding into the existing accounting/clipping sections rather than new subsections — they capture the migration off GDP decided this session._

**Additions to §1.3 / §1.4 — exact accounting, fixed-size-WOR subsampling, and adjacency:**

The thesis is migrating off the GDP CLT accountant (Bu-Dong-Long-Su, §1.3) because it applies **Poisson-subsampling** amplification to a sampler that is actually **fixed-size without replacement** (`approx_max_k` in `environments/dp.py`) and reports a CLT approximation rather than a bound. Target: **RDP on a fixed integer order α per projection, with α\* adaptive between steps**; PLD as a **post-hoc exact-reporting** accountant only.

- **Wang, Balle, Kasiviswanathan (2019)** *(already §1.3/§1.4)* — reaffirm as **the** without-replacement / subsampled RDP amplification and analytical moments accountant; its implementation in **autodp** (Yu-Xiang Wang's library) is the intended **validation reference** for the from-scratch JAX RDP (Google `dp_accounting`/Opacus default to Poisson and cannot check the WOR bound).
- **Balle, Barthe, Gaboardi (2018)** *(already §1.4)* — reaffirm: gives tight amplification for **remove / add / substitute** neighbouring relations under both Poisson and sampling-without-replacement; the source for the fixed-size-WOR + **replace-one** bound.
- **Zhu & Wang (2019).** *Poisson Subsampled Rényi Differential Privacy.* ICML 2019. — Contrasts Poisson against other subsampling schemes; motivates matching the accountant to the actual (non-Poisson) sampler rather than borrowing Poisson formulas. *(verify citation)*
- **Kifer & Machanavajjhala (2011).** *No Free Lunch in Data Privacy.* SIGMOD 2011. — **Bounded** (substitution / replace-one, fixed n) vs **unbounded** (add/remove, variable n) DP; the formal basis for the adjacency decision. Replace-one is the natural relation for a fixed-size sampler and carries **sensitivity 2C** (⇒ per-step μ doubles vs add/remove).
- **Desfontaines & Pejó (2020).** *SoK: Differential Privacies.* PoPETs 2020(2). — Taxonomy of DP variants and neighbouring relations; cite for precise adjacency terminology and for situating the replace-one choice.
- **Sommer, Meiser, Mohammadi (2019).** *Privacy Loss Classes: The Central Limit Theorem in Differential Privacy.* PoPETs 2019(2). — Privacy Loss Distribution (PLD) foundations; the exact composition object used for post-hoc reporting.
- **Meiser & Mohammadi (2018).** *Tight on Budget? Tight Bounds for r-Fold Approximate Differential Privacy.* CCS 2018. — "Privacy buckets" / numerical PLD composition; background for the PLD accountant.
- **Doroshenko, Ghazi, Kamath, Kumar, Manurangsi (2022).** *Connect the Dots: Tighter Discrete Approximations of Privacy Loss Distributions.* PoPETs 2022(4). — Modern PLD discretisation; the kind of accountant used to report an exact ε for the final schedule. Complements Koskela (2020) FFT and Gopi-Lee-Wutschitz (2021) PRV, already in §1.3. **PLD is deliberately post-hoc only** — its non-separable convolution budget has no cheap ∇g to project onto during training (see §2.6 / ADR-0007).

**Additions to §4.1 — DP-PSAC / automatic clipping and the clip↔privacy decoupling:**

The **primary methodology** uses a *decoupled* schedule (`DecoupledSigmaAndClipSchedule`) in the **DP-PSAC / automatic-clipping** regime: per-sample gradients are normalised so the clip threshold `C` acts as a pure scaling (learning-rate-like) constant and **cancels from the per-step privacy cost** (`μ_step = 2/σ_mult`). This is what makes the privacy constraint **σ-only** and the manifold of §2.6 low-dimensional; the clip schedule is a free utility knob under plain Euclidean SGD.

- **Bu, Wang, Zha, Karypis (2023)** *(already §4.1)* — reaffirm as the key enabler: **automatic (normalised) clipping** removes clip tuning by making `C` a scaling constant decoupled from the DP guarantee. The decoupled schedule exploits exactly this decoupling so that only σ enters the budget.
- **Xia, Shen, Yao, Fu, Xu, Xu, Fu (2023).** *DP-PSAC: Differentially Private Learning with Per-Sample Adaptive Clipping.* AAAI 2023. — Per-sample adaptive clipping; the named framing in which the clip behaves as an adaptive per-sample scale rather than a privacy-cost lever. Cite as the direct antecedent of the decoupled σ/C parameterisation. Already in the thesis `ref.bib` as `Xia_Shen_Yao_Fu_Xu_Xu_Fu_2023`.
