"""Factory for the outer-loop training-loss + gradient function.

Two modes are supported, selected by ``schedule_optimizer.es.enabled``:

* **Analytic** (default): the existing path, JIT + ``filter_value_and_grad``
  vmapped over ``schedule_batch_size`` independent DP-SGD runs.
* **Evolutionary Strategies**: vanilla antithetic OpenAI-ES with NES
  log-utility rank shaping. Each outer step samples a population of
  perturbations on the leaves selected by ``schedule.es_filter()``, evaluates
  the inner DP-SGD loss with antithetic-pair Common Random Numbers (CRN), and
  returns a finite-difference gradient estimate on the ES-opted-in leaves with
  zero-fill on frozen leaves so the rest of ``main.py`` is unchanged.

In both cases the call signature returned to ``main.py`` is::

    get_training_loss(schedule, mb_key, init_key, noise_keys)
        -> ((loss, (losses, accuracies, val_accs)), grads)

The leading axis of ``noise_keys`` is the sharded parallelism axis. For the
analytic path it is ``schedule_batch_size``; for ES it is ``population_size //
2`` (one CRN key per antithetic pair).
"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, PyTree

from conf.config import ESConfig
from conf.singleton_conf import SingletonConfig
from environments.dp import DPTrainingParams, TrainingStatistics, train_with_noise
from environments.nes import ESState, adaptation_sampling_update, nes_es_step
from policy.schedules.abstract import AbstractNoiseAndClipSchedule


def make_initial_es_state() -> ESState | None:
    """Build the initial ``ESState`` from the active sweep config, or ``None``
    when ES is disabled. ``main.py`` threads this through ``get_training_loss``
    across outer-loop steps."""
    es_conf: ESConfig = SingletonConfig.get_sweep_config_instance().schedule_optimizer.es
    if not es_conf.enabled:
        return None
    sigma = float(es_conf.perturbation_sigma.sample())
    eta_sigma = float(es_conf.eta_sigma.sample())
    return ESState(
        log_sigma=jnp.log(jnp.float32(sigma)),
        eta_sigma=jnp.float32(eta_sigma),
    )


def make_training_loss_fn(env_params: DPTrainingParams):
    """Return the JIT-compiled outer-loop training-loss + gradient function.

    Signature (both modes)::

        get_training_loss(schedule, mb_key, init_key, noise_keys, es_state)
            -> ((loss, (losses, accs, val_accs)), grads, new_es_state)

    For the analytic path ``es_state`` is ignored and ``new_es_state`` is
    just the input echoed back (so ``main.py`` doesn't need to branch).
    """
    es_conf: ESConfig = SingletonConfig.get_sweep_config_instance().schedule_optimizer.es
    if es_conf.enabled:
        population_size = int(es_conf.population_size.sample())
        if population_size % 2 != 0:
            raise ValueError(
                f"ES population_size must be even (antithetic pairs); got {population_size}.",
            )
        return _make_es_training_loss_fn(env_params, population_size, es_conf)
    return _make_analytic_training_loss_fn(env_params)


# ---------------------------------------------------------------------------
# Analytic gradient (existing behaviour)
# ---------------------------------------------------------------------------


def _make_analytic_training_loss_fn(env_params: DPTrainingParams):
    vmapped_train_with_noise = eqx.filter_vmap(
        train_with_noise,
        in_axes=(None, None, None, None, 0),
    )

    @partial(eqx.filter_value_and_grad, has_aux=True)
    def _value_and_grad(
        schedule: AbstractNoiseAndClipSchedule,
        mb_key: PRNGKeyArray,
        init_key: PRNGKeyArray,
        noise_keys: PRNGKeyArray,
    ) -> tuple[Array, TrainingStatistics]:
        _, statistics = vmapped_train_with_noise(
            schedule,
            env_params,
            mb_key,
            init_key,
            noise_keys,
        )
        to_diff = jnp.mean(statistics.val_loss).squeeze()
        return to_diff, statistics

    @eqx.filter_jit
    def get_training_loss(
        schedule: AbstractNoiseAndClipSchedule,
        mb_key: PRNGKeyArray,
        init_key: PRNGKeyArray,
        noise_keys: PRNGKeyArray,
        es_state: ESState | None = None,
    ):
        (loss, aux), grads = _value_and_grad(schedule, mb_key, init_key, noise_keys)
        return (loss, aux), grads, es_state

    return get_training_loss


# ---------------------------------------------------------------------------
# Evolutionary Strategies
# ---------------------------------------------------------------------------


def _make_es_training_loss_fn(
    env_params: DPTrainingParams,
    population_size: int,
    es_conf: ESConfig,
):
    half_pop = population_size // 2
    adaptation_enabled = es_conf.adaptation_enabled
    adaptation_c = float(es_conf.adaptation_c)
    adaptation_rho = float(es_conf.adaptation_rho)
    adaptation_step = float(es_conf.adaptation_step)
    eta_sigma_max = jnp.float32(es_conf.eta_sigma_max)
    eta_sigma_init = jnp.float32(float(es_conf.eta_sigma.sample()))

    def _sample_perturbations(diff: PyTree, key: PRNGKeyArray) -> PyTree:
        """Sample one Gaussian perturbation per pair, shape (half_pop, *leaf)."""
        leaves, treedef = jax.tree.flatten(diff)
        keys = jr.split(key, len(leaves))
        eps_leaves = [
            jr.normal(k, (half_pop, *leaf.shape), dtype=leaf.dtype) for k, leaf in zip(keys, leaves)
        ]
        return jax.tree.unflatten(treedef, eps_leaves)

    @eqx.filter_jit
    def get_training_loss(
        schedule: AbstractNoiseAndClipSchedule,
        mb_key: PRNGKeyArray,
        init_key: PRNGKeyArray,
        noise_keys: PRNGKeyArray,
        es_state: ESState,
    ):
        # noise_keys: shape (half_pop,) of CRN spherical-noise keys (one per
        # antithetic pair). mb_key & init_key are shared across the population
        # — matching the existing analytic vmap pattern — so the
        # pure_callback batch fetcher inside train_with_noise isn't vmapped.
        sigma = jnp.exp(es_state.log_sigma)
        spec = schedule.es_filter()
        diff, static = eqx.partition(schedule, spec)

        # Derive an independent key for ε sampling so mb_key remains untouched.
        eps_key, _ = jr.split(mb_key)
        eps = _sample_perturbations(diff, eps_key)  # leading axis half_pop

        def _eval_pair(
            eps_p: PyTree,
            crn_k: PRNGKeyArray,
        ) -> tuple[Array, Array, Array, Array, Array, Array]:
            """Evaluate antithetic +ε / -ε with shared spherical-noise CRN."""

            def _one_side(sign: float) -> tuple[Array, Array, Array, Array, Array, Array]:
                delta = jax.tree.map(lambda e: sign * sigma * e, eps_p)
                diff_perturbed = jax.tree.map(lambda d, dl: d + dl, diff, delta)
                sched = eqx.combine(diff_perturbed, static)
                sched = sched.project()
                _, statistics = train_with_noise(
                    sched,
                    env_params,
                    mb_key,
                    init_key,
                    crn_k,
                )
                return (
                    statistics.val_loss,
                    statistics.val_accuracy,
                    statistics.test_loss,
                    statistics.test_accuracy,
                    statistics.losses,
                    statistics.accuracies,
                )

            # vl_pos, l_pos, a_pos, va_pos = _one_side(+1.0)
            # vl_neg, l_neg, a_neg, va_neg = _one_side(-1.0)
            signs = jnp.asarray([1.0, -1.0])
            return eqx.filter_vmap(_one_side)(signs)

        # Vmap over the half_pop pairs; mb_key/init_key are closure-shared.
        val_losses, val_accs, test_losses, test_accs, losses, accuracies = eqx.filter_vmap(
            _eval_pair,
            in_axes=(0, 0),
        )(eps, noise_keys)

        # Shapes: val_losses (half_pop, 2); losses (half_pop, 2, T+1); etc.
        # Flatten to population layout (population_size, ...) with order
        # [pos_0, neg_0, pos_1, neg_1, ...] so adjacent indices share a pair.
        def _flatten_pop(x):
            # (half_pop, 2, ...) → (population_size, ...)
            return x.reshape(population_size, *x.shape[2:])

        statistics = TrainingStatistics(
            val_loss=_flatten_pop(val_losses),
            val_accuracy=_flatten_pop(val_accs),
            test_loss=_flatten_pop(test_losses),
            test_accuracy=_flatten_pop(test_accs),
            losses=_flatten_pop(losses),
            accuracies=_flatten_pop(accuracies),
        )
        # F = _flatten_pop(val_losses)
        # losses_pop = _flatten_pop(losses)
        # accs_pop = _flatten_pop(accuracies)
        # val_accs_pop = _flatten_pop(val_accs)

        # Mean-parameter gradient + log-σ natural-gradient update.
        g_diff, new_es_state = nes_es_step(es_state, eps, statistics.val_loss)

        if adaptation_enabled:
            new_eta = adaptation_sampling_update(
                eps=eps,
                fitnesses=statistics.val_loss,
                eta_sigma=new_es_state.eta_sigma,
                eta_sigma_init=eta_sigma_init,
                c=adaptation_c,
                rho=adaptation_rho,
                step=adaptation_step,
                eta_sigma_max=eta_sigma_max,
            )
            new_es_state = ESState(
                log_sigma=new_es_state.log_sigma,
                eta_sigma=new_eta,
            )

        # Zero-fill frozen leaves so the gradient pytree matches `schedule`.
        zero_static = jax.tree.map(
            lambda x: jnp.zeros_like(x) if eqx.is_array(x) else x,
            static,
        )
        grads = eqx.combine(g_diff, zero_static)

        loss = jnp.mean(statistics.val_loss).squeeze()
        return (loss, statistics), grads, new_es_state

    return get_training_loss
