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
from environments.dp import DPTrainingParams, train_with_noise
from policy.schedules.abstract import AbstractNoiseAndClipSchedule


def make_training_loss_fn(env_params: DPTrainingParams):
    """Return the JIT-compiled outer-loop training-loss + gradient function."""
    es_conf: ESConfig = SingletonConfig.get_sweep_config_instance().schedule_optimizer.es
    if es_conf.enabled:
        sigma = float(es_conf.perturbation_sigma.sample())
        population_size = int(es_conf.population_size.sample())
        if population_size % 2 != 0:
            raise ValueError(
                f"ES population_size must be even (antithetic pairs); got {population_size}.",
            )
        return _make_es_training_loss_fn(env_params, sigma, population_size)
    return _make_analytic_training_loss_fn(env_params)


# ---------------------------------------------------------------------------
# Analytic gradient (existing behaviour)
# ---------------------------------------------------------------------------


def _make_analytic_training_loss_fn(env_params: DPTrainingParams):
    vmapped_train_with_noise = eqx.filter_vmap(
        train_with_noise,
        in_axes=(None, None, None, None, 0),
    )

    @eqx.filter_jit
    @partial(eqx.filter_value_and_grad, has_aux=True)
    def get_training_loss(
        schedule: AbstractNoiseAndClipSchedule,
        mb_key: PRNGKeyArray,
        init_key: PRNGKeyArray,
        noise_keys: PRNGKeyArray,
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        _, to_diff, losses, accuracies, val_acc = vmapped_train_with_noise(
            schedule,
            env_params,
            mb_key,
            init_key,
            noise_keys,
        )
        to_diff = jnp.mean(to_diff).squeeze()
        return to_diff, (losses, accuracies, val_acc)

    return get_training_loss


# ---------------------------------------------------------------------------
# Evolutionary Strategies
# ---------------------------------------------------------------------------


def _nes_log_utilities(fitnesses: Array) -> Array:
    """NES log-utility rank shaping (Wierstra et al.).

    For minimisation: lower fitness → higher utility. Returns utilities
    summing to zero, suitable as weights for the OpenAI-ES gradient sum.
    """
    n = fitnesses.shape[0]
    # Rank: rank[i] = number of fitnesses strictly less than fitnesses[i],
    # so the lowest-loss sample gets rank 0 (highest utility).
    order = jnp.argsort(fitnesses)
    ranks = jnp.empty(n, dtype=jnp.int32).at[order].set(jnp.arange(n))
    raw = jnp.maximum(0.0, jnp.log(n / 2 + 1.0) - jnp.log(ranks.astype(jnp.float32) + 1.0))
    return raw / jnp.sum(raw) - 1.0 / n


def _make_es_training_loss_fn(
    env_params: DPTrainingParams,
    sigma: float,
    population_size: int,
):
    half_pop = population_size // 2

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
    ) -> tuple[
        tuple[Array, tuple[Array, Array, Array]],
        AbstractNoiseAndClipSchedule,
    ]:
        # noise_keys: shape (half_pop,) of CRN spherical-noise keys (one per
        # antithetic pair). mb_key & init_key are shared across the population
        # — matching the existing analytic vmap pattern — so the
        # pure_callback batch fetcher inside train_with_noise isn't vmapped.
        spec = schedule.es_filter()
        diff, static = eqx.partition(schedule, spec)

        # Derive an independent key for ε sampling so mb_key remains untouched.
        eps_key, _ = jr.split(mb_key)
        eps = _sample_perturbations(diff, eps_key)  # leading axis half_pop

        def _eval_pair(
            eps_p: PyTree,
            crn_k: PRNGKeyArray,
        ) -> tuple[Array, Array, Array, Array]:
            """Evaluate antithetic +ε / -ε with shared spherical-noise CRN."""

            def _one_side(sign: float) -> tuple[Array, Array, Array, Array]:
                delta = jax.tree.map(lambda e: sign * sigma * e, eps_p)
                diff_perturbed = jax.tree.map(lambda d, dl: d + dl, diff, delta)
                sched = eqx.combine(diff_perturbed, static)
                sched = sched.project()
                _, val_loss, losses, accs, val_acc = train_with_noise(
                    sched,
                    env_params,
                    mb_key,
                    init_key,
                    crn_k,
                )
                return val_loss, losses, accs, val_acc

            vl_pos, l_pos, a_pos, va_pos = _one_side(+1.0)
            vl_neg, l_neg, a_neg, va_neg = _one_side(-1.0)
            return (
                jnp.stack([vl_pos, vl_neg]),
                jnp.stack([l_pos, l_neg]),
                jnp.stack([a_pos, a_neg]),
                jnp.stack([va_pos, va_neg]),
            )

        # Vmap over the half_pop pairs; mb_key/init_key are closure-shared.
        val_losses, losses, accuracies, val_accs = eqx.filter_vmap(
            _eval_pair,
            in_axes=(0, 0),
        )(eps, noise_keys)

        # Shapes: val_losses (half_pop, 2); losses (half_pop, 2, T+1); etc.
        # Flatten to population layout (population_size, ...) with order
        # [pos_0, neg_0, pos_1, neg_1, ...] so adjacent indices share a pair.
        def _flatten_pop(x):
            # (half_pop, 2, ...) → (population_size, ...)
            return x.reshape(population_size, *x.shape[2:])

        F = _flatten_pop(val_losses)
        losses_pop = _flatten_pop(losses)
        accs_pop = _flatten_pop(accuracies)
        val_accs_pop = _flatten_pop(val_accs)

        # NES rank-shaped utilities (sum to 0). Lowest loss → highest utility.
        u = _nes_log_utilities(F)  # (population_size,)
        u_pos = u[0::2]
        u_neg = u[1::2]
        # Loss-minimisation gradient: ∇L ≈ -(1/(Nσ)) Σ (u_pos - u_neg) ε_p
        # (optax does params -= lr*grad, so grad must point toward increasing L).
        w = (u_neg - u_pos) / (population_size * sigma)  # (half_pop,)

        # g_diff[leaf] = Σ_p w[p] * eps[p, leaf]
        g_diff = jax.tree.map(
            lambda e_leaf: jnp.tensordot(w, e_leaf, axes=1),
            eps,
        )

        # Zero-fill frozen leaves so the gradient pytree matches `schedule`.
        zero_static = jax.tree.map(
            lambda x: jnp.zeros_like(x) if eqx.is_array(x) else x,
            static,
        )
        grads = eqx.combine(g_diff, zero_static)

        loss = jnp.mean(F).squeeze()
        return (loss, (losses_pop, accs_pop, val_accs_pop)), grads

    return get_training_loss
