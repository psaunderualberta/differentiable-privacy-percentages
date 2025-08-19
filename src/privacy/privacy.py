from functools import partial

import chex
import equinox as eqx
import jax.debug as debug
import jax.lax as jlax
import jax.numpy as jnp
from jax import jit, vmap
import timeit


def renyi_add_sigma(rho: float, sigma: float) -> chex.Array:
    return jnp.ones(1) * rho + 1 / (2 * sigma)


def renyi_clamp_sigma(rho: float, sigma: float, privacy_budget: float) -> chex.Array:
    tmp_rho = rho + 1 / (2 * sigma)
    return jlax.select(
        tmp_rho > privacy_budget,
        jnp.ones(1) * 1 / (2 * (privacy_budget - rho)),
        jnp.ones(1) * sigma,
    )


def renyi_is_done(rho: float, privacy_budget: float) -> bool:
    return rho >= privacy_budget


""" Copying for later use, if needed"""
#  def _renyi_unsampled_dp(accountantState):
#     rho = accountantState["rho"]
#     assert accountantState["bound"] >= rho

#     return accountantState, math.exp(
#         -((accountantState["bound"] - rho) ** 2) / (4 * rho)
#     )


# def _renyi_sampled_dp(accountantState):
#     p = accountantState["sampling_prob"]
#     a = accountantState["alpha"]
#     prefix = 1 / (a**2 - a)
#     first_term = (1 - p) ** (a - 1) * (1 + (a - 1) * p)
#     second_term = sum(
#         (1 - p) ** (a - k)
#         * (1 + (a - 1) * p)
#         * math.comb(a, k)
#         * math.exp((k - 1) * accountantState["rho"])
#         for k in range(2, a + 1)
#     )

#     sampled_rho = prefix * math.log(first_term + second_term)
#     assert accountantState["bound"] >= sampled_rho
#     return accountantState, math.exp(
#         -((accountantState["bound"] - sampled_rho) ** 2) / (4 * sampled_rho)
#     )


def log_factorial(n, start=1):
    # Computes log(n!)
    n = n.squeeze()
    return jlax.fori_loop(start, n + 1, lambda i, val: jnp.log(i) + val, 0.0)


def log_factorial_stirling(n):
    # Computes log(n!) using Stirling's approximation
    return jlax.select(
        n == 0,
        jnp.zeros_like(n).astype(jnp.float32),
        n * jnp.log(n) - n + 0.5 * jnp.log(2 * jnp.pi * n),
    )


def log_comb(n, k):
    # Computes log(n choose k)
    return log_factorial(n) - log_factorial(k) - log_factorial(n - k)


def log_comb_opt(n, k):
    # Computes log(n choose k) using Stirling's approximation for large n
    n = n.squeeze()
    k = k.squeeze()
    return jlax.cond(n > 20, log_comb_stirling, log_comb, n, k)


def log_comb_stirling(n, k):
    return (
        log_factorial_stirling(n)
        - log_factorial_stirling(k)
        - log_factorial_stirling(n - k)
    )


def _unsampled_gaussian_renyi_dp(alpha, std):
    return alpha / (2 * std**2)


@jit
# very slow, even when jitted. Keeping for reference
def _sampled_gaussian_renyi_dp(alpha, std, p):
    alpha = alpha.squeeze()

    def get_vj(j):
        eps_j = _unsampled_gaussian_renyi_dp(j, std)
        return (
            jnp.log(2)
            + j * jnp.log(p)
            + jlax.select(
                j == 2,
                # edge case when 'j == 2', see derivations in https://arxiv.org/pdf/1808.00087
                jnp.log(jnp.minimum(2 * (jnp.exp(eps_j) - 1), jnp.exp(eps_j))),
                (j - 1) * eps_j,
            )
            + log_comb_opt(alpha, j)
        )

    def stable_logaddexp(j, running_sum):
        # log(exp(running_sum) + exp(get_vj(j))
        return jnp.logaddexp(running_sum, get_vj(j))

    # first element is 0
    logsumexp = jlax.fori_loop(2, alpha + 1, stable_logaddexp, 0)

    return logsumexp / (alpha - 1)

@jit
def _vectorized_sampled_gaussian_renyi_dp(alphas, std, p, nCrs):
    alphas = alphas.squeeze()
    eps_js = _unsampled_gaussian_renyi_dp(alphas, std).reshape(-1)

    @partial(vmap, in_axes=(1,))
    def get_log_vj(j):
        eps_j = eps_js[j - 2]  # 'j' starts at 2
        return (
            jnp.log(2)
            + j * jnp.log(p)
            + jlax.select(
                j == 2,
                # edge case when 'j == 2', see derivations in https://arxiv.org/pdf/1808.00087
                jnp.log(jnp.minimum(2 * (jnp.exp(eps_j) - 1), jnp.exp(eps_j))),
                (j - 1) * eps_j,
            )
        )

    # vjs[j] \approx log_vj(i)
    alphas_reshaped = alphas.reshape(-1, 1)
    vjs = get_log_vj(alphas_reshaped)
    ncr_vjs = nCrs + vjs

    # take lower triangular matrix, where log_comb_opt is defined
    # need to use 'exp' beforehand so the subsequent sum doesn't add tons of 1s
    exp_vjs = jnp.tril(jnp.exp(ncr_vjs))

    # sum over rows, adding '1' as in derivations
    logsumexps = jnp.log(1 + exp_vjs.sum(axis=1)).squeeze()

    # print(logsumexps, alphas)
    return logsumexps / (alphas - 1)


@jit
def get_all_nCr(alphas):
    def get_nCr(alpha, j):
        return log_comb_opt(alpha, j)
        
    all_combinations_log_comb_opt = vmap(
        vmap(get_nCr, in_axes=(None, 0)), in_axes=(0, None)
    )

    alphas_reshaped = alphas.reshape(-1, 1)
    num_alphas = alphas_reshaped.shape[0]
    return all_combinations_log_comb_opt(alphas_reshaped, alphas_reshaped).reshape(
        num_alphas, num_alphas
    )


class PrivacyAccountantState(eqx.Module):
    moments: chex.Array


class PrivacyAccountant(eqx.Module):
    original_moments: chex.Array
    delta_bound: chex.Array
    eps_bound: chex.Array
    sample_prob: chex.Array
    lambdas: chex.Array
    const_ncr: chex.Array

    def __init__(self, moments, delta_bound, eps_bound, sample_prob):
        chex.assert_shape(moments, (None,))

        self.original_moments = jnp.zeros_like(moments).astype(jnp.float32)
        self.lambdas = jnp.arange(2, moments.shape[0] + 2).reshape(1, -1)
        self.delta_bound = jnp.asarray(delta_bound).squeeze()
        self.eps_bound = jnp.asarray(eps_bound).squeeze()
        self.sample_prob = jnp.asarray(sample_prob).squeeze()
        self.const_ncr = get_all_nCr(self.lambdas)
        
    def replace(self, **kwargs) -> 'PrivacyAccountant':
        """Replace attributes in the privacy accountant object with new values, akin to flax's 'dataclass.replace'"""

        els = list(kwargs.items())
        return eqx.tree_at(
            lambda t: tuple(getattr(t, k) for k, _ in els),
            self,
            tuple(v for _, v in els),
            is_leaf=lambda x: x is None,
        )

    def reset_state(self):
        return PrivacyAccountantState(self.original_moments)

    @jit
    def add_sigma(self, state, sigma):
        added_moments = _vectorized_sampled_gaussian_renyi_dp(
            self.lambdas, sigma, self.sample_prob, self.const_ncr
        )
        return PrivacyAccountantState(moments=state.moments + added_moments)

    @partial(jit, static_argnames=('return_new_state',))
    def get_correct_noise(self, state, sigma, return_new_state=True):
        correct_noise, new_state = self._bin_search_sigma(state, sigma)
        if return_new_state:
            return (correct_noise, new_state)
        
        return correct_noise

    def _bin_search_sigma(self, state, sigma, tol=1e-2):
        """
        Test if adding 'sigma' exceeds the privacy budget. If it does, continue doubling it until we find a noise
        value that *doesn't* exceed the budget, then binary search to find the largest noise value that exceeds the budget
        and add it instead. If sigma doesn't exceed the budget, then just return this.

        Note that the code is structured s.t. if 'sigma' doesn't exceed the budget,
        only one call to '_vectorized_sampled_gaussian_renyi_dp' is performed. 
        """
        max_noise = 15

        # Doubling sigma
        def doubling_cond(tup):
            noise_value, new_state = tup

            return jnp.logical_and(
                (noise_value < max_noise).squeeze(),  # guard against many iterations of doubling
                self.is_done(new_state)
            )

        def doubling_body(tup):
            noise_value, _ = tup
            new_noise_value = noise_value * 2.0
            new_state = self.add_sigma(state, new_noise_value)
            return (new_noise_value, new_state)

        current_state = self.add_sigma(state, sigma)
        high, upper_bound_state = jlax.while_loop(doubling_cond, doubling_body, (sigma, current_state))

        # Binary Search
        def cond(loop_state):
            low, high, _ = loop_state
            return (low + tol < high).squeeze()

        def body(loop_state):
            low, high, _ = loop_state
            midpoint = (low + high) / 2.0
            new_state = self.add_sigma(state, midpoint)
            invalid_noise = self.is_done(new_state)
            low = jnp.where(invalid_noise, midpoint, low)
            high = jnp.where(invalid_noise, high, midpoint)
            return (low, high, new_state)

        low, _, final_state = jlax.while_loop(cond, body, (sigma, high, upper_bound_state))

        # return low to subtly exceed budget if the original sigma exeeded
        low = jnp.clip(low, max=max_noise)
        return low, final_state

    def get_privacy_expenditure(self, state):
        return (
            jnp.min(jnp.log(1 / self.delta_bound) / (self.lambdas - 1) + state.moments),
            jnp.min(jnp.exp((self.lambdas - 1) * (state.moments - self.eps_bound))),
        )

    def is_done(self, state):
        return jnp.all(self.get_privacy_expenditure(state)[0] >= self.eps_bound).squeeze()

    def get_eps_bound(self):
        return self.eps_bound

    def get_delta_bound(self):
        return self.delta_bound


def test_equal(a, b):  #
    chex.assert_trees_all_equal(a, b)


def test_close(a, b, tol=10**-5):
    chex.assert_trees_all_close(a, b, atol=tol, rtol=tol)
    # pass


def test_sampled_gaussian_renyi_dp():
    jitted = jit(_vectorized_sampled_gaussian_renyi_dp)  # shorthand
    std, p = jnp.array(0.5), jnp.array(0.001)

    # computed using desmos
    answers = jnp.array(
        [
            0.000109190338584,
            0.000326442653517,
            0.0178265300241,
            1.53926581116,
            12.4017327101,
            22.6483441426,
        ]
    )

    moments_to_test = [3, 4, 5, 6]
    for i in moments_to_test:
        moments = jnp.arange(2, i).reshape(1, -1)
        nCrs = get_all_nCr(moments)
        target = answers[:i-2]
        test_close(jitted(moments, std, p, nCrs), target)
    # test_close(jitted(jnp.arange(2, 11).reshape(1, -1), std, p), 12.4017327101)
    # test_close(jitted(jnp.arange(2, 16).reshape(1, -1), std, p), 22.6483441426)
    # test_close(jitted(1000, std, p), 22.6483441426)


def test_jittable_create_privacy_accountant():
    def jitted(moments, delta_bound, eps_bound, sample_prob):

        accountant = PrivacyAccountant(moments, delta_bound, eps_bound, sample_prob)
        return accountant, accountant.reset_state()

    # Test the creation of the PrivacyAccountant
    moments = jnp.arange(2, 32).reshape(-1)
    delta_bound = jnp.array(0.05)
    eps_bound = jnp.array(0.1)
    sample_prob = jnp.array(0.01)

    jitted_accountant, jitted_state = jit(jitted)(
        moments, delta_bound, eps_bound, sample_prob
    )


def test_add_sigma():
    # Test the add_sigma method
    moments = jnp.arange(2, 5).reshape(-1)
    delta_bound = jnp.array(10**-5)
    eps_bound = jnp.array(2.0)
    sample_prob = jnp.array(0.01)

    accountant = PrivacyAccountant(moments, delta_bound, eps_bound, sample_prob)
    state = accountant.reset_state()
    sigma = 0.5

    prior_moments = state.moments
    state = accountant.add_sigma(state, sigma)

    # Check if the moments have been updated correctly
    expected_moments = prior_moments + vmap(
        _sampled_gaussian_renyi_dp, in_axes=(1, None, None)
    )(accountant.lambdas, sigma, accountant.sample_prob)

    chex.assert_trees_all_equal(state.moments, expected_moments)


def profile():
    moments = jnp.arange(50).reshape(-1)
    delta_bound = jnp.array(1e-6)
    eps_bound = jnp.array(2.0)
    sample_prob = jnp.array(0.01)

    accountant = PrivacyAccountant(moments, delta_bound, eps_bound, sample_prob)
    state = accountant.reset_state()

    accountant.get_correct_noise(state, 1e-10)
    print(timeit.timeit("accountant.get_correct_noise(state, 1e-10)", globals=locals(), number=1000))


if __name__ == "__main__":
    # test_sampled_gaussian_renyi_dp()
    profile()
    # test_add_sigma()
    # test_jittable_create_privacy_accountant()
