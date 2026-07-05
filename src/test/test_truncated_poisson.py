"""Tests for the truncated-Poisson DP-SGD sampler (ADR 0009).

Covers:
- compute_poisson_buffer_size: certificate bound, minimality, monotonicity, cap at N
- poisson_buffer_indices: exact top_k selection, overflow keep-first-B, valid mask
- sum_clipped_per_example_grads masking: B-slot buffer == m valid rows
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy.stats import binom

from privacy.gdp_privacy import compute_poisson_buffer_size
from util.util import poisson_buffer_indices, sum_clipped_per_example_grads


def _certificate(B: int, N: int, batch_size: int, T: int, eps: float) -> float:
    """The additive-delta the truncation costs: (1 + e^eps) * T * P(Binom > B)."""
    p = batch_size / N
    return (1.0 + np.exp(eps)) * T * binom.sf(B, N, p)


class TestComputePoissonBufferSize:
    def test_honors_certificate_and_is_minimal(self):
        N, batch_size, T, eps, delta, c = 60000, 250, 100, 1.0, 1e-6, 1e-3
        B = compute_poisson_buffer_size(N, batch_size, T, eps, delta, c)
        # Returned B meets the bound; B-1 does not (smallest integer).
        assert _certificate(B, N, batch_size, T, eps) <= c * delta
        assert _certificate(B - 1, N, batch_size, T, eps) > c * delta

    def test_monotone_nondecreasing_in_T(self):
        N, batch_size, eps, delta = 60000, 250, 1.0, 1e-6
        sizes = [
            compute_poisson_buffer_size(N, batch_size, T, eps, delta) for T in (10, 100, 1000, 7000)
        ]
        assert sizes == sorted(sizes)
        assert sizes[-1] > sizes[0]

    def test_monotone_nondecreasing_in_eps(self):
        N, batch_size, T, delta = 60000, 250, 1000, 1e-6
        sizes = [
            compute_poisson_buffer_size(N, batch_size, T, eps, delta)
            for eps in (0.5, 1.0, 4.0, 10.0)
        ]
        assert sizes == sorted(sizes)
        assert sizes[-1] > sizes[0]

    def test_capped_at_N_for_small_N_large_p(self):
        # Small-N / large-p regime (california-like): the tail would demand a
        # buffer larger than the dataset, so B is capped at N.
        N, batch_size, T, eps, delta = 100, 95, 7000, 10.0, 1e-9
        B = compute_poisson_buffer_size(N, batch_size, T, eps, delta)
        assert B == N


class TestPoissonBufferIndices:
    def test_no_overflow_valid_set_is_exactly_included_records(self):
        # p small enough that m << B — no overflow. The valid buffer entries
        # must be exactly the Bernoulli-included records {i : u_i < p}.
        key = jr.PRNGKey(0)
        N, p, B = 1000, 0.05, 300
        idxs, valid = poisson_buffer_indices(key, N, p, B)
        u = jr.uniform(key, (N,))
        expected = set(np.where(np.asarray(u) < p)[0].tolist())
        got = set(np.asarray(idxs)[np.asarray(valid)].tolist())
        assert got == expected
        assert int(valid.sum()) == len(expected)

    def test_overflow_keeps_first_B_included_records(self):
        # p*N >> B — the buffer overflows. Every slot is valid, and the buffer
        # holds exactly the lowest-index B of the included records.
        key = jr.PRNGKey(1)
        N, p, B = 1000, 0.5, 100
        idxs, valid = poisson_buffer_indices(key, N, p, B)
        u = jr.uniform(key, (N,))
        included = np.where(np.asarray(u) < p)[0]
        assert included.size > B  # precondition: genuine overflow
        assert bool(valid.all())
        expected = set(included[:B].tolist())
        assert set(np.asarray(idxs).tolist()) == expected


class TestSumClippedPerExampleGradsMasking:
    def test_masked_buffer_equals_sum_over_valid_rows(self):
        # A B-slot buffer with the last rows masked out must produce exactly the
        # same summed clipped gradient as running only the m valid rows.
        key = jr.PRNGKey(2)
        B, d, m = 8, 5, 3
        grads = {"w": jr.normal(key, (B, d))}
        C = jnp.array(1.0)
        valid = jnp.array([True] * m + [False] * (B - m))

        masked = sum_clipped_per_example_grads(grads, C, valid)
        reference = sum_clipped_per_example_grads({"w": grads["w"][:m]}, C)
        assert jnp.allclose(masked["w"], reference["w"], atol=1e-6)
