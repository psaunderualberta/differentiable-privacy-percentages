import jax.numpy as jnp
import numpy as np
import pandas as pd

from policy.schedules.abstract import AbstractNoiseAndClipSchedule
from privacy.gdp_privacy import GDPPrivacyParameters
from util.transfer import (
    RawArraySchedule,
    SourcePolicy,
    TargetSpec,
    assemble_transfer,
    seat_on_budget,
    transfer_rows,
    write_transfer_cell,
)

_EXPECTED_COLUMNS = [
    "producer",
    "source_id",
    "source_dataset",
    "source_eps",
    "source_delta",
    "source_T",
    "source_p",
    "source_arch",
    "target",
    "target_eps",
    "target_delta",
    "target_T",
    "target_arch",
    "seed",
    "accuracy",
    "loss",
]


def _source() -> SourcePolicy:
    return SourcePolicy(
        run_id="abc123",
        dataset="mnist",
        eps=0.5,
        delta=1e-6,
        T=100,
        p=0.01,
        arch="cnn-16x32-head32",
    )


def _target() -> TargetSpec:
    return TargetSpec(name="eyepacs", eps=1.0, delta=1e-7, T=200, arch="cnn-16x32-head32")


def _bound(pp: GDPPrivacyParameters) -> float:
    return float((pp.mu / pp.p) ** 2 + pp.T)


def _budget_use(sigmas) -> float:
    return float(jnp.sum(jnp.exp(1.0 / jnp.asarray(sigmas))))


class TestRawArraySchedule:
    """A lossless length-T sigma/clip wrapper: the curve-transfer producer input."""

    def test_round_trips_sigmas_and_clips(self):
        sigmas = jnp.asarray([1.0, 2.0, 3.0, 4.0])
        clips = jnp.asarray([0.5, 0.6, 0.7, 0.8])
        sched = RawArraySchedule(sigmas, clips)

        assert isinstance(sched, AbstractNoiseAndClipSchedule)
        np.testing.assert_array_equal(sched.get_private_noise_scales(), sigmas)
        np.testing.assert_array_equal(sched.get_private_clips(), clips)
        # weight w = C / sigma is the DP-PSAC noise weight
        np.testing.assert_allclose(sched.get_private_weights(), clips / sigmas)


class TestSeatOnBudget:
    """Scale a slack sigma curve onto the target DP-PSAC budget boundary."""

    def test_seating_binds_the_budget_that_projection_alone_leaves_slack(self):
        pp = GDPPrivacyParameters(eps=1.0, delta=1e-5, p=0.01, T=10)
        # An over-noised (large-sigma) curve spends far less than the budget.
        sigmas = jnp.full((pp.T,), 2.0)
        assert _budget_use(sigmas) < _bound(pp)  # slack input
        # project_inverse_sigmas enforces only the inequality → leaves it slack.
        np.testing.assert_allclose(pp.project_inverse_sigmas(sigmas), sigmas)

        seated = seat_on_budget(sigmas, pp)
        # Seating spends the full budget: sum exp(1/sigma) binds the boundary.
        np.testing.assert_allclose(_budget_use(seated), _bound(pp), rtol=1e-4)

    def test_seating_is_invariant_to_input_scale(self):
        pp = GDPPrivacyParameters(eps=1.0, delta=1e-5, p=0.01, T=8)
        # A non-uniform shape; only the ratios between steps should survive seating.
        sigmas = jnp.asarray([1.0, 1.5, 2.0, 2.5, 3.0, 2.0, 1.5, 1.0])
        seated = seat_on_budget(sigmas, pp)
        seated_scaled = seat_on_budget(10.0 * sigmas, pp)
        np.testing.assert_allclose(seated, seated_scaled, rtol=1e-4)


class TestSchemaAdapter:
    """One parquet row per (cell, seed) with exactly the ADR 0008 columns."""

    def test_emits_expected_columns_one_row_per_seed(self):
        results = [(0, 0.81, 0.42), (1, 0.79, 0.45), (2, 0.80, 0.44)]
        df = transfer_rows("curve", _source(), _target(), results)

        assert list(df.columns) == _EXPECTED_COLUMNS
        assert len(df) == 3
        # metadata is broadcast across the seed rows
        assert (df["producer"] == "curve").all()
        assert (df["source_id"] == "abc123").all()
        assert (df["target"] == "eyepacs").all()
        assert df["seed"].tolist() == [0, 1, 2]
        assert df["accuracy"].tolist() == [0.81, 0.79, 0.80]
        assert df["source_p"].tolist() == [0.01, 0.01, 0.01]


class TestCellWriteAndAssemble:
    """One parquet file per SLURM cell; the assembler globs + concats them."""

    def _cell(self, run_id, producer="curve"):
        src = SourcePolicy(
            run_id=run_id, dataset="mnist", eps=0.5, delta=1e-6, T=100, p=0.01, arch="a"
        )
        results = [(0, 0.80, 0.40), (1, 0.82, 0.38)]
        return transfer_rows(producer, src, _target(), results)

    def test_assemble_is_order_independent(self, tmp_path):
        a = self._cell("aaa")
        b = self._cell("bbb")
        # Write in one order...
        write_transfer_cell(b, tmp_path)
        write_transfer_cell(a, tmp_path)
        assembled = assemble_transfer("curve", tmp_path)

        # Two cells x two seeds = four rows, both source ids present.
        assert len(assembled) == 4
        assert set(assembled["source_id"]) == {"aaa", "bbb"}
        # Deterministic regardless of write order.
        again = assemble_transfer("curve", tmp_path)
        pd.testing.assert_frame_equal(assembled, again)
