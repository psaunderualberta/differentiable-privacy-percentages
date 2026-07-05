import math

import numpy as np
import pandas as pd

from transfer_curve import load_source_policies, resample_curve
from util.transfer import SourcePolicy


class TestResampleCurve:
    """Resample a source length-T curve onto the target T, endpoint-preserving.

    Resampling is a pure shape-preserving reshape over normalized position spanning
    [0, 1] (linspace, i.e. i/(T-1)), so both the first and last learned step survive
    and no tail step is fabricated by extrapolation.
    """

    def test_resamples_a_linear_ramp_onto_the_target_grid(self):
        # A source curve that is exactly linear in normalized position: v = 1 + 3*pos.
        src_T = 4
        src_pos = np.linspace(0.0, 1.0, src_T)
        values = 1.0 + 3.0 * src_pos

        target_T = 10
        out = resample_curve(values, target_T)

        # Length matches the target, and every point tracks the same linear law
        # evaluated on the target's normalized grid — both endpoints preserved,
        # nothing extrapolated.
        assert len(out) == target_T
        expected = 1.0 + 3.0 * np.linspace(0.0, 1.0, target_T)
        np.testing.assert_allclose(out, expected, rtol=1e-6)


def _schedule_rows(run_id, dataset, eps, T, arch, seed, sigmas, clips):
    """Rows as compile_results_fetch writes them: one per (run_id, inner_step)."""
    return [
        {
            "run_id": run_id,
            "dataset": dataset,
            "eps": eps,
            "T": T,
            "arch_label": arch,
            "seed": seed,
            "inner_step": i,
            "step_norm": i / T,
            "sigma": s,
            "clip": c,
        }
        for i, (s, c) in enumerate(zip(sigmas, clips))
    ]


class TestLoadSourcePolicies:
    """Read schedules.parquet into per-run (SourcePolicy, sigmas, clips) records."""

    def test_groups_by_run_and_orders_by_inner_step(self, tmp_path):
        rows = _schedule_rows(
            "run_a",
            "mnist",
            0.5,
            3,
            "cnn-16x32-head32",
            7,
            sigmas=[2.0, 3.0, 4.0],
            clips=[0.5, 0.6, 0.7],
        )
        rows += _schedule_rows(
            "run_b",
            "cifar-10",
            1.0,
            3,
            "cnn-32x64-head256",
            9,
            sigmas=[1.0, 1.1, 1.2],
            clips=[0.1, 0.2, 0.3],
        )
        # Shuffle so ordering must come from inner_step, not row order.
        df = pd.DataFrame(rows).sample(frac=1.0, random_state=0)
        path = tmp_path / "schedules.parquet"
        df.to_parquet(path, index=False)

        records = load_source_policies(path)

        by_id = {rec[0].run_id: rec for rec in records}
        assert set(by_id) == {"run_a", "run_b"}

        src_a, sigmas_a, clips_a = by_id["run_a"]
        assert isinstance(src_a, SourcePolicy)
        assert src_a.dataset == "mnist"
        assert src_a.eps == 0.5
        assert src_a.T == 3
        assert src_a.arch == "cnn-16x32-head32"
        # Ordered by inner_step regardless of the shuffled parquet.
        np.testing.assert_array_equal(sigmas_a, [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(clips_a, [0.5, 0.6, 0.7])
        # delta/p are source-side provenance absent from the parquet → NaN.
        assert math.isnan(src_a.delta)
        assert math.isnan(src_a.p)


class TestBuildCurveSchedule:
    """Seat a resampled source curve onto the target budget as a RawArraySchedule."""

    def test_sigma_is_resampled_to_target_T_and_binds_the_budget(self):
        from jax import numpy as jnp

        from privacy.gdp_privacy import GDPPrivacyParameters
        from transfer_curve import build_curve_schedule
        from util.transfer import RawArraySchedule

        # Source trained at a different T than the target.
        source_sigmas = np.array([3.0, 3.5, 4.0, 3.5, 3.0])
        source_clips = np.array([0.5, 0.7, 0.9, 0.7, 0.5])
        pp = GDPPrivacyParameters(eps=1.0, delta=1e-5, p=0.01, T=12)

        sched = build_curve_schedule(source_sigmas, source_clips, pp)

        assert isinstance(sched, RawArraySchedule)
        sigmas = np.asarray(sched.get_private_noise_scales())
        clips = np.asarray(sched.get_private_clips())
        # Both curves land at the target T.
        assert len(sigmas) == pp.T
        assert len(clips) == pp.T
        # Matched privacy: the seated sigma curve binds the target DP-PSAC boundary.
        bound = float((pp.mu / pp.p) ** 2 + pp.T)
        used = float(jnp.sum(jnp.exp(1.0 / jnp.asarray(sigmas))))
        np.testing.assert_allclose(used, bound, rtol=1e-4)
        # Clip is privacy-neutral: carried across as the plain resample, untouched by seating.
        np.testing.assert_allclose(clips, resample_curve(source_clips, pp.T), rtol=1e-6)


class TestScheduleDataToResults:
    """Adapt Baseline.generate_schedule_data rows into transfer_rows (seed, acc, loss)."""

    def test_maps_each_rep_row_to_a_seed_indexed_result(self):
        from transfer_curve import schedule_data_to_results

        # One row per rep, as generate_schedule_data emits (step==0 for all).
        df = pd.DataFrame(
            {
                "type": ["Curve Transfer"] * 3,
                "step": [0, 0, 0],
                "loss": [0.42, 0.45, 0.44],
                "accuracy": [0.81, 0.79, 0.80],
                "losses": [[], [], []],
                "accuracies": [[], [], []],
            }
        )

        results = schedule_data_to_results(df)

        # Rep index becomes the seed; accuracy/loss carried through in order.
        assert results == [(0, 0.81, 0.42), (1, 0.79, 0.45), (2, 0.80, 0.44)]
