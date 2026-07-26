"""Behaviour of the experiment matrix: the ladders and the runs built from them.

These tests pin the *matrix* — which architectures are run, under which arms, on
which datasets — because it is the thing the launch is committed to and there is
no cheap way to notice a wrong one after 480 SLURM jobs are in flight.
"""

from __future__ import annotations

import pytest

import create_experiments as ce
from conf.config_util import dist_config_helper
from conf.optimizer_config import AdamConfig, SGDConfig
from experiments.architectures import LADDERS
from networks.cnn.config import CNNConfig
from networks.mlp.config import MLPConfig


def _sgd(momentum: float, distribution: str = "constant") -> SGDConfig:
    return SGDConfig(
        learning_rate=dist_config_helper(value=0.1, distribution="constant"),
        momentum=dist_config_helper(value=momentum, distribution=distribution),
    )


class TestLadders:
    def test_mlp_width_ladder_rungs(self):
        widths = [a.hidden_sizes for a in LADDERS["mlp-width"]]
        assert widths == [(64,), (128,), (512,)]

    def test_cnn_width_ladder_rungs(self):
        channels = [a.channels for a in LADDERS["cnn-width"]]
        assert channels == [(8, 16), (16, 32), (32, 64)]

    def test_cnn_depth_ladder_rungs(self):
        archs = LADDERS["cnn-depth"]
        assert [a.channels for a in archs] == [(16,), (16, 16), (16, 16, 16), (16, 16, 16, 16)]

    def test_no_other_ladders(self):
        assert set(LADDERS) == {"mlp-width", "cnn-width", "cnn-depth"}

    def test_every_rung_is_a_network_config(self):
        for archs in LADDERS.values():
            for arch in archs:
                assert isinstance(arch, MLPConfig | CNNConfig)


class TestOptTag:
    """The arm (inner SGD momentum) must survive into the run's optimizer tag."""

    def test_momentum_arms_get_distinct_tags(self):
        assert ce._opt_tag(_sgd(0.9)) == "sgd-m0.9"
        assert ce._opt_tag(_sgd(0.0)) == "sgd-m0.0"

    def test_optimizers_without_momentum_stay_bare(self):
        adam = AdamConfig(learning_rate=dist_config_helper(value=1e-3, distribution="constant"))
        assert ce._opt_tag(adam) == "adam"

    def test_swept_momentum_is_rejected(self):
        # A per-run continuous momentum cannot serve as a categorical arm split.
        with pytest.raises(AssertionError):
            ce._opt_tag(_sgd(0.9, distribution="uniform"))


class TestArchLabelInjectivity:
    """``_arch_label`` keys a forest-plot row, so two archs may not share one."""

    def test_real_matrix_has_no_label_collision(self):
        labels = [ce._arch_label(arch) for arch, _tags in ce._arch_ladder_tags()]
        assert len(labels) == len(set(labels))

    def test_distinct_archs_sharing_a_label_are_rejected(self, monkeypatch):
        # The label drops kernel/stride/padding, so these two silently merge.
        block = {"pool_kernel_size": 2, "mlp": MLPConfig(hidden_sizes=(64,))}
        monkeypatch.setattr(
            ce,
            "LADDERS",
            {
                "a": [
                    CNNConfig(
                        channels=(16,), kernel_sizes=(3,), paddings=(1,), strides=(1,), **block
                    )
                ],
                "b": [
                    CNNConfig(
                        channels=(16,), kernel_sizes=(8,), paddings=(2,), strides=(2,), **block
                    )
                ],
            },
        )
        with pytest.raises(AssertionError):
            ce._arch_ladder_tags()


class TestBuiltMatrix:
    """The concrete set of runs the launcher would create."""

    @pytest.fixture(scope="class")
    def experiments(self):
        return ce._build_experiments()

    def test_both_momentum_arms_are_built(self, experiments):
        assert set(experiments) == {"sgd-m0.9", "sgd-m0.0"}

    def test_t_sweep_excludes_cifar(self, experiments):
        for bucket in experiments.values():
            datasets = {c.dataset for tags, _g, _n, c in bucket if "T-sweep" in tags}
            assert datasets == {"mnist", "fashion-mnist"}

    def test_arch_axis_includes_cifar(self, experiments):
        for bucket in experiments.values():
            datasets = {c.dataset for tags, _g, _n, c in bucket if "arch" in tags}
            assert datasets == {"mnist", "fashion-mnist", "cifar-10"}

    def test_arch_axis_run_count(self, experiments):
        # 10 archs x 3 datasets x 1 eps x 8 seeds, per arm.
        for bucket in experiments.values():
            arch_runs = [r for r in bucket if "arch" in r[0]]
            assert len(arch_runs) == 240

    def test_every_run_shares_the_outer_step_budget(self, experiments):
        for bucket in experiments.values():
            assert {c.num_outer_steps for _t, _g, _n, c in bucket} == {1000}

    def test_arms_differ_only_in_inner_momentum(self, experiments):
        momenta = {
            tag: {c.env.optimizer.momentum.value for _t, _g, _n, c in bucket}
            for tag, bucket in experiments.items()
        }
        assert momenta == {"sgd-m0.9": {0.9}, "sgd-m0.0": {0.0}}
