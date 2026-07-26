"""Ordering of the stateful DP-SGD inner step (ADR 0013).

Step *t* must clip and count against the *incoming* threshold ``C_t`` and derive
``C_{t+1}`` afterwards, matching Andrew et al.'s Algorithm 1. Applying the freshly
updated threshold to the batch it was derived from leaked the batch a second way and
made ``c_0`` almost meaningless.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest
from jaxtyping import Array, PRNGKeyArray

from conf.config import (
    Config,
    EnvConfig,
    ScheduleOptimizerConfig,
    SweepConfig,
    WandbConfig,
)
from conf.singleton_conf import SingletonConfig
from environments.dp import train_with_stateful_noise
from environments.dp_params import DPTrainingParams
from networks.mlp.config import MLPConfig
from networks.mlp.MLP import MLP
from policy.stateful_schedules.abstract import (
    AbstractScheduleState,
    AbstractStatefulNoiseAndClipSchedule,
)
from policy.stateful_schedules.median_gradient import (
    StatefulMedianGradientNoiseAndClipSchedule,
)
from privacy.gdp_privacy import GDPPrivacyParameters
from util.dataloaders import DatasetLoader
from util.util import reinit_model

N_TRAIN = 64
N_VAL = 8
N_FEATURES = 4
N_CLASSES = 2
BATCH_SIZE = 8


class _FixedState(AbstractScheduleState):
    C: Array
    sigma: Array

    def get_clip(self) -> Array:
        return self.C

    def get_noise(self) -> Array:
        return self.sigma


class _ZeroThenOpenSchedule(AbstractStatefulNoiseAndClipSchedule):
    """Clips everything away on the first step, then lets gradients through.

    Noise is zero throughout, so the model moves if and only if a step clipped at a
    non-zero threshold. The initial threshold is 0, so under the correct ordering the
    first step leaves the model exactly where it started.
    """

    iteration_array: Array

    def get_initial_state(self) -> _FixedState:
        return _FixedState(C=jnp.asarray(0.0), sigma=jnp.asarray(0.0))

    def update_state(
        self,
        state: AbstractScheduleState,
        grads: Array,
        iter: Array,
        batch_x: Array,
        batch_y: Array,
        valid: Array,
        key: PRNGKeyArray,
    ) -> _FixedState:
        return _FixedState(C=jnp.asarray(1e6), sigma=jnp.asarray(0.0))

    def get_logging_schemas(self):
        return []

    def get_loggables(self, force=False):
        return []


@pytest.fixture
def loader(tmp_path) -> DatasetLoader:
    rng = np.random.default_rng(0)
    paths = {}
    for name, n in (("x", N_TRAIN), ("val_x", N_VAL)):
        paths[name] = str(tmp_path / f"{name}.npy")
        np.save(paths[name], rng.normal(size=(n, N_FEATURES)).astype(np.float32))
    for name, n in (("y", N_TRAIN), ("val_y", N_VAL)):
        paths[name] = str(tmp_path / f"{name}.npy")
        labels = np.eye(N_CLASSES, dtype=np.float32)[rng.integers(0, N_CLASSES, size=n)]
        np.save(paths[name], labels)

    return DatasetLoader(
        x_path=paths["x"],
        y_path=paths["y"],
        val_x_path=paths["val_x"],
        val_y_path=paths["val_y"],
        n_train=N_TRAIN,
        n_val=4,
        n_test=4,
        sample_shape=(N_FEATURES,),
        label_shape=(N_CLASSES,),
        dataset_name="california",  # float32 passthrough preprocessing for synthetic data
        val_chunk_size=4,
    )


@pytest.fixture
def singleton():
    SingletonConfig.config = Config(
        wandb_conf=WandbConfig(),
        sweep=SweepConfig(
            env=EnvConfig(batch_size=BATCH_SIZE, num_training_steps=1),
            schedule_optimizer=ScheduleOptimizerConfig(max_sigma=10.0),
        ),
    )
    yield
    SingletonConfig.config = None


def _params(loader: DatasetLoader, T: int) -> DPTrainingParams:
    network = MLP.build(MLPConfig(hidden_sizes=(4,)), (1, N_FEATURES), (1, N_CLASSES), key=0)
    return DPTrainingParams(
        loader=loader,
        optimizer=optax.sgd(1.0),
        lr=1.0,
        network=network,
        num_training_steps=T,
        scan_segments=T,
        buffer_size=BATCH_SIZE * 2,
    )


def _leaves(model) -> list:
    return jax.tree.leaves(eqx.filter(model, eqx.is_array))


@pytest.mark.parametrize("T,moves", [(1, False), (2, True)])
def test_step_clips_with_the_incoming_threshold(singleton, loader, T, moves):
    """A zero initial clip freezes step 0; only step 1 sees the opened threshold."""
    params = _params(loader, T)
    schedule = _ZeroThenOpenSchedule(iteration_array=jnp.arange(T))

    trained, _ = train_with_stateful_noise(
        schedule, params, jr.PRNGKey(0), jr.PRNGKey(1), jr.PRNGKey(2)
    )
    initial = reinit_model(params.network, jr.PRNGKey(1))

    changed = any(not jnp.allclose(a, b) for a, b in zip(_leaves(trained), _leaves(initial)))
    assert changed is moves


def test_adaptive_clip_baseline_trains_end_to_end(singleton, loader):
    """The privatised adaptive-clip schedule runs through the real DP-SGD scan."""
    T = 4
    privacy_params = GDPPrivacyParameters(1.0, 1e-6, BATCH_SIZE / N_TRAIN, T)
    # r must shrink with the (tiny, synthetic) batch size or the count release would
    # eat the per-step budget and construction refuses — see the guard's own test.
    schedule = StatefulMedianGradientNoiseAndClipSchedule(1.0, 0.2, privacy_params, r=2.0)

    trained, stats = train_with_stateful_noise(
        schedule, _params(loader, T), jr.PRNGKey(0), jr.PRNGKey(1), jr.PRNGKey(2)
    )

    assert all(bool(jnp.isfinite(leaf).all()) for leaf in _leaves(trained))
    assert bool(jnp.isfinite(stats.losses).all())
