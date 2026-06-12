import dataclasses
from dataclasses import dataclass
from pprint import pprint
from typing import Annotated, Any, Literal, Union

import tyro

from conf.config_util import (
    DistributionConfig,
    dist_config_helper,
    merge_wandb_sweep_union,
    to_wandb_conf,
    to_wandb_sweep_params,
)
from conf.optimizer_config import OptimizerConfig, SGDConfig
from networks.auto.config import AutoNetworkConfig
from networks.cnn.config import CNNConfig
from networks.mlp.config import MLPConfig
from policy.schedules.config import (
    AlternatingSigmaAndClipScheduleConfig,
    DecoupledSigmaAndClipScheduleConfig,
    DynamicDPSGDScheduleConfig,
    ParallelSigmaAndClipScheduleConfig,
    PolicyAndClipScheduleConfig,
    SigmaAndClipScheduleConfig,
    WarmupAlternatingSigmaAndClipScheduleConfig,
    WarmupParallelSigmaAndClipScheduleConfig,
    WarmupSigmaAndClipScheduleConfig,
)
from policy.stateful_schedules.config import StatefulMedianGradientNoiseAndClipConfig

# ---------------------------------------------------------------------------
# Union type aliases — tyro treats a Union of dataclasses as subcommands,
# so only one variant is ever instantiated.  ConsolidateSubcommandArgs in the
# tyro.cli call folds the subcommand selection into the flat --flag style.
# ---------------------------------------------------------------------------

ScheduleConfig = Union[
    Annotated[
        AlternatingSigmaAndClipScheduleConfig,
        tyro.conf.subcommand("alternating-sigma-and-clip"),
    ],
    Annotated[SigmaAndClipScheduleConfig, tyro.conf.subcommand("sigma-and-clip")],
    Annotated[
        DecoupledSigmaAndClipScheduleConfig,
        tyro.conf.subcommand("decoupled-sigma-and-clip"),
    ],
    Annotated[PolicyAndClipScheduleConfig, tyro.conf.subcommand("policy-and-clip")],
    Annotated[DynamicDPSGDScheduleConfig, tyro.conf.subcommand("dynamic-dp-sgd")],
    Annotated[
        StatefulMedianGradientNoiseAndClipConfig,
        tyro.conf.subcommand("median-gradient"),
    ],
    Annotated[
        WarmupAlternatingSigmaAndClipScheduleConfig,
        tyro.conf.subcommand("warmup-alternating"),
    ],
    Annotated[
        WarmupSigmaAndClipScheduleConfig,
        tyro.conf.subcommand("warmup-sigma-and-clip"),
    ],
    Annotated[
        ParallelSigmaAndClipScheduleConfig,
        tyro.conf.subcommand("parallel-sigma-and-clip"),
    ],
    Annotated[
        WarmupParallelSigmaAndClipScheduleConfig,
        tyro.conf.subcommand("warmup-parallel-sigma-and-clip"),
    ],
]

NetworkConfig = Union[
    Annotated[AutoNetworkConfig, tyro.conf.subcommand("auto")],
    Annotated[MLPConfig, tyro.conf.subcommand("mlp")],
    Annotated[CNNConfig, tyro.conf.subcommand("cnn")],
]

# ---
# Config for the schedule optimizer
# ---


@dataclass
class ESConfig:
    """Evolutionary-Strategies gradient estimator settings.

    When ``enabled`` is True, the outer loop replaces analytic
    ``value_and_grad`` with an antithetic OpenAI-ES estimator over the leaves
    selected by each schedule's ``es_filter()``.
    """

    enabled: bool = False
    population_size: DistributionConfig = dist_config_helper(value=32, distribution="constant")
    """Number of perturbation samples per outer step. Must be even (antithetic
    pairs) and divisible by the GPU count; asserted at startup."""
    perturbation_sigma: DistributionConfig = dist_config_helper(value=0.1, distribution="constant")
    """Initial std-dev of Gaussian perturbations on ES-opted-in leaves.
    Treated as the *initial* σ when ``eta_sigma > 0`` (natural-gradient σ
    update enabled, Wierstra et al. 2014)."""  # noqa: RUF001
    eta_sigma: DistributionConfig = dist_config_helper(value=0.1, distribution="constant")
    """Learning rate for the natural-gradient log-σ update (sNES, Wierstra et
    al. 2014). 0 disables the σ update (σ stays at ``perturbation_sigma``)."""  # noqa: RUF001
    adaptation_enabled: bool = False
    """Enable Wierstra et al. (2014) §6.2 adaptation sampling for ``η_σ``."""  # noqa: RUF001
    adaptation_c: float = 1.5
    """Hypothetical-σ multiplier used by adaptation sampling."""  # noqa: RUF001
    adaptation_rho: float = 0.5
    """U-statistic threshold above which adaptation sampling grows ``η_σ``."""  # noqa: RUF001
    adaptation_step: float = 0.1
    """Multiplicative step size for the ``η_σ`` update."""  # noqa: RUF001
    eta_sigma_max: float = 1.0
    """Upper clamp on ``η_σ`` under adaptation sampling."""  # noqa: RUF001


@dataclass
class ScheduleOptimizerConfig:
    schedule: ScheduleConfig = dataclasses.field(
        default_factory=ParallelSigmaAndClipScheduleConfig,
    )
    batch_size: int = 1
    lr: DistributionConfig = dist_config_helper(value=0.05, distribution="constant")
    momentum: DistributionConfig = dist_config_helper(
        values=(0.0, 0.1, 0.3),
        distribution="values",
    )
    max_sigma: float = 10.0
    # Global-norm clip on the outer-loop schedule gradient, applied before the
    # SGD update as a safety net against the rare divergent inner-DP-SGD run
    # whose backward pass overflows to Inf/NaN. Paired with optax.zero_nans()
    # (which neutralises a corrupt step rather than letting it crash the run).
    # 1.0 is the conventional default; check the logged ``grad-global-norm``
    # metric to confirm it is not throttling normal steps and raise if needed.
    max_grad_norm: float = 1.0
    es: ESConfig = dataclasses.field(default_factory=ESConfig)
    # When non-empty, sweep over these schedule type names rather than fixing
    # _type to the current schedule's class.  Set programmatically before
    # calling to_wandb_sweep(); not exposed as a CLI flag.
    sweep_schedule_conf_types: Annotated[tuple[str, ...], tyro.conf.Fixed] = (
        AlternatingSigmaAndClipScheduleConfig.__name__,
        WarmupAlternatingSigmaAndClipScheduleConfig.__name__,
        WarmupSigmaAndClipScheduleConfig.__name__,
        SigmaAndClipScheduleConfig.__name__,
    )

    def to_wandb_sweep(self) -> dict[str, Any]:
        result = to_wandb_sweep_params(self)
        if self.sweep_schedule_conf_types:
            # Lazy import avoids circular: singleton_conf imports config at top-level.
            from conf.singleton_conf import _get_config_classes

            config_classes = _get_config_classes()
            instances = []
            for name in self.sweep_schedule_conf_types:
                if name not in config_classes:
                    raise ValueError(
                        f"Unknown schedule type '{name}'. Known: {sorted(config_classes)}",
                    )
                instances.append(config_classes[name]())
            result["parameters"]["schedule"] = merge_wandb_sweep_union(instances)
        return result

    def to_wandb_conf(self) -> dict[str, Any]:
        return to_wandb_conf(self)  # ---


# Config for the private environment
# ---


@dataclass
class EnvConfig:
    "Configuration for the private training environment"

    network: NetworkConfig = dataclasses.field(default_factory=AutoNetworkConfig)

    optimizer: OptimizerConfig = dataclasses.field(default_factory=SGDConfig)
    loss_type: Literal["mse", "cce"] = "cce"

    # Privacy Parameters
    eps: float = 0.5
    delta: float = 1e-6
    batch_size: int = 250
    num_training_steps: int = 100
    scan_segments: int = -1
    """Number of segments for the scan-of-scans. Must divide num_training_steps.
    K=1 is equivalent to the current single scan with no behaviour change.
    K>1 reduces peak gradient tape memory by a factor of K at negligible runtime cost."""

    microbatch_size: int = -1
    """Per-microbatch size for DP-SGD per-sample gradient accumulation.
    -1 (or >= batch_size) means no microbatching: the whole Poisson minibatch's
    per-sample gradients are materialised at once. A smaller value (must divide
    batch_size) caps the live per-sample-gradient working set at
    microbatch_size x params instead of batch_size x params, trading compute for
    memory. Numerically identical to the no-microbatching path (per-example clip
    multipliers depend only on each example's own gradient)."""

    @property
    def scan_segments_derived(self) -> int:
        if self.scan_segments < 0:
            return self.num_training_steps
        return self.scan_segments

    @property
    def microbatch_size_derived(self) -> int:
        if self.microbatch_size <= 0 or self.microbatch_size >= self.batch_size:
            return self.batch_size
        if self.batch_size % self.microbatch_size != 0:
            raise ValueError(
                f"microbatch_size ({self.microbatch_size}) must divide "
                f"batch_size ({self.batch_size}).",
            )
        return self.microbatch_size

    def to_wandb_sweep(self) -> dict[str, Any]:
        return to_wandb_sweep_params(self)

    def to_wandb_conf(self) -> dict[str, Any]:
        return to_wandb_conf(self)


@dataclass
class SweepConfig:
    env: EnvConfig
    schedule_optimizer: ScheduleOptimizerConfig
    method: str = "grid"
    metric_name: str = "val-accuracy"
    metric_goal: str = "maximize"
    plotting_interval: int = 1
    name: str | None = None
    description: str | None = None
    with_baselines: bool = False
    # Only used when with_baselines=True. Evaluate and log baselines every this
    # many outer steps (0 = only log at the end of training, i.e. old behaviour).
    baseline_log_interval: int = 0
    dataset: Literal["mnist", "cifar-10", "fashion-mnist", "eyepacs"] = "mnist"
    val_test_split: float = 0.8  # Percentage of validation set to use for validation (i.e. evaluating policy), rest for test
    dataset_poly_d: int | None = None
    num_outer_steps: int = 100
    # Seeds the global numpy RNG once at startup so that non-constant
    # DistributionConfig defaults (e.g. prng_seed, momentum) are sampled
    # reproducibly for a given command. Change this to vary which value is
    # drawn from those distributions. Does not affect W&B sweeps, where the
    # agent injects concrete constants that bypass sampling.
    master_seed: int = 0
    prng_seed: DistributionConfig = dist_config_helper(
        values=(447831761, 159020393, 435372193),
        distribution="values",
    )
    train_on_single_network: bool = False
    shutdown_buffer_secs: int = 180
    """Seconds before SLURM wall-time expiry to trigger graceful checkpoint-and-resubmit."""

    @property
    def plotting_steps(self) -> int:
        if self.plotting_interval >= self.num_outer_steps:
            return 1
        return self.num_outer_steps // self.plotting_interval

    def to_wandb_sweep(self) -> dict[str, Any]:
        config = {
            "method": self.method,
            "metric": {
                "name": self.metric_name,
                "goal": self.metric_goal,
            },
            **to_wandb_sweep_params(self),
        }
        if self.description is not None:
            config["description"] = self.description
        if self.name is not None:
            config["name"] = self.name
        return config

    def to_wandb_conf(self) -> dict[str, Any]:
        return to_wandb_conf(self)


@dataclass
class WandbConfig:
    """Wandb Configuration"""

    project: str | None = None
    entity: str | None = None
    mode: Literal["disabled", "online", "offline"] = "disabled"
    restart_run_id: str | None = None
    # Directory for W&B local run storage.  Mainly relevant in offline mode:
    # the run dir must survive until it is synced, so it should live on
    # persistent storage (not the per-job SLURM_TMPDIR, which is wiped at
    # job end).  None falls back to ./wandb in the working directory.
    wandb_dir: str | None = None
    # Offline mode only: how often (seconds) a background thread runs
    # `wandb sync` on the in-progress run so the dashboard updates near-live
    # without coupling the training loop to the network.  0 disables the
    # background daemon (data is still synced once at the end of the run).
    wandb_sync_interval_secs: int = 0

    # --- Checkpointing ---
    # Run ID whose checkpoint artifact to restore.  Set to the same value as
    # restart_run_id to resume a crashed job; set to a different run's ID to
    # branch from another run's state.
    checkpoint_run_id: str | None = None
    # Specific outer-loop step to restore.  None means the latest checkpoint.
    # If set to a value other than the latest step, a new W&B run is created
    # in ``{project}-branched`` with a note referencing the original run.
    checkpoint_step: int | None = None
    # Save a checkpoint (locally + W&B artifact) every this many outer steps.
    checkpoint_every: int = 25


@dataclass
class Config:
    wandb_conf: WandbConfig
    sweep: SweepConfig
    data_dir: str | None = None  # Used in log_wandb.py, ignored otherwise


if __name__ == "__main__":
    args = tyro.cli(
        Config,
        config=(tyro.conf.ConsolidateSubcommandArgs,),
    )
    pprint(args)
