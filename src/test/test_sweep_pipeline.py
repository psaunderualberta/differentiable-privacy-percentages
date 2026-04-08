"""Integration tests: sweep config → (mock) W&B → run reconstruction pipeline.

Covers the complete path that sweep.py and singleton_conf.py implement:

  SweepConfig.to_wandb_sweep()
    → wandb.sweep()          (mocked — returns a fake sweep ID)
    → wandb.agent(starter)   (mocked — calls starter() with sampled configs)
    → starter() writes run IDs to cc/sweeps/<sweep_id>.txt   (real I/O, tmp_path)
    → wandb.Api().run()      (mocked — returns the sampled run config)
    → get_wandb_run_conf()
    → _reconstruct_from_dict()
    → assert correct schedule type / fields restored

W&B API calls are replaced with lightweight mocks so nothing leaves the
machine and all file I/O is scoped to pytest's tmp_path.
"""

import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conf.config import EnvConfig, ScheduleOptimizerConfig, SweepConfig, WandbConfig
from conf.singleton_conf import _reconstruct_from_dict, get_wandb_run_conf
from policy.schedules.config import (
    AlternatingSigmaAndClipScheduleConfig,
    SigmaAndClipScheduleConfig,
    WarmupAlternatingSigmaAndClipScheduleConfig,
)

# Fire @register decorators so registries are populated (same pattern as test_config.py).
for _mod in [
    "policy.base_schedules.constant",
    "policy.base_schedules.exponential",
    "policy.base_schedules.clipped",
    "policy.schedules.alternating",
    "policy.schedules.sigma_and_clip",
    "policy.schedules.policy_and_clip",
    "policy.schedules.dynamic_dpsgd",
    "policy.schedules.warmup_alternating",
    "policy.stateful_schedules.median_gradient",
    "networks.mlp.MLP",
    "networks.cnn.CNN",
]:
    importlib.import_module(_mod)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_from_sweep_spec(sweep_spec: dict, type_overrides: dict | None = None) -> dict:
    """Simulate W&B sampling one run config from a sweep parameter spec.

    Traverses the nested ``{"parameters": {...}}`` tree and produces a dict
    matching the structure of ``wandb.run.config``:

    - ``{"value": v}``      → ``v``
    - ``{"values": [v, …]}`` → first element (or ``type_overrides[dot_path]``)
    - ``{"min": …, …}``     → ``min`` (for numeric distributions)
    - nested ``{"parameters": …}`` → recurse

    Args:
        sweep_spec:    The dict returned by ``SweepConfig.to_wandb_sweep()``.
        type_overrides: Dot-path → chosen value for ``"values"`` distributions,
            e.g. ``{"schedule_optimizer.schedule._type": "SigmaAndClipScheduleConfig"}``.
    """
    type_overrides = type_overrides or {}

    def _sample(node: dict, path: str) -> dict:
        result = {}
        for k, v in node.get("parameters", {}).items():
            child_path = f"{path}.{k}" if path else k
            if "parameters" in v:
                result[k] = _sample(v, child_path)
            elif child_path in type_overrides:
                result[k] = type_overrides[child_path]
            elif "value" in v:
                result[k] = v["value"]
            elif "values" in v:
                result[k] = v["values"][0]
            elif v.get("distribution") == "int_uniform":
                result[k] = int(v.get("min", 0))
            elif "min" in v:
                result[k] = float(v["min"])
            else:
                result[k] = 0.0
        return result

    return _sample(sweep_spec, "")


def _simulate_agent(
    sweep_id: str,
    sweep_spec: dict,
    sweeps_dir: Path,
    run_configs: list[dict],
) -> list[str]:
    """Replicate what sweep.py's ``starter()`` does for each agent run.

    Writes one run ID per config to ``sweeps_dir/<sweep_id>.txt`` and
    returns the list of run IDs — exactly as sweep.py does via wandb.agent.
    """
    sweep_file = sweeps_dir / f"{sweep_id}.txt"
    run_ids = []
    for i, _ in enumerate(run_configs):
        run_id = f"mock-run-{sweep_id}-{i:03d}"
        run_ids.append(run_id)
        with open(sweep_file, "a") as f:
            f.write(run_id + "\n")
    return run_ids


def _mock_wandb_conf(run_id: str) -> WandbConfig:
    return WandbConfig(
        entity="test-entity",
        project="test-project",
        mode="disabled",
        restart_run_id=run_id,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSweepPipeline:
    """End-to-end pipeline: sweep config → agent runs → run ID file → reconstruction."""

    # --- File I/O ---

    def test_run_ids_written_to_sweep_file(self, tmp_path):
        """Agent writes one run ID per run to <sweeps_dir>/<sweep_id>.txt."""
        sweep_id = "test-sweep-abc"
        run_configs = [{}, {}, {}]
        run_ids = _simulate_agent(sweep_id, {}, tmp_path, run_configs)

        sweep_file = tmp_path / f"{sweep_id}.txt"
        assert sweep_file.exists()
        written = sweep_file.read_text().strip().splitlines()
        assert written == run_ids
        assert len(written) == 3

    def test_multiple_sweeps_use_separate_files(self, tmp_path):
        """Each sweep ID gets its own run-ID file."""
        for sweep_id in ("sweep-aaa", "sweep-bbb"):
            _simulate_agent(sweep_id, {}, tmp_path, [{}])

        assert (tmp_path / "sweep-aaa.txt").exists()
        assert (tmp_path / "sweep-bbb.txt").exists()

    # --- Round-trip: sweep spec → sampled run config → reconstruction ---

    def test_default_schedule_optimizer_round_trip(self, tmp_path):
        """Default ScheduleOptimizerConfig (3 schedule types) reconstructs to first type."""
        sweep = SweepConfig(
            env=EnvConfig(), schedule_optimizer=ScheduleOptimizerConfig(), num_outer_steps=10
        )
        sweep_spec = sweep.to_wandb_sweep()
        run_conf = _sample_from_sweep_spec(sweep_spec)

        sweep_id = "sweep-default"
        _ = _simulate_agent(sweep_id, sweep_spec, tmp_path, [run_conf])
        run_id = (tmp_path / f"{sweep_id}.txt").read_text().strip()

        mock_api_run = MagicMock()
        mock_api_run.config = run_conf
        with patch("conf.singleton_conf.wandb") as mock_wandb:
            mock_wandb.Api.return_value.run.return_value = mock_api_run
            fetched = get_wandb_run_conf(_mock_wandb_conf(run_id), run_id)

        reconstructed = _reconstruct_from_dict(sweep, fetched)
        # First type in default sweep_schedule_conf_types is AlternatingSigmaAndClip.
        assert isinstance(
            reconstructed.schedule_optimizer.schedule, AlternatingSigmaAndClipScheduleConfig
        )

    @pytest.mark.parametrize(
        "chosen_type, expected_cls",
        [
            (
                "AlternatingSigmaAndClipScheduleConfig",
                AlternatingSigmaAndClipScheduleConfig,
            ),
            ("SigmaAndClipScheduleConfig", SigmaAndClipScheduleConfig),
            (
                "WarmupAlternatingSigmaAndClipScheduleConfig",
                WarmupAlternatingSigmaAndClipScheduleConfig,
            ),
        ],
    )
    def test_each_schedule_type_reconstructs_correctly(self, tmp_path, chosen_type, expected_cls):
        """Each schedule type sampled by W&B reconstructs to the right class."""
        sweep = SweepConfig(
            env=EnvConfig(), schedule_optimizer=ScheduleOptimizerConfig(), num_outer_steps=10
        )
        sweep_spec = sweep.to_wandb_sweep()
        run_conf = _sample_from_sweep_spec(
            sweep_spec,
            type_overrides={"schedule_optimizer.schedule._type": chosen_type},
        )

        sweep_id = f"sweep-{chosen_type[:8].lower()}"
        _ = _simulate_agent(sweep_id, sweep_spec, tmp_path, [run_conf])
        run_id = (tmp_path / f"{sweep_id}.txt").read_text().strip()

        mock_api_run = MagicMock()
        mock_api_run.config = run_conf
        with patch("conf.singleton_conf.wandb") as mock_wandb:
            mock_wandb.Api.return_value.run.return_value = mock_api_run
            fetched = get_wandb_run_conf(_mock_wandb_conf(run_id), run_id)

        reconstructed = _reconstruct_from_dict(sweep, fetched)
        assert isinstance(reconstructed.schedule_optimizer.schedule, expected_cls), (
            f"Expected {expected_cls.__name__}, got {type(reconstructed.schedule_optimizer.schedule).__name__}"
        )

    def test_multiple_runs_in_one_sweep(self, tmp_path):
        """A sweep with multiple agent runs writes distinct IDs and each reconstructs."""
        sweep = SweepConfig(
            env=EnvConfig(), schedule_optimizer=ScheduleOptimizerConfig(), num_outer_steps=10
        )
        sweep_spec = sweep.to_wandb_sweep()

        type_sequence = [
            "AlternatingSigmaAndClipScheduleConfig",
            "SigmaAndClipScheduleConfig",
            "WarmupAlternatingSigmaAndClipScheduleConfig",
        ]
        expected_classes = [
            AlternatingSigmaAndClipScheduleConfig,
            SigmaAndClipScheduleConfig,
            WarmupAlternatingSigmaAndClipScheduleConfig,
        ]
        run_confs = [
            _sample_from_sweep_spec(
                sweep_spec, type_overrides={"schedule_optimizer.schedule._type": t}
            )
            for t in type_sequence
        ]

        sweep_id = "sweep-multi"
        run_ids = _simulate_agent(sweep_id, sweep_spec, tmp_path, run_confs)

        # Verify the run-ID file contains all three IDs in order.
        written = (tmp_path / f"{sweep_id}.txt").read_text().strip().splitlines()
        assert written == run_ids

        # Verify each run reconstructs to the right schedule class.
        for run_id, run_conf, expected_cls in zip(run_ids, run_confs, expected_classes):
            mock_api_run = MagicMock()
            mock_api_run.config = run_conf
            with patch("conf.singleton_conf.wandb") as mock_wandb:
                mock_wandb.Api.return_value.run.return_value = mock_api_run
                fetched = get_wandb_run_conf(_mock_wandb_conf(run_id), run_id)
            reconstructed = _reconstruct_from_dict(sweep, fetched)
            assert isinstance(reconstructed.schedule_optimizer.schedule, expected_cls), (
                f"Run {run_id}: expected {expected_cls.__name__}, "
                f"got {type(reconstructed.schedule_optimizer.schedule).__name__}"
            )

    def test_api_called_with_correct_run_path(self, tmp_path):
        """get_wandb_run_conf calls wandb.Api().run() with entity/project/run_id."""
        sweep = SweepConfig(
            env=EnvConfig(), schedule_optimizer=ScheduleOptimizerConfig(), num_outer_steps=10
        )
        sweep_spec = sweep.to_wandb_sweep()
        run_conf = _sample_from_sweep_spec(sweep_spec)

        sweep_id = "sweep-api-check"
        run_ids = _simulate_agent(sweep_id, sweep_spec, tmp_path, [run_conf])
        run_id = run_ids[0]

        mock_api_run = MagicMock()
        mock_api_run.config = run_conf
        with patch("conf.singleton_conf.wandb") as mock_wandb:
            mock_wandb.Api.return_value.run.return_value = mock_api_run
            wandb_conf = _mock_wandb_conf(run_id)
            get_wandb_run_conf(wandb_conf, run_id)
            mock_wandb.Api.return_value.run.assert_called_once_with(
                f"{wandb_conf.entity}/{wandb_conf.project}/{run_id}"
            )

    def test_env_params_survive_round_trip(self, tmp_path):
        """Non-schedule parameters (eps, batch_size, dataset) reconstruct correctly."""
        sweep = SweepConfig(
            env=EnvConfig(eps=2.0, batch_size=512),
            schedule_optimizer=ScheduleOptimizerConfig(),
            num_outer_steps=50,
            dataset="fashion-mnist",
        )
        sweep_spec = sweep.to_wandb_sweep()
        run_conf = _sample_from_sweep_spec(sweep_spec)

        mock_api_run = MagicMock()
        mock_api_run.config = run_conf
        with patch("conf.singleton_conf.wandb") as mock_wandb:
            mock_wandb.Api.return_value.run.return_value = mock_api_run
            fetched = get_wandb_run_conf(_mock_wandb_conf("run-env-check"), "run-env-check")

        reconstructed = _reconstruct_from_dict(sweep, fetched)
        assert reconstructed.env.eps == pytest.approx(2.0)
        assert reconstructed.env.batch_size == 512
        assert reconstructed.num_outer_steps == 50
        assert reconstructed.dataset == "fashion-mnist"
