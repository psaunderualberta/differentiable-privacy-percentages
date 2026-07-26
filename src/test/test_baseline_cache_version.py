"""Baseline caches are versioned so pre-privatisation results cannot be restored.

``restore_from_cache`` keys purely on ``run_id`` — no config hash, no content check —
so privatising the adaptive-clip baseline (ADR 0013) would otherwise silently restore
stale numbers and skip the sweep entirely. The cache must fail *closed*.
"""

import pathlib
from unittest.mock import patch

from util.baselines import _baseline_artifact_name, _baseline_path

RUN_ID = "abc123"

# Names in use before the count release was privatised.
LEGACY_ARTIFACT_NAME = f"baseline-{RUN_ID}"
LEGACY_FILENAME = "baseline_data.pkl"


def test_pre_privatisation_caches_are_unreachable(tmp_path):
    with patch("util.baselines._ckpt_dir", return_value=pathlib.Path(tmp_path)):
        path = _baseline_path(RUN_ID)

    assert _baseline_artifact_name(RUN_ID) != LEGACY_ARTIFACT_NAME
    assert path.name != LEGACY_FILENAME
    # Still keyed on the run, so distinct runs stay distinct.
    assert RUN_ID in _baseline_artifact_name(RUN_ID)
