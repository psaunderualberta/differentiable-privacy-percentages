"""The slug-level artefacts an SR synthesis shares between its per-target jobs.

``sr-run-starter.py`` submits **one SLURM job per target**, and ``sr_identity`` excludes
``targets`` from the synthesis identity — so every target's job of one arm resolves to the
same ``out_dir`` and races the others writing ``features_full.parquet``, ``manifest.json``
and ``category_map.json`` there. These tests pin the three properties that make that safe:
writes are atomic, the multi-gigabyte parquet is written once, and the manifest stops
under-reporting which targets the directory holds.
"""

import json

import pandas as pd
import pytest

from symbolic_regression import (
    PySRConfig,
    _atomic_write,
    _write_manifest,
    write_features_full,
)


def _frame(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": [f"r{i}" for i in range(n)],
            "dataset": ["mnist"] * n,
            "arch_label": ["mlp"] * n,
            "optimizer": ["sgd-m0.9"] * n,
            "eps": [1.0] * n,
            "T": [20] * n,
            "sigma": [1.0 + i for i in range(n)],
        }
    )


def _leftovers(d):
    return [p.name for p in d.iterdir() if ".tmp" in p.name]


# ---------------------------------------------------------------------------
# _atomic_write
# ---------------------------------------------------------------------------


def test_atomic_write_materialises_the_target(tmp_path):
    path = tmp_path / "thing.json"

    _atomic_write(path, lambda p: p.write_text('{"a": 1}'))

    assert json.loads(path.read_text()) == {"a": 1}
    assert _leftovers(tmp_path) == []


def test_atomic_write_leaves_no_partial_file_when_the_writer_raises(tmp_path):
    """A crashed write must not leave a truncated file at the real path — that is the
    torn read a concurrent job would otherwise pick up."""
    path = tmp_path / "thing.json"

    def explode(p):
        p.write_text("half a fi")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _atomic_write(path, explode)

    assert not path.exists()
    assert _leftovers(tmp_path) == []


def test_atomic_write_preserves_the_previous_version_on_failure(tmp_path):
    path = tmp_path / "thing.json"
    path.write_text("original")

    with pytest.raises(RuntimeError):
        _atomic_write(path, lambda p: (_ for _ in ()).throw(RuntimeError("boom")))

    assert path.read_text() == "original"


def test_atomic_write_temp_lives_beside_the_target(tmp_path):
    """os.replace is only atomic within one filesystem, so the temp must be a sibling
    rather than in $TMPDIR."""
    seen = []
    path = tmp_path / "thing.json"

    def record(p):
        seen.append(p.parent)
        p.write_text("x")

    _atomic_write(path, record)

    assert seen == [tmp_path]


def test_atomic_write_temp_names_do_not_repeat(tmp_path):
    """The concurrent writers are array tasks on different nodes, where pids collide —
    two writers sharing a temp path would interleave and rename the result into place."""
    seen = []
    path = tmp_path / "thing.json"

    for _ in range(8):
        _atomic_write(path, lambda p: (seen.append(p.name), p.write_text("x")))

    assert len(set(seen)) == len(seen)


# ---------------------------------------------------------------------------
# write_features_full — the write-once guard
# ---------------------------------------------------------------------------


def test_write_features_full_writes_the_frame(tmp_path):
    assert write_features_full(tmp_path, _frame()) is True

    back = pd.read_parquet(tmp_path / "features_full.parquet")
    assert len(back) == 4
    assert list(back.columns) == list(_frame().columns)


def test_write_features_full_skips_a_complete_existing_file(tmp_path):
    """The second per-target job of an arm must not write a second full-size copy —
    two ~1.08M-row parquets landing in one directory is what filled the disk."""
    write_features_full(tmp_path, _frame(4))
    before = (tmp_path / "features_full.parquet").stat().st_mtime_ns

    assert write_features_full(tmp_path, _frame(9)) is False

    assert (tmp_path / "features_full.parquet").stat().st_mtime_ns == before
    assert len(pd.read_parquet(tmp_path / "features_full.parquet")) == 4


def test_write_features_full_rewrites_an_unreadable_file(tmp_path):
    """A parquet left truncated by a pre-fix crash must not be preserved by the guard."""
    write_features_full(tmp_path, _frame(4))
    path = tmp_path / "features_full.parquet"
    path.write_bytes(path.read_bytes()[:64])  # footer gone

    assert write_features_full(tmp_path, _frame(4)) is True

    assert len(pd.read_parquet(path)) == 4


def test_write_features_full_leaves_no_temp_behind(tmp_path):
    write_features_full(tmp_path, _frame())

    assert _leftovers(tmp_path) == []


# ---------------------------------------------------------------------------
# _write_manifest — targets accumulate across the directory's jobs
# ---------------------------------------------------------------------------


def _manifest(out_dir, targets):
    conf = PySRConfig(cache_dir="/nowhere/psaunder__FirSweep", targets=targets)
    _write_manifest(out_dir, conf, {"r0", "r1"}, _frame())
    return json.loads((out_dir / "manifest.json").read_text())


def test_manifest_records_this_jobs_target(tmp_path):
    assert _manifest(tmp_path, ("sigma",))["targets"] == ["sigma"]


def test_manifest_unions_targets_with_an_earlier_job(tmp_path):
    """Both jobs write this file seconds apart; the survivor must describe the directory,
    not just the last writer."""
    _manifest(tmp_path, ("sigma",))

    assert _manifest(tmp_path, ("clip",))["targets"] == ["clip", "sigma"]


def test_manifest_records_only_this_jobs_target_under_config(tmp_path):
    """``config`` is the invoking job's own config — ``transfer_equation.synthesis_arm``
    reads ``config.optimizers`` from it, so it must stay a faithful config dump."""
    manifest = _manifest(tmp_path, ("clip",))

    assert manifest["config"]["targets"] == ["clip"]
    assert manifest["config"]["optimizers"] == []


def test_manifest_survives_a_corrupt_existing_file(tmp_path):
    (tmp_path / "manifest.json").write_text("{ truncated")

    assert _manifest(tmp_path, ("sigma",))["targets"] == ["sigma"]


def test_manifest_leaves_no_temp_behind(tmp_path):
    _manifest(tmp_path, ("sigma",))

    assert _leftovers(tmp_path) == []


def test_manifest_still_carries_the_identity_derived_fields(tmp_path):
    manifest = _manifest(tmp_path, ("sigma",))

    assert manifest["optimizers"] == ["sgd-m0.9"]
    assert manifest["datasets"] == ["mnist"]
    assert manifest["n_runs"] == 2
    assert manifest["n_rows_full"] == 4
