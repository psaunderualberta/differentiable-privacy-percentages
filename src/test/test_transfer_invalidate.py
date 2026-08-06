"""ADR 0021 invalidates the sgd-m0.0 half of the transfer batch.

Those cells' filenames are *unchanged* by the arm fix — a curve cell's name is keyed on
its W&B run id, which already implies the arm — so the launcher's skip filter would read
every one of them as finished and never re-run them. They have to be removed from the
cache explicitly, and only they: the sgd-m0.9 cells ran against a matched target and
their numbers survive the re-run untouched.
"""

import pandas as pd

from transfer_invalidate import invalid_cells


def _cell(path, producer, source_id, arm):
    pd.DataFrame(
        [
            {
                "producer": producer,
                "source_id": source_id,
                "source_arm": arm,
                "accuracy": 0.5,
            }
        ]
    ).to_parquet(path, index=False)
    return path


class TestInvalidCells:
    def test_only_the_mismatched_arms_cells_are_selected(self, tmp_path):
        curve = tmp_path / "curve"
        curve.mkdir()
        matched = _cell(curve / "m09.parquet", "curve", "run09", "sgd-m0.9")
        mismatched = _cell(curve / "m00.parquet", "curve", "run00", "sgd-m0.0")

        assert invalid_cells(tmp_path, keep_arm="sgd-m0.9") == [mismatched]
        assert matched.exists()

    def test_cells_predating_the_arm_are_invalid_too(self, tmp_path):
        # The three stragglers from a parquet older than ADR 0011 carry no arm at all,
        # so nothing says which momentum they were learned under — unusable either way.
        curve = tmp_path / "curve"
        curve.mkdir()
        armless = _cell(curve / "old.parquet", "curve", "runOld", "")

        assert invalid_cells(tmp_path, keep_arm="sgd-m0.9") == [armless]

    def test_cells_written_before_the_arm_column_existed_are_invalid(self, tmp_path):
        # Older than the schema itself, not merely older than ADR 0011: the parquet has
        # no source_arm column at all. Reading it must not raise — a crash here strands
        # the whole invalidation, and the cell is unsalvageable for the same reason an
        # empty arm is.
        curve = tmp_path / "curve"
        curve.mkdir()
        legacy = curve / "prehistoric.parquet"
        pd.DataFrame([{"producer": "curve", "source_id": "r", "accuracy": 0.5}]).to_parquet(
            legacy, index=False
        )

        assert invalid_cells(tmp_path, keep_arm="sgd-m0.9") == [legacy]

    def test_reference_cells_are_never_touched(self, tmp_path):
        # A reference's arm is its *target* momentum (ADR 0021), and the existing 18 all
        # ran at m=0.9 — correctly, since the bug pinned every target there. They are
        # re-run because the arm fix renames them, not because they are wrong, so this
        # deletion must leave them alone.
        reference = tmp_path / "reference"
        reference.mkdir()
        _cell(reference / "Constant__chexpert.parquet", "reference", "Constant", "")

        assert invalid_cells(tmp_path, keep_arm="sgd-m0.9") == []
