"""Tests for the self-describing sweep-file format (util/sweep_file.py).

The sweep file is a tab-separated table whose header row names per-run
``run-starter.py`` flags (``run_id``, ``mem_per_gpu``, ...). See
docs/adr/0004-self-describing-sweep-file-flags.md.
"""

from util.sweep_file import read_sweep_file, row_to_run_args, write_sweep_file


def test_write_then_read_round_trips_header_file(tmp_path):
    path = tmp_path / "sweep.txt"
    rows = [
        {"run_id": "abc123", "mem_per_gpu": "32G"},
        {"run_id": "def456", "mem_per_gpu": "16G"},
    ]
    write_sweep_file(path, rows)
    assert read_sweep_file(path) == rows


def test_read_legacy_bare_id_list_without_header(tmp_path):
    # Files written by sweep.py / older create_experiments are bare run IDs,
    # one per line, with no header. Each line is a run_id with no extra flags.
    path = tmp_path / "legacy.txt"
    path.write_text("abc123\ndef456\n")
    assert read_sweep_file(path) == [{"run_id": "abc123"}, {"run_id": "def456"}]


def test_row_to_run_args_renders_every_column_as_a_flag():
    row = {"run_id": "abc123", "mem_per_gpu": "32G"}
    assert row_to_run_args(row) == ["--run_id=abc123", "--mem_per_gpu=32G"]


def test_round_trip_preserves_unknown_extra_column(tmp_path):
    # A future per-run flag is just another column; the format must carry it
    # through read/write untouched without the module knowing what it means.
    path = tmp_path / "sweep.txt"
    rows = [{"run_id": "abc123", "mem_per_gpu": "32G", "runtime.days": "2"}]
    write_sweep_file(path, rows)
    assert read_sweep_file(path) == rows


def test_read_tolerates_blank_and_trailing_lines(tmp_path):
    path = tmp_path / "sweep.txt"
    path.write_text("run_id\tmem_per_gpu\nabc123\t32G\n\n")
    assert read_sweep_file(path) == [{"run_id": "abc123", "mem_per_gpu": "32G"}]


def test_in_place_refine_of_mem_column_preserves_other_columns_and_order(tmp_path):
    # predict_memory.py reads the file, recomputes mem_per_gpu, and rewrites in
    # place. Other columns and row order must survive the rewrite.
    path = tmp_path / "sweep.txt"
    write_sweep_file(
        path,
        [
            {"run_id": "abc123", "mem_per_gpu": "32G", "runtime.days": "2"},
            {"run_id": "def456", "mem_per_gpu": "16G", "runtime.days": "2"},
        ],
    )
    rows = read_sweep_file(path)
    for row in rows:
        row["mem_per_gpu"] = "8G"
    write_sweep_file(path, rows)

    assert read_sweep_file(path) == [
        {"run_id": "abc123", "mem_per_gpu": "8G", "runtime.days": "2"},
        {"run_id": "def456", "mem_per_gpu": "8G", "runtime.days": "2"},
    ]
