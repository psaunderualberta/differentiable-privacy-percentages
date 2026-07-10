import json
from pathlib import Path

import pandas as pd
import pytest

import compile_results_fetch as crf
from compile_results_fetch import (
    DATASET_SHAPES,
    LocalArtifact,
    LocalRun,
    assert_shapes_consistent,
    build_run_manifest,
)


class FakeRun:
    """Duck-types the subset of wandb.Run that the fetch/dump code reads."""

    def __init__(self):
        self.id = "abc123"
        self.name = "run-abc"
        self.tags = ["T-sweep", "ladder:mlp-depth"]
        self.state = "finished"
        self.config = {"dataset": "mnist", "env": {"eps": 0.5, "num_training_steps": 100}}
        self.summary = {"test-accuracy": 0.9, "test-loss": 0.3}
        self.notes = "a note"
        self.group = "grp"
        self.job_type = "train"
        self.created_at = "2026-01-01T00:00:00"
        self.url = "https://wandb.ai/e/p/runs/abc123"
        self._history = [
            {"test-accuracy": 0.5, "test-loss": 1.0, "train-loss": 2.0},
            {"test-accuracy": 0.9, "test-loss": 0.3, "train-loss": 0.4},
        ]

    def scan_history(self, keys=None):
        if keys is None:
            return iter([dict(r) for r in self._history])
        return iter([{k: r[k] for k in keys} for r in self._history])


class TestBuildRunManifest:
    def test_captures_full_config_metadata_and_history(self):
        m = build_run_manifest(FakeRun())

        assert m["config"] == {
            "dataset": "mnist",
            "env": {"eps": 0.5, "num_training_steps": 100},
        }
        assert m["summary"] == {"test-accuracy": 0.9, "test-loss": 0.3}
        assert m["meta"]["id"] == "abc123"
        assert m["meta"]["name"] == "run-abc"
        assert m["meta"]["tags"] == ["T-sweep", "ladder:mlp-depth"]
        assert m["meta"]["state"] == "finished"
        assert m["meta"]["notes"] == "a note"
        assert m["meta"]["group"] == "grp"
        assert m["meta"]["job_type"] == "train"
        assert m["meta"]["created_at"] == "2026-01-01T00:00:00"
        assert m["meta"]["url"].endswith("abc123")

    def test_history_keeps_all_logged_keys(self):
        m = build_run_manifest(FakeRun())
        # Unlike _history(), the dump must keep every logged key (e.g. train-loss),
        # not just test-accuracy/test-loss.
        assert m["history"] == [
            {"test-accuracy": 0.5, "test-loss": 1.0, "train-loss": 2.0},
            {"test-accuracy": 0.9, "test-loss": 0.3, "train-loss": 0.4},
        ]


class TestLocalRunReplay:
    """A LocalRun rebuilt from a manifest must be read exactly like a wandb.Run."""

    def test_replays_scalar_attributes(self):
        orig = FakeRun()
        lr = LocalRun(build_run_manifest(orig), artifact_root=None)
        assert lr.id == orig.id
        assert lr.name == orig.name
        assert lr.config == orig.config
        assert lr.tags == orig.tags
        assert lr.summary == orig.summary

    def test_scan_history_matches_original_with_and_without_keys(self):
        orig = FakeRun()
        lr = LocalRun(build_run_manifest(orig), artifact_root=None)

        want_all = list(orig.scan_history(keys=None))
        assert list(lr.scan_history(keys=None)) == want_all

        keys = ["test-accuracy", "test-loss"]
        want_subset = list(orig.scan_history(keys=keys))
        assert list(lr.scan_history(keys=keys)) == want_subset


class TestLocalArtifact:
    """LocalArtifact serves downloaded artifact files like a wandb.Artifact."""

    def _write_table(self, directory, name, columns, data):
        (directory / f"{name}.table.json").write_text(
            json.dumps({"columns": columns, "data": data})
        )

    def test_get_returns_table_with_data_and_columns(self, tmp_path):
        self._write_table(tmp_path, "sigmas", ["step", "0", "1"], [[0, 0.5, 0.6]])
        art = LocalArtifact(name="run-abc-sigmas:v3", directory=tmp_path)

        assert art.name == "run-abc-sigmas:v3"
        table = art.get("sigmas")
        assert table.columns == ["step", "0", "1"]
        assert table.data == [[0, 0.5, 0.6]]

    def test_download_returns_dir_holding_the_files(self, tmp_path):
        self._write_table(tmp_path, "clips", ["step", "0"], [[0, 1.0]])
        art = LocalArtifact(name="run-abc-clips:v0", directory=tmp_path)

        local = art.download()
        assert (Path(local) / "clips.table.json").exists()


class TestDatasetShapesConsistency:
    """DATASET_SHAPES is a fetch-side mirror of the dataloader's cached layout.

    It silently drifted once (eyepacs cached 256x256/5-class but the mirror said
    224x224/2-class); these tests pin the ground truth and guard the mirror.
    """

    def test_eyepacs_matches_cached_layout(self):
        # Ground truth: _eyepacs_download_and_cache writes (N, 3, 256, 256) uint8
        # with 5-class one-hot labels.
        assert DATASET_SHAPES["eyepacs"] == ((3, 256, 256), 5)

    def test_assert_shapes_consistent_passes_on_current_table(self):
        assert_shapes_consistent()  # should not raise

    def test_assert_shapes_consistent_catches_drift(self, monkeypatch):
        monkeypatch.setitem(DATASET_SHAPES, "eyepacs", ((3, 224, 224), 2))
        with pytest.raises(AssertionError, match="eyepacs"):
            assert_shapes_consistent()


# ---------------------------------------------------------------------------
# Integration fakes: a duck-typed live run + api that both the fetch path and
# the archive writer can consume, so a round-trip can be asserted.
# ---------------------------------------------------------------------------


class FakeTable:
    def __init__(self, columns, data):
        self.columns = columns
        self.data = data


class FakeLoggedArtifact:
    """A logged sigmas/clips table artifact (has both .get and .download)."""

    def __init__(self, name, table_name, columns, data):
        self.name = name
        self._table_name = table_name
        self._table = FakeTable(columns, data)

    def get(self, table_name):
        assert table_name == self._table_name
        return self._table

    def download(self, root=None):
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / f"{self._table_name}.table.json").write_text(
            json.dumps({"columns": self._table.columns, "data": self._table.data})
        )
        return str(root)


class FakeBaselineArtifact:
    def __init__(self, df):
        self.name = "baseline-run"
        self._df = df

    def download(self, root=None):
        Path(root).mkdir(parents=True, exist_ok=True)
        self._df.to_pickle(str(Path(root) / "baseline.pkl"))
        return str(root)


class FakeRunWithArtifacts:
    def __init__(self, run_id="abc123"):
        self.id = run_id
        self.name = f"run-{run_id}"
        self.tags = ["T-sweep"]
        self.state = "finished"
        self.config = {
            "dataset": "mnist",
            "prng_seed": 0,
            "env": {
                "eps": 0.5,
                "num_training_steps": 2,
                "optimizer": "sgd",
                "network": {"_type": "MLPConfig", "hidden_sizes": [8]},
            },
        }
        self.summary = {"test-accuracy": 0.9}
        self.notes = None
        self.group = None
        self.job_type = None
        self.created_at = "2026-01-01"
        self.url = f"https://wandb.ai/e/p/runs/{run_id}"
        self._history = [
            {"test-accuracy": 0.5, "test-loss": 1.0},
            {"test-accuracy": 0.9, "test-loss": 0.3},
        ]

    def scan_history(self, keys=None):
        if keys is None:
            return iter([dict(r) for r in self._history])
        return iter([{k: r[k] for k in keys} for r in self._history])

    def logged_artifacts(self):
        return [
            FakeLoggedArtifact(
                "sigmas:v1", "sigmas", ["step", "0", "1"], [[0, 0.4, 0.5], [1, 0.6, 0.7]]
            ),
            FakeLoggedArtifact(
                "clips:v1", "clips", ["step", "0", "1"], [[0, 1.0, 1.1], [1, 1.2, 1.3]]
            ),
        ]


class FakeApi:
    def __init__(self, run, baseline_df):
        self._run = run
        self._baseline = FakeBaselineArtifact(baseline_df)

    def run(self, path):
        return self._run

    def artifact(self, path):
        assert "baseline-" in path
        return self._baseline


class MultiRunFakeApi:
    """Serves several runs (and their baselines) resolved by id from the path.

    Unlike FakeApi (single fixed run), this maps ``.../<run_id>`` and
    ``.../baseline-<run_id>:latest`` to per-run objects, so the archive writer
    can be driven over multiple runs.
    """

    def __init__(self, runs, baseline_df, failing_ids=()):
        self._runs = {r.id: r for r in runs}
        self._baseline_df = baseline_df
        self._failing_ids = set(failing_ids)

    def run(self, path):
        run_id = path.split("/")[-1]
        if run_id in self._failing_ids:
            raise RuntimeError(f"boom fetching {run_id}")
        return self._runs[run_id]

    def artifact(self, path):
        name = path.split("/")[-1]
        assert name.startswith("baseline-")
        return FakeBaselineArtifact(self._baseline_df)


def _baseline_df():
    return pd.DataFrame(
        [
            {"type": "Constant σ/clip", "accuracy": 0.6, "loss": 0.8},
            {"type": "Constant σ/clip", "accuracy": 0.8, "loss": 0.6},
        ]
    )


class TestFetchOneRunInjectedApi:
    def test_fetch_uses_injected_api(self, tmp_path, monkeypatch):
        monkeypatch.setattr(crf, "ARTIFACT_ROOT", tmp_path / "artifacts")
        api = FakeApi(FakeRunWithArtifacts(), _baseline_df())

        scalars, schedules, histories = crf._fetch_one_run("e", "p", "abc123", api=api)

        # Learned scalar row + one baseline row.
        by_sched = {s["schedule"]: s for s in scalars}
        assert by_sched["Learned Schedule"]["mean_acc"] == 0.9
        assert by_sched["Constant σ/clip"]["mean_acc"] == pytest.approx(0.7)
        # Final-step schedule row length == T (2), values from the last table row.
        assert len(schedules) == 2
        assert schedules[-1]["sigma"] == 0.7
        assert schedules[-1]["clip"] == 1.3
        assert len(histories) == 2


class TestFullArchiveRoundTrip:
    """Deletion is lossless iff a fetch from the archive == a fetch from W&B."""

    def test_archive_replays_fetch_identically(self, tmp_path, monkeypatch):
        monkeypatch.setattr(crf, "ARTIFACT_ROOT", tmp_path / "artifacts")
        api = FakeApi(FakeRunWithArtifacts(), _baseline_df())

        live = crf._fetch_one_run("e", "p", "abc123", api=api)

        zip_path = tmp_path / "full_config.zip"
        crf.write_full_archive(["abc123"], api, "e", "p", zip_path)
        assert zip_path.exists()

        local_api = crf.open_full_archive(zip_path)
        replay = crf._fetch_one_run("e", "p", "abc123", api=local_api)

        assert replay == live

    def test_archives_multiple_runs_into_one_zip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(crf, "ARTIFACT_ROOT", tmp_path / "artifacts")
        run_ids = ["r1", "r2", "r3"]
        api = MultiRunFakeApi([FakeRunWithArtifacts(rid) for rid in run_ids], _baseline_df())

        live = {rid: crf._fetch_one_run("e", "p", rid, api=api) for rid in run_ids}

        zip_path = tmp_path / "full_config.zip"
        crf.write_full_archive(run_ids, api, "e", "p", zip_path)

        local_api = crf.open_full_archive(zip_path)
        for rid in run_ids:
            assert crf._fetch_one_run("e", "p", rid, api=local_api) == live[rid]

    def test_one_failing_run_does_not_sink_the_others(self, tmp_path, monkeypatch):
        monkeypatch.setattr(crf, "ARTIFACT_ROOT", tmp_path / "artifacts")
        run_ids = ["r1", "bad", "r3"]
        api = MultiRunFakeApi(
            [FakeRunWithArtifacts(rid) for rid in ("r1", "r3")],
            _baseline_df(),
            failing_ids=["bad"],
        )

        zip_path = tmp_path / "full_config.zip"
        missing = crf.write_full_archive(run_ids, api, "e", "p", zip_path)

        # The bad run is reported, not raised.
        assert [m["run_id"] for m in missing] == ["bad"]
        assert "boom" in missing[0]["reason"]

        # The good runs are still fully archived and replayable.
        local_api = crf.open_full_archive(zip_path)
        for rid in ("r1", "r3"):
            scalars, _, _ = crf._fetch_one_run("e", "p", rid, api=local_api)
            assert any(s["schedule"] == "Learned Schedule" for s in scalars)

    def test_workers_build_their_own_api_when_none_injected(self, tmp_path, monkeypatch):
        # Constraint: workers must NOT share a wandb.Api across threads. When no
        # api is injected, each worker fetches via _get_api() rather than a shared
        # client passed down from main.
        monkeypatch.setattr(crf, "ARTIFACT_ROOT", tmp_path / "artifacts")
        run_ids = ["r1", "r2"]
        worker_api = MultiRunFakeApi([FakeRunWithArtifacts(rid) for rid in run_ids], _baseline_df())
        calls = {"n": 0}

        def fake_get_api():
            calls["n"] += 1
            return worker_api

        monkeypatch.setattr(crf, "_get_api", fake_get_api)

        zip_path = tmp_path / "full_config.zip"
        missing = crf.write_full_archive(run_ids, None, "e", "p", zip_path)

        assert missing == []
        assert calls["n"] >= 1  # _get_api was used to build the worker api
        local_api = crf.open_full_archive(zip_path)
        for rid in run_ids:
            scalars, _, _ = crf._fetch_one_run("e", "p", rid, api=local_api)
            assert any(s["schedule"] == "Learned Schedule" for s in scalars)


class _RunsOnlyApi:
    def __init__(self, run_meta):
        self._run_meta = run_meta

    def runs(self, path, filters=None):
        import types

        return [types.SimpleNamespace(id=i, name=n) for i, n in self._run_meta]


class TestMainFullConfigWiring:
    def _patch_common(self, monkeypatch, run_meta):
        monkeypatch.setattr(crf.wandb, "Api", lambda: _RunsOnlyApi(run_meta))
        monkeypatch.setattr(crf, "_fetch_one_run", lambda e, p, rid: ([], [], []))

    def test_writes_archive_when_flag_set(self, tmp_path, monkeypatch):
        self._patch_common(monkeypatch, [("r1", "n1"), ("r2", "n2")])
        calls = {}

        def fake_archive(run_ids, api, entity, project, zip_path, num_workers):
            calls.update(
                run_ids=list(run_ids),
                api=api,
                zip_path=str(zip_path),
                num_workers=num_workers,
            )
            return []

        monkeypatch.setattr(crf, "write_full_archive", fake_archive)
        conf = crf.FetchConfig(
            project="p",
            entity="e",
            out_dir=str(tmp_path),
            full_config=True,
            num_workers=5,
        )
        crf.main(conf)

        assert calls["run_ids"] == ["r1", "r2"]
        assert calls["zip_path"].endswith("full_config.zip")
        assert Path(calls["zip_path"]).parent == tmp_path
        # main threads its worker count through, and does NOT share its own Api
        # with the worker threads (constraint #1).
        assert calls["num_workers"] == 5
        assert calls["api"] is None

    def test_no_archive_when_flag_unset(self, tmp_path, monkeypatch):
        self._patch_common(monkeypatch, [("r1", "n1")])
        called = {"n": 0}
        monkeypatch.setattr(
            crf,
            "write_full_archive",
            lambda *a, **k: called.update(n=called["n"] + 1),
        )
        conf = crf.FetchConfig(project="p", entity="e", out_dir=str(tmp_path))
        crf.main(conf)

        assert called["n"] == 0
