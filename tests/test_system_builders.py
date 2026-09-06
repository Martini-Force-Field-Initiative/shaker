"""Tests for shaker/system_builders.py — iteration checkpointing."""

import json

import pytest

from shaker.system_builders import list_iterations, _checkpoint


def _make_cleaned_traj(dir_path, pdb_text="ATOM\n", xtc_text="xtc"):
    (dir_path / "pbc.pdb").write_text(pdb_text)
    (dir_path / "pbc.xtc").write_text(xtc_text)


class TestListIterations:
    def test_missing_dir_returns_empty(self, tmp_path):
        assert list_iterations(tmp_path / "does_not_exist") == []

    def test_no_iterations_returns_empty(self, tmp_path):
        (tmp_path / "some_other_dir").mkdir()
        assert list_iterations(tmp_path) == []

    def test_finds_and_sorts_by_index(self, tmp_path):
        for name in ("iter_2_20260101_000200", "iter_0_20260101_000000",
                     "iter_1_20260101_000100"):
            (tmp_path / name).mkdir()

        result = list_iterations(tmp_path)
        assert [p.name for p in result] == [
            "iter_0_20260101_000000",
            "iter_1_20260101_000100",
            "iter_2_20260101_000200",
        ]

    def test_ignores_non_matching_names(self, tmp_path):
        (tmp_path / "iter_0_20260101_000000").mkdir()
        (tmp_path / "iterations").mkdir()
        (tmp_path / "iter_abc_20260101_000000").mkdir()
        (tmp_path / "iter_1_20260101_000000").write_text("not a dir")

        result = list_iterations(tmp_path)
        assert [p.name for p in result] == ["iter_0_20260101_000000"]


class TestCheckpoint:
    def test_returns_none_without_cleaned_trajectory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _checkpoint() is None
        assert list_iterations(tmp_path) == []

    def test_creates_first_iteration(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _make_cleaned_traj(tmp_path)
        (tmp_path / "initial_CG.itp").write_text("; itp contents")

        iter_dir = _checkpoint()

        assert iter_dir is not None
        assert iter_dir.name.startswith("iter_0_")
        assert (iter_dir / "pbc.pdb").exists()
        assert (iter_dir / "pbc.xtc").exists()
        assert (iter_dir / "initial_CG.itp").read_text() == "; itp contents"

    def test_increments_index_across_runs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _make_cleaned_traj(tmp_path)

        first = _checkpoint()
        second = _checkpoint()

        assert first.name.startswith("iter_0_")
        assert second.name.startswith("iter_1_")
        assert [p.name for p in list_iterations(tmp_path)] == [first.name, second.name]

    def test_saves_mapping_json_when_provided(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _make_cleaned_traj(tmp_path)
        mapping = {"MOL": {"B1": {"type": "SC3", "charge": 0, "atoms": ["C1"]}}}

        iter_dir = _checkpoint(mapping=mapping)

        saved = json.loads((iter_dir / "mapping.json").read_text())
        assert saved == mapping

    def test_no_mapping_json_when_not_provided(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _make_cleaned_traj(tmp_path)

        iter_dir = _checkpoint()

        assert not (iter_dir / "mapping.json").exists()

    def test_multiple_itp_files_all_copied(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _make_cleaned_traj(tmp_path)
        (tmp_path / "a.itp").write_text("a")
        (tmp_path / "b.itp").write_text("b")

        iter_dir = _checkpoint()

        assert (iter_dir / "a.itp").read_text() == "a"
        assert (iter_dir / "b.itp").read_text() == "b"

    def test_keep_last_prunes_old_iterations(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _make_cleaned_traj(tmp_path)

        for _ in range(3):
            _checkpoint(keep_last=2)

        remaining = list_iterations(tmp_path)
        assert [p.name.split("_")[1] for p in remaining] == ["1", "2"]

    def test_keep_last_minus_one_keeps_everything(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _make_cleaned_traj(tmp_path)

        for _ in range(3):
            _checkpoint(keep_last=-1)

        assert len(list_iterations(tmp_path)) == 3

    def test_keep_last_none_disables_checkpointing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _make_cleaned_traj(tmp_path)

        result = _checkpoint(keep_last=None)

        assert result is None
        assert list_iterations(tmp_path) == []

    @pytest.mark.parametrize("bad_value", [0, -2, -5])
    def test_keep_last_invalid_values_raise(self, tmp_path, monkeypatch, bad_value):
        monkeypatch.chdir(tmp_path)
        _make_cleaned_traj(tmp_path)

        with pytest.raises(ValueError, match="keep_last must be"):
            _checkpoint(keep_last=bad_value)
