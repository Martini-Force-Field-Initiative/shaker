"""Tests for shaker/system_builders.py — iteration checkpointing."""

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from shaker.system_builders import (
    _checkpoint,
    _run_mdrun,
    _StageLine,
    list_iterations,
    run_status,
)

DATA = Path(__file__).parent / "data"
FINISHED_LOG = DATA / "prod_finished.log"


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
        for name in (
            "iter_2_20260101_000200",
            "iter_0_20260101_000000",
            "iter_1_20260101_000100",
        ):
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


class TestRunMdrun:
    """_run_mdrun with a fake mdrun (a python one-liner)."""

    def _fake(self, code):
        return [sys.executable, "-c", code]

    def test_echoes_output_and_returns(self, capsys):
        _run_mdrun(self._fake("print('step 100, will finish soon')"))
        assert "step 100" in capsys.readouterr().out

    def test_nonzero_exit_raises_with_output_tail(self, tmp_path, capsys):
        code = "import sys; print('bad topology'); sys.exit(1)"
        with (
            open(tmp_path / "gmx_run.log", "w") as log,
            pytest.raises(subprocess.CalledProcessError, match="bad topology"),
        ):
            _run_mdrun(self._fake(code), log=log)

        # Quiet: output went to the log, not the cell.
        assert "bad topology" in (tmp_path / "gmx_run.log").read_text()
        assert capsys.readouterr().out == ""

    def test_kills_process_stuck_after_fatal_error(self):
        code = "import time; print('Fatal error:', flush=True); time.sleep(60)"
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="fatal error"):
            _run_mdrun(self._fake(code), fatal_grace=0.5, poll=0.1)
        assert time.monotonic() - start < 15

    def test_kills_silent_process(self):
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="no output"):
            _run_mdrun(
                self._fake("import time; time.sleep(60)"), hang_timeout=1, poll=0.1
            )
        assert time.monotonic() - start < 15


class TestRunStatus:
    def test_finished_production(self, tmp_path, capsys):
        shutil.copy(DATA / "min_finished.log", tmp_path / "m.log")
        shutil.copy(FINISHED_LOG, tmp_path / "p.log")
        _make_cleaned_traj(tmp_path)

        status = run_status(tmp_path)

        assert status["min"]["steps_done"] == 3491
        assert status["min"]["fmax"] == pytest.approx(24.808376)

        prod = status["prod"]
        assert prod["finished"] and not prod["stopped_early"]
        assert prod["steps_done"] == prod["nsteps"] == 10000000
        assert status["rel"] is None and status["done"]
        out = capsys.readouterr().out
        assert "  min    3491 steps   Fmax 24.8 (target 10)   ✓\n" in out
        assert "  prod   200.0/200.0 ns   31733 ns/day   0:09:05   ✓\n" in out

    def test_flags_truncated_run_with_lincs_warnings(self, tmp_path, capsys):
        log = FINISHED_LOG.read_text()
        log = log.replace("Statistics over 10000001", "Statistics over 3000001")
        log = log.replace(
            "Writing checkpoint",
            "Step 0, time 0 (ps)  LINCS WARNING\nWriting checkpoint",
        )
        (tmp_path / "p.log").write_text(log)

        run_status(tmp_path)

        out = capsys.readouterr().out
        assert "60.0/200.0 ns" in out
        assert "⚠ stopped early · 1 LINCS warnings" in out


def test_stage_line_is_append_only(tmp_path, capsys):
    shutil.copy(FINISHED_LOG, tmp_path / "p.log")  # nsteps = 10000000
    line = _StageLine("", tmp_path / "p.log")

    for step in (100, 600000, 2500000, 5000000, 7600000, 9999900):
        line(f"\rstep {step}, will finish soon")
    line.finish("RESULT")

    out = capsys.readouterr().out
    assert re.match(r"ETA \d\d:\d\d \(.+\) · 25 · 50 · 75 +RESULT\n$", out)
    assert out.index("RESULT") == _StageLine.WIDTH
