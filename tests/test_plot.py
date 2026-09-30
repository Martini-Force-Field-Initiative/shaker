"""Tests for figure saving/display control (shaker/plot.py, shaker/options.py)."""

import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from shaker import plot_bonded_distributions, plot_sasa_dir, set_options
from shaker.plot import _distance_xlim, _finish_figure, _term_hints


class TestFinishFigure:
    def test_svg_by_default_other_formats_by_extension(self, tmp_path):
        fig = plt.figure()
        _finish_figure(fig, tmp_path / "stem", True, show=False)
        _finish_figure(fig, tmp_path / "sub" / "fit.png", True, False, tag="A-B")

        saved = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.*"))
        assert saved == ["stem.svg", "sub/fit_A-B.png"]

    def test_show_false_closes_and_none_follows_set_options(self):
        fig = plt.figure()
        assert _finish_figure(fig, None, True, show=None) is True
        assert plt.fignum_exists(fig.number)

        set_options(show=False)
        assert _finish_figure(fig, None, True, show=None) is False
        assert not plt.fignum_exists(fig.number)


def test_plot_bonded_distributions_quiet(tmp_path):
    bins = np.linspace(0, 10, 50)
    hist = np.exp(-((bins - 5) ** 2))[None, :]
    bonded = {
        "distances": {"targets": [("A", "B")], "bins": bins, "hist": hist},
        "angles": {"targets": [], "bins": bins, "hist": np.empty((0, 50))},
        "dihedrals": {"targets": [], "bins": bins, "hist": np.empty((0, 50))},
    }

    fig = plot_bonded_distributions(
        bonded, bonded, outfile=tmp_path / "bonds", show=False
    )

    assert (tmp_path / "bonds.svg").exists()  # svg by default
    assert not plt.fignum_exists(fig.number)


def test_plot_sasa_dir_reports_difference_from_aa(tmp_path, capsys):
    for name, value in (("AA", 5.0), ("CG", 5.4)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "SASA.xvg").write_text(f"@ header\n0 {value}\n1 {value}\n")

    plot_sasa_dir(tmp_path, outfile=None, show=False)

    assert (
        "CG            5.40 ± 0.00   +0.40 nm²  (+8.0%)  ⚠" in capsys.readouterr().out
    )


def test_bonded_report_wraps_dihedrals_and_filters(capsys):
    x = np.arange(-180, 180, 2.0)

    def peak(mu):
        h = np.exp(-0.5 * (((x - mu + 180) % 360 - 180) / 15) ** 2)
        return h / h.sum() / 2

    def bonded(*mus):
        empty = {"targets": [], "bins": x, "hist": np.empty((0, x.size))}
        dihedrals = [["A", "B", "C", str(i)] for i in range(len(mus))]
        hists = np.array([peak(mu) for mu in mus])
        return {
            "distances": empty,
            "angles": empty,
            "dihedrals": {"targets": dihedrals, "bins": x, "hist": hists},
        }

    # Term 0: -179° vs 179° (2° apart across ±180); term 1: 60° shift.
    ref, cg = bonded(-179, 0), bonded(179, 60)
    plot_bonded_distributions(ref, cg, outfile=None, show=False)
    out = capsys.readouterr().out
    assert re.search(r"A-B-C-0 .* -2\.0  OC 0\.\d\d   W 2\.0  ✓", out)
    assert "2 terms: 1 ✓  0 ⚠  1 ✗, 1 mean-off" in out

    plot_bonded_distributions(ref, cg, outfile=None, show=False, only_flagged=True)
    out = capsys.readouterr().out
    assert "A-B-C-0" not in out and "A-B-C-1" in out


def test_distance_xlim_keeps_4A_window_around_peaks():
    x = np.arange(0.05, 15, 0.1)

    def g(mu, sd):
        return np.exp(-0.5 * ((x - mu) / sd) ** 2)

    assert _distance_xlim(x, [g(2.2, 0.1), g(2.3, 0.1)]) == (1.5, 5.5)
    lo, hi = _distance_xlim(x, [g(13.9, 1.0), g(12.7, 1.8)])
    assert hi - lo == pytest.approx(4.0) and lo < 12.7 and hi > 13.9


def test_set_options_verbose_silences_reports_unless_asked(tmp_path, capsys):
    (tmp_path / "AA").mkdir()
    (tmp_path / "AA" / "SASA.xvg").write_text("0 5.0\n1 5.1\n")

    set_options(verbose=False)
    plot_sasa_dir(tmp_path, outfile=None, show=False)
    assert capsys.readouterr().out == ""

    plot_sasa_dir(tmp_path, outfile=None, show=False, verbose=True)
    assert "SASA (nm²)" in capsys.readouterr().out


def test_term_hints():
    ang, dih = np.arange(1, 180, 2.0), np.arange(-179, 180, 2.0)

    def g(x, *peaks, sd=5, periodic=False):
        d = [((x - mu + 180) % 360 - 180) if periodic else x - mu for mu in peaks]
        return sum(np.exp(-0.5 * (di / sd) ** 2) for di in d)

    assert _term_hints("angles", ang, g(ang, 110)) == []
    assert _term_hints("angles", ang, g(ang, 60, 90)) == ["multimodal"]
    assert _term_hints("angles", ang, g(ang, 165)) == ["near-linear"]
    assert _term_hints("dihedrals", dih, g(dih, 0, sd=90, periodic=True)) == ["flat"]
    assert _term_hints("dihedrals", dih, g(dih, 178, periodic=True)) == ["planar"]
    assert _term_hints("dihedrals", dih, g(dih, -90, 90, sd=20, periodic=True)) == [
        "multimodal"
    ]
