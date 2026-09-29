"""Tests for figure saving/display control (shaker/plot.py, shaker/_options.py)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from shaker import plot_bonded_distributions, set_options
from shaker._options import _OPTIONS
from shaker.plot import _finish_figure


@pytest.fixture(autouse=True)
def _restore_options():
    saved = dict(_OPTIONS)
    yield
    _OPTIONS.update(saved)


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


@pytest.mark.filterwarnings("ignore:No targets found")
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
