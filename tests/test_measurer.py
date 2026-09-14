"""
Tests for shaker/measurer.py — bonded distribution measurement.

Uses in-memory MDAnalysis universes built from known atom positions so that
expected distances and angles are exact.  Positions are passed in Å (MDAnalysis
internal units).
"""

import numpy as np
import pytest
from conftest import make_universe

from shaker.measurer import measure_bonded_terms

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def peak_bin(results, term_type, index=0):
    """Return the bin centre with the highest histogram count."""
    bins = results[term_type]["bins"]
    hist = results[term_type]["hist"][index]
    return bins[np.argmax(hist)]


# ---------------------------------------------------------------------------
# Distance measurements
# ---------------------------------------------------------------------------


class TestDistanceMeasurement:
    def test_known_distance_peak(self):
        """Two atoms 5 Å apart → distribution centred at 5 Å."""
        u = make_universe([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], ["B1", "B2"])
        res = measure_bonded_terms(u, "MOL", [("B1", "B2")], [], [])
        assert peak_bin(res, "distances") == pytest.approx(5.0, abs=0.3)

    def test_different_separation(self):
        """Three Å separation should give peak near 3 Å."""
        u = make_universe([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], ["B1", "B2"])
        res = measure_bonded_terms(u, "MOL", [("B1", "B2")], [], [])
        assert peak_bin(res, "distances") == pytest.approx(3.0, abs=0.3)

    def test_multiple_distance_targets(self):
        """Two distance targets → hist array has shape (2, n_bins)."""
        u = make_universe(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [7.0, 0.0, 0.0]], ["B1", "B2", "B3"]
        )
        res = measure_bonded_terms(u, "MOL", [("B1", "B2"), ("B2", "B3")], [], [])
        assert res["distances"]["hist"].shape[0] == 2

    def test_histogram_is_normalized(self):
        """Probability density × bin width should integrate to ~1."""
        u = make_universe([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], ["B1", "B2"])
        res = measure_bonded_terms(u, "MOL", [("B1", "B2")], [], [])
        bins = res["distances"]["bins"]
        hist = res["distances"]["hist"][0]
        bin_width = bins[1] - bins[0]
        assert np.sum(hist) * bin_width == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# Angle measurements
# ---------------------------------------------------------------------------


class TestAngleMeasurement:
    def test_right_angle(self):
        """B1-B2-B3 at 90° should be measured as such."""
        positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
        u = make_universe(positions, ["B1", "B2", "B3"])
        res = measure_bonded_terms(u, "MOL", [], [("B1", "B2", "B3")], [])
        assert peak_bin(res, "angles") == pytest.approx(90.0, abs=2.0)

    def test_near_linear_arrangement(self):
        """Nearly collinear atoms (175°) should give a peak near 175°."""
        angle_rad = np.deg2rad(175.0)
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0 + np.cos(np.pi - angle_rad), np.sin(np.pi - angle_rad), 0.0],
        ]
        u = make_universe(positions, ["B1", "B2", "B3"])
        res = measure_bonded_terms(u, "MOL", [], [("B1", "B2", "B3")], [])
        assert peak_bin(res, "angles") == pytest.approx(175.0, abs=3.0)

    def test_multiple_angle_targets(self):
        positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        u = make_universe(positions, ["B1", "B2", "B3", "B4"])
        res = measure_bonded_terms(
            u, "MOL", [], [("B1", "B2", "B3"), ("B2", "B3", "B4")], []
        )
        assert res["angles"]["hist"].shape[0] == 2


# ---------------------------------------------------------------------------
# Dihedral measurements
#
# Geometry used (all positions in Å):
#   B1=[0,1,0]  B2=[0,0,0]  B3=[1,0,0]  B4 varies
#
#   B4=[1, 1, 0]  → cis,      dihedral =   0°
#   B4=[1, 0, 1]  → gauche+,  dihedral =  90°
#   B4=[1, 0,-1]  → gauche−,  dihedral = −90°
#
# Derivation (IUPAC convention, signed angle between planes B1-B2-B3 / B2-B3-B4):
#   n1 = (B2-B1) × (B3-B2) = [0,0,1]  for all cases
#   n2 depends on B4; the sign follows the right-hand rule along B2→B3.
# ---------------------------------------------------------------------------


class TestDihedralMeasurement:
    def test_cis_dihedral(self):
        """Planar cis arrangement should give dihedral peak near 0°."""
        positions = [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
        u = make_universe(positions, ["B1", "B2", "B3", "B4"])
        res = measure_bonded_terms(u, "MOL", [], [], [("B1", "B2", "B3", "B4")])
        assert abs(peak_bin(res, "dihedrals")) == pytest.approx(0.0, abs=2.0)

    def test_positive_90_dihedral(self):
        """Gauche+ arrangement should give dihedral peak near +90°."""
        positions = [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]
        u = make_universe(positions, ["B1", "B2", "B3", "B4"])
        res = measure_bonded_terms(u, "MOL", [], [], [("B1", "B2", "B3", "B4")])
        assert peak_bin(res, "dihedrals") == pytest.approx(90.0, abs=2.0)

    def test_negative_90_dihedral(self):
        """Gauche− arrangement should give dihedral peak near −90°."""
        positions = [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, -1.0],
        ]
        u = make_universe(positions, ["B1", "B2", "B3", "B4"])
        res = measure_bonded_terms(u, "MOL", [], [], [("B1", "B2", "B3", "B4")])
        assert peak_bin(res, "dihedrals") == pytest.approx(-90.0, abs=2.0)

    def test_multiple_dihedral_targets(self):
        """Two dihedral targets → hist array has shape (2, n_bins)."""
        positions = [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],  # B4: cis w.r.t. B1-B2-B3
            [1.0, 0.0, 1.0],  # B5: gauche+ w.r.t. B2-B3-B4
        ]
        u = make_universe(positions, ["B1", "B2", "B3", "B4", "B5"])
        res = measure_bonded_terms(
            u, "MOL", [], [], [("B1", "B2", "B3", "B4"), ("B2", "B3", "B4", "B5")]
        )
        assert res["dihedrals"]["hist"].shape[0] == 2

    def test_dihedral_bins_span_full_range(self):
        """Dihedral bins should cover −180 to +180°."""
        positions = [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
        u = make_universe(positions, ["B1", "B2", "B3", "B4"])
        res = measure_bonded_terms(u, "MOL", [], [], [("B1", "B2", "B3", "B4")])
        bins = res["dihedrals"]["bins"]
        assert bins.min() < -170
        assert bins.max() > 170


# ---------------------------------------------------------------------------
# Result dictionary structure
# ---------------------------------------------------------------------------


class TestResultStructure:
    def test_all_keys_present(self):
        u = make_universe([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], ["B1", "B2"])
        res = measure_bonded_terms(u, "MOL", [("B1", "B2")], [], [])
        for section in ("distances", "angles", "dihedrals"):
            assert section in res
            for key in ("bins", "hist", "targets"):
                assert key in res[section]

    def test_targets_stored_in_result(self):
        u = make_universe([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], ["B1", "B2"])
        dist_tgts = [("B1", "B2")]
        res = measure_bonded_terms(u, "MOL", dist_tgts, [], [])
        assert list(res["distances"]["targets"][0]) == ["B1", "B2"]

    def test_empty_targets_give_empty_hist(self):
        u = make_universe([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], ["B1", "B2"])
        res = measure_bonded_terms(u, "MOL", [], [], [])
        assert res["distances"]["hist"].shape[0] == 0
        assert res["angles"]["hist"].shape[0] == 0
