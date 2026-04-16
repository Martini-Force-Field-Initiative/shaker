"""
Tests for shaker/estimator.py — the physics core.

Strategy: feed synthetic Gaussian distributions with known σ and verify that
the equipartition-theorem estimators recover the expected force constants.
  k = kB T / σ²
"""

import numpy as np
import pytest
from scipy.stats import norm

from shaker.estimator import (
    _estimate_bond_params_from_hist,
    _estimate_angle_params_from_hist,
    _estimate_improper_params_from_hist,
    _estimate_bonded_from_dict,
    _KB,
)

_DIHED_BINS = np.arange(-180, 180, 2, dtype=float)  # standard dihedral bin edges → centres

T = 300
KT = _KB * T  # ≈ 2.494 kJ/mol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gaussian_hist(bins, loc, scale):
    """Normalised Gaussian probability density at given bin centres."""
    return norm.pdf(bins, loc=loc, scale=scale)


def _minimal_bonded_dict(dist_bins=None, dist_hist=None, dist_tgts=None,
                          ang_bins=None,  ang_hist=None,  ang_tgts=None,
                          dihed_bins=None, dihed_hist=None, dihed_tgts=None):
    """Build the dict that _estimate_bonded_from_dict expects."""
    if dist_bins is None:
        dist_bins, dist_hist, dist_tgts = np.array([]), np.array([[]]), []
    if ang_bins is None:
        ang_bins, ang_hist, ang_tgts = np.array([]), np.array([[]]), []
    if dihed_bins is None:
        dihed_bins, dihed_hist, dihed_tgts = np.array([]), np.array([[]]), []
    return {
        "distances": {"bins": dist_bins,  "hist": np.atleast_2d(dist_hist),  "targets": dist_tgts},
        "angles":    {"bins": ang_bins,   "hist": np.atleast_2d(ang_hist),   "targets": ang_tgts},
        "dihedrals": {"bins": dihed_bins, "hist": np.atleast_2d(dihed_hist), "targets": dihed_tgts},
    }


def _dihedral_hist(peak_deg=0.0, sigma_deg=30.0):
    """Gaussian-ish dihedral histogram peaked at peak_deg."""
    bins = _DIHED_BINS
    hist = np.exp(-0.5 * ((bins - peak_deg) / sigma_deg) ** 2) + 0.01
    return bins, hist


# ---------------------------------------------------------------------------
# Bond parameter estimation
# ---------------------------------------------------------------------------

class TestBondEstimation:
    def test_r0_recovered_angstrom(self):
        """Equilibrium length from a Gaussian centred at 5 Å should be 0.5 nm."""
        bins = np.linspace(2, 8, 500)
        hist = gaussian_hist(bins, loc=5.0, scale=0.3)
        r0, _ = _estimate_bond_params_from_hist(bins, hist, T=T, units="A")
        assert r0 == pytest.approx(0.5, rel=1e-3)

    def test_k_recovered_angstrom(self):
        """Force constant should match kT / σ_nm²."""
        sigma_A = 0.5
        bins = np.linspace(2, 8, 500)
        hist = gaussian_hist(bins, loc=5.0, scale=sigma_A)
        _, k = _estimate_bond_params_from_hist(bins, hist, T=T, units="A")
        sigma_nm = sigma_A / 10.0
        assert k == pytest.approx(KT / sigma_nm**2, rel=1e-3)

    def test_nm_units_no_rescaling(self):
        """With units='nm', values should not be rescaled."""
        sigma_nm = 0.05
        bins = np.linspace(0.2, 0.8, 500)
        hist = gaussian_hist(bins, loc=0.5, scale=sigma_nm)
        r0, k = _estimate_bond_params_from_hist(bins, hist, T=T, units="nm")
        assert r0 == pytest.approx(0.5, rel=1e-3)
        assert k == pytest.approx(KT / sigma_nm**2, rel=1e-3)

    def test_narrower_distribution_gives_larger_k(self):
        bins = np.linspace(2, 8, 500)
        _, k_wide   = _estimate_bond_params_from_hist(bins, gaussian_hist(bins, 5.0, 1.0), T=T, units="A")
        _, k_narrow = _estimate_bond_params_from_hist(bins, gaussian_hist(bins, 5.0, 0.1), T=T, units="A")
        assert k_narrow > k_wide

    def test_invalid_units_raises(self):
        bins = np.array([4.0, 5.0, 6.0])
        hist = np.array([0.1, 0.8, 0.1])
        with pytest.raises(ValueError):
            _estimate_bond_params_from_hist(bins, hist, units="angstrom")


# ---------------------------------------------------------------------------
# Angle parameter estimation
# ---------------------------------------------------------------------------

class TestAngleEstimation:
    def test_theta0_recovered(self):
        bins = np.linspace(60, 180, 500)
        hist = gaussian_hist(bins, loc=120.0, scale=8.0)
        t0, _ = _estimate_angle_params_from_hist(bins, hist, T=T, units="deg")
        assert t0 == pytest.approx(120.0, abs=0.2)

    def test_k_recovered(self):
        sigma_deg = 10.0
        bins = np.linspace(60, 180, 500)
        hist = gaussian_hist(bins, loc=120.0, scale=sigma_deg)
        _, k = _estimate_angle_params_from_hist(bins, hist, T=T, units="deg")
        sigma_rad = np.deg2rad(sigma_deg)
        assert k == pytest.approx(KT / sigma_rad**2, rel=1e-3)

    def test_result_in_degrees(self):
        """theta0 is returned in degrees, not radians."""
        bins = np.linspace(60, 180, 500)
        hist = gaussian_hist(bins, loc=120.0, scale=8.0)
        t0, _ = _estimate_angle_params_from_hist(bins, hist, T=T, units="deg")
        assert 60 < t0 < 180

    def test_invalid_units_raises(self):
        with pytest.raises(ValueError):
            _estimate_angle_params_from_hist(
                np.array([90., 120., 150.]), np.array([0.1, 0.8, 0.1]), units="degrees")


# ---------------------------------------------------------------------------
# Improper dihedral parameter estimation
# ---------------------------------------------------------------------------

class TestImproperEstimation:
    def test_standard_peak(self):
        """Peak away from ±180 — should match standard Gaussian stats."""
        sigma_deg = 15.0
        bins = np.linspace(-180, 180, 500)
        hist = gaussian_hist(bins, loc=0.0, scale=sigma_deg)
        t0, k = _estimate_improper_params_from_hist(bins, hist, T=T, units="deg")
        sigma_rad = np.deg2rad(sigma_deg)
        assert t0 == pytest.approx(0.0, abs=1.0)
        assert k == pytest.approx(KT / sigma_rad**2, rel=0.05)

    def test_periodic_boundary_correction(self):
        """
        A distribution straddling ±180 would naively have inflated variance.
        The corrected estimator should return a k close to kT/σ_rad².
        """
        sigma_deg = 5.0
        bins = np.linspace(-180, 180, 720)
        # Symmetric peak split across the ±180 boundary
        hist = (gaussian_hist(bins, loc=180.0, scale=sigma_deg) +
                gaussian_hist(bins, loc=-180.0, scale=sigma_deg))
        _, k = _estimate_improper_params_from_hist(bins, hist, T=T, units="deg")
        sigma_rad = np.deg2rad(sigma_deg)
        k_expected = KT / sigma_rad**2
        # Allow generous tolerance given discretisation, but it should be in the right ballpark
        assert k == pytest.approx(k_expected, rel=0.3)


# ---------------------------------------------------------------------------
# Full output formatting via _estimate_bonded_from_dict
# ---------------------------------------------------------------------------

class TestEstimateBondedFromDict:
    def _bond_dict(self, r0_A=5.0, sigma_A=0.5):
        bins = np.linspace(2, 8, 500)
        hist = gaussian_hist(bins, loc=r0_A, scale=sigma_A)
        return _minimal_bonded_dict(
            dist_bins=bins, dist_hist=hist, dist_tgts=[["B1", "B2"]])

    def _angle_dict(self, theta0=120.0, sigma_deg=10.0):
        bins = np.linspace(60, 180, 500)
        hist = gaussian_hist(bins, loc=theta0, scale=sigma_deg)
        return _minimal_bonded_dict(
            ang_bins=bins, ang_hist=hist, ang_tgts=[["B1", "B2", "B3"]])

    def test_bonds_section_present(self):
        out = _estimate_bonded_from_dict(self._bond_dict(), dist_tgts=[["B1", "B2"]], T=T)
        assert "[ bonds ]" in out

    def test_angles_section_present(self):
        out = _estimate_bonded_from_dict(self._angle_dict(), ang_tgts=[["B1", "B2", "B3"]], T=T)
        assert "[ angles ]" in out

    def test_constraint_when_k_exceeds_threshold(self):
        """Very narrow bond → huge k → should be written as a constraint."""
        bins = np.linspace(4.8, 5.2, 500)
        hist = gaussian_hist(bins, loc=5.0, scale=0.005)  # σ ≈ 0.0005 nm → huge k
        d = _minimal_bonded_dict(dist_bins=bins, dist_hist=hist, dist_tgts=[["B1", "B2"]])
        out = _estimate_bonded_from_dict(d, dist_tgts=[["B1", "B2"]], T=T,
                                          constraint_threshold=25000)
        assert "[ constraints ]" in out
        assert "[ bonds ]" not in out

    def test_angle_cap_applied(self):
        """Very narrow angle distribution → k capped and 'capped' noted in output."""
        bins = np.linspace(115, 125, 500)
        hist = gaussian_hist(bins, loc=120.0, scale=0.05)  # enormous k
        d = _minimal_bonded_dict(ang_bins=bins, ang_hist=hist, ang_tgts=[["B1", "B2", "B3"]])
        out = _estimate_bonded_from_dict(d, ang_tgts=[["B1", "B2", "B3"]], T=T,
                                          angle_cap=250.0)
        assert "capped" in out
        assert "250.0" in out

    def test_missing_target_raises(self):
        """Requesting a target not present in the dict should raise ValueError."""
        d = self._bond_dict()
        with pytest.raises(ValueError):
            _estimate_bonded_from_dict(d, dist_tgts=[["B1", "B99"]], T=T)

    def test_bead_names_in_output(self):
        out = _estimate_bonded_from_dict(self._bond_dict(), dist_tgts=[["B1", "B2"]], T=T)
        assert "B1" in out
        assert "B2" in out

    def test_empty_targets_returns_empty_string(self):
        d = _minimal_bonded_dict()
        out = _estimate_bonded_from_dict(d, T=T)
        assert out.strip() == ""


# ---------------------------------------------------------------------------
# Proper dihedral fitting output (harm_dihed_tgts → type 9 lines)
# ---------------------------------------------------------------------------

class TestProperDihedralOutput:
    def _dihed_dict(self, tgt=None):
        tgt = tgt or ["B1", "B2", "B3", "B4"]
        bins, hist = _dihedral_hist(peak_deg=0.0, sigma_deg=30.0)
        return _minimal_bonded_dict(
            dihed_bins=bins, dihed_hist=hist, dihed_tgts=[tgt])

    def test_dihedrals_section_present(self):
        d = self._dihed_dict()
        out = _estimate_bonded_from_dict(d, harm_dihed_tgts=[["B1", "B2", "B3", "B4"]], T=T)
        assert "[ dihedrals ]" in out

    def test_type_9_lines_present(self):
        """Proper dihedral lines must carry function type 9."""
        d = self._dihed_dict()
        out = _estimate_bonded_from_dict(d, harm_dihed_tgts=[["B1", "B2", "B3", "B4"]], T=T)
        type9_lines = [l for l in out.splitlines() if "   9   " in l]
        assert len(type9_lines) >= 1

    def test_type_9_line_format(self):
        """Each type-9 line must have: beads, funct=9, phase, k, mult."""
        d = self._dihed_dict()
        out = _estimate_bonded_from_dict(d, harm_dihed_tgts=[["B1", "B2", "B3", "B4"]], T=T)
        for line in out.splitlines():
            if "   9   " not in line:
                continue
            # Strip inline comments
            tokens = line.split(";")[0].split()
            # tokens: B1 B2 B3 B4  9  phase  k  mult
            assert len(tokens) == 8
            assert tokens[4] == "9"
            float(tokens[5])   # phase — must be a float
            float(tokens[6])   # k     — must be a float
            int(tokens[7])     # mult  — must be an int

    def test_bead_names_in_comment(self):
        """Bead names should appear in the comment line above the type-9 terms."""
        d = self._dihed_dict()
        out = _estimate_bonded_from_dict(d, harm_dihed_tgts=[["B1", "B2", "B3", "B4"]], T=T)
        comment_lines = [l for l in out.splitlines() if l.strip().startswith(";")]
        joined = " ".join(comment_lines)
        assert "B1" in joined and "B4" in joined

    def test_multiplicity_is_positive_integer(self):
        """Multiplicity on every type-9 line must be a positive integer."""
        d = self._dihed_dict()
        out = _estimate_bonded_from_dict(d, harm_dihed_tgts=[["B1", "B2", "B3", "B4"]], T=T)
        for line in out.splitlines():
            if "   9   " not in line:
                continue
            mult = int(line.split(";")[0].split()[-1])
            assert mult >= 1

    def test_amplitude_is_non_negative(self):
        """Force constant k on every type-9 line must be ≥ 0."""
        d = self._dihed_dict()
        out = _estimate_bonded_from_dict(d, harm_dihed_tgts=[["B1", "B2", "B3", "B4"]], T=T)
        for line in out.splitlines():
            if "   9   " not in line:
                continue
            k = float(line.split(";")[0].split()[-2])
            assert k >= 0.0


# ---------------------------------------------------------------------------
# Improper dihedral output (imp_dihed_tgts → type 2 lines)
# ---------------------------------------------------------------------------

class TestImproperDihedralOutput:
    def _imp_dict(self, tgt=None):
        tgt = tgt or ["B1", "B2", "B3", "B4"]
        bins, hist = _dihedral_hist(peak_deg=0.0, sigma_deg=15.0)
        return _minimal_bonded_dict(
            dihed_bins=bins, dihed_hist=hist, dihed_tgts=[tgt])

    def test_dihedrals_section_present(self):
        d = self._imp_dict()
        out = _estimate_bonded_from_dict(d, imp_dihed_tgts=[["B1", "B2", "B3", "B4"]], T=T)
        assert "[ dihedrals ]" in out

    def test_type_2_lines_present(self):
        """Improper dihedral lines must carry function type 2."""
        d = self._imp_dict()
        out = _estimate_bonded_from_dict(d, imp_dihed_tgts=[["B1", "B2", "B3", "B4"]], T=T)
        type2_lines = [l for l in out.splitlines() if "   2   " in l]
        assert len(type2_lines) == 1

    def test_type_2_line_format(self):
        """Each type-2 line must have: beads, funct=2, angle, k."""
        d = self._imp_dict()
        out = _estimate_bonded_from_dict(d, imp_dihed_tgts=[["B1", "B2", "B3", "B4"]], T=T)
        for line in out.splitlines():
            if "   2   " not in line:
                continue
            tokens = line.split(";")[0].split()
            # tokens: B1 B2 B3 B4  2  angle  k
            assert len(tokens) == 7
            assert tokens[4] == "2"
            float(tokens[5])   # equilibrium angle
            float(tokens[6])   # force constant

    def test_equilibrium_angle_near_peak(self):
        """Improper equilibrium angle should be near the histogram peak (0°)."""
        d = self._imp_dict()
        out = _estimate_bonded_from_dict(d, imp_dihed_tgts=[["B1", "B2", "B3", "B4"]], T=T)
        for line in out.splitlines():
            if "   2   " not in line:
                continue
            angle = float(line.split(";")[0].split()[5])
            assert angle == pytest.approx(0.0, abs=5.0)
