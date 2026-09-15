"""
Tests for shaker/dihedral_fitting.py.

Strategy:
 - inverted_boltzmann: feed known densities, verify energy properties
 - fit_periodic_harmonics: fit a known single-cosine potential, verify recovery
 - evaluate_model: round-trip check that reconstructed potential matches input
"""

import numpy as np
import pytest

from shaker.dihedral_fitting import (
    evaluate_model,
    fit_dihedral_workflow,
    fit_periodic_harmonics,
    inverted_boltzmann,
)


def cosine_potential(bins_deg, mult, k, phase_deg=0.0):
    """V(θ) = k * (1 + cos(n·θ − phase))  — GROMACS 1+cos form."""
    th = np.deg2rad(bins_deg)
    return k * (1.0 + np.cos(mult * th - np.deg2rad(phase_deg)))


# ---------------------------------------------------------------------------
# inverted_boltzmann
# ---------------------------------------------------------------------------


class TestInvertedBoltzmann:
    def test_uniform_density_gives_flat_zero_pmf(self):
        """Uniform histogram → PMF is everywhere 0 after zeroing the minimum."""
        hist = np.ones(100)
        pmf = inverted_boltzmann(hist, temp=300)
        assert np.allclose(pmf, 0.0, atol=1e-10)

    def test_minimum_is_zero(self):
        hist = np.array([0.1, 0.5, 1.0, 0.5, 0.1], dtype=float)
        pmf = inverted_boltzmann(hist, temp=300)
        assert np.nanmin(pmf) == pytest.approx(0.0, abs=1e-10)

    def test_peak_bin_has_lowest_energy(self):
        """The highest-density bin should map to the lowest PMF."""
        hist = np.zeros(51)
        hist[25] = 10.0
        hist[24] = hist[26] = 1.0
        pmf = inverted_boltzmann(hist, interpolate=False, temp=300)
        assert np.nanargmin(pmf) == 25

    def test_kcal_vs_kj_ratio(self):
        """kcal=True values should be ~4.184× smaller than kJ/mol values."""
        hist = np.array([0.1, 1.0, 0.1], dtype=float)
        pmf_kj = inverted_boltzmann(hist, kcal=False, temp=300)
        pmf_kcal = inverted_boltzmann(hist, kcal=True, temp=300)
        mask = np.isfinite(pmf_kj) & (pmf_kj > 1e-6)
        if mask.any():
            ratios = pmf_kj[mask] / pmf_kcal[mask]
            assert np.allclose(ratios, 4.184, rtol=1e-2)

    def test_returns_array_same_length(self):
        hist = np.random.rand(72) + 0.01
        pmf = inverted_boltzmann(hist, temp=300)
        assert len(pmf) == 72


# ---------------------------------------------------------------------------
# fit_periodic_harmonics
# ---------------------------------------------------------------------------


class TestFitPeriodicHarmonics:
    def test_single_term_multiplicity_recovered(self):
        """Fit a single-cosine potential — correct multiplicity must be in the model."""
        bins = np.linspace(-180, 180, 72, endpoint=False)
        potential = cosine_potential(bins, mult=2, k=5.0)
        model = fit_periodic_harmonics(
            bins, potential, max_terms=3, max_multiplicity=4, weight_mode="none"
        )
        assert 2 in model["mults"]

    def test_single_term_amplitude_recovered(self):
        """Fitted amplitude should be close to the true k."""
        bins = np.linspace(-180, 180, 72, endpoint=False)
        k_true, mult_true = 4.0, 1
        potential = cosine_potential(bins, mult=mult_true, k=k_true)
        model = fit_periodic_harmonics(
            bins, potential, max_terms=2, max_multiplicity=3, weight_mode="none"
        )
        idx = list(model["mults"]).index(mult_true)
        assert model["amps"][idx] == pytest.approx(k_true, rel=0.05)

    def test_minimum_is_zero_after_zeroing(self):
        """evaluate_model on the fitted model should have min ≈ 0."""
        bins = np.linspace(-180, 180, 72, endpoint=False)
        potential = cosine_potential(bins, mult=3, k=2.0, phase_deg=45.0)
        model = fit_periodic_harmonics(
            bins,
            potential,
            max_terms=3,
            max_multiplicity=4,
            weight_mode="none",
            zero_min=True,
        )
        reconstructed = evaluate_model(bins, model)
        assert np.min(reconstructed) == pytest.approx(0.0, abs=0.05)

    def test_result_keys_present(self):
        bins = np.linspace(-180, 180, 72, endpoint=False)
        potential = cosine_potential(bins, mult=1, k=3.0)
        model = fit_periodic_harmonics(
            bins, potential, max_terms=2, max_multiplicity=2, weight_mode="none"
        )
        for key in ("mults", "amps", "phases_deg", "rss_w", "score", "criterion"):
            assert key in model

    def test_no_valid_data_raises(self):
        with pytest.raises(ValueError, match="No valid data"):
            fit_periodic_harmonics(np.array([np.nan]), np.array([np.nan]))

    def test_bic_selects_simpler_model_than_aic(self):
        """BIC penalises complexity more than AIC, so it should prefer fewer terms."""
        bins = np.linspace(-180, 180, 72, endpoint=False)
        # Noisy single-term potential — BIC should resist adding extra terms
        rng = np.random.default_rng(0)
        potential = cosine_potential(bins, mult=1, k=3.0) + rng.normal(
            0, 0.1, len(bins)
        )
        model_aic = fit_periodic_harmonics(
            bins,
            potential,
            max_terms=4,
            max_multiplicity=4,
            criterion="AIC",
            weight_mode="none",
        )
        model_bic = fit_periodic_harmonics(
            bins,
            potential,
            max_terms=4,
            max_multiplicity=4,
            criterion="BIC",
            weight_mode="none",
        )
        assert len(model_bic["mults"]) <= len(model_aic["mults"])


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------


class TestEvaluateModel:
    def test_roundtrip_single_term(self):
        """Perfect single-term potential should be reconstructed to within noise."""
        bins = np.linspace(-180, 180, 72, endpoint=False)
        potential = cosine_potential(bins, mult=2, k=3.0)
        model = fit_periodic_harmonics(
            bins, potential, max_terms=2, max_multiplicity=3, weight_mode="none"
        )
        reconstructed = evaluate_model(bins, model)
        assert np.allclose(reconstructed, potential, atol=0.2)

    def test_output_shape_matches_input(self):
        bins = np.linspace(-180, 180, 72, endpoint=False)
        potential = cosine_potential(bins, mult=1, k=2.0)
        model = fit_periodic_harmonics(
            bins, potential, max_terms=1, max_multiplicity=2, weight_mode="none"
        )
        out = evaluate_model(bins, model)
        assert out.shape == bins.shape


# ---------------------------------------------------------------------------
# fit_dihedral_workflow (integration)
# ---------------------------------------------------------------------------


class TestFitDihedralWorkflow:
    def test_returns_expected_keys(self):
        bins = np.linspace(-180, 180, 72, endpoint=False)
        # Simple peaked histogram
        hist = np.exp(-0.5 * ((bins - 0) / 30) ** 2) + 0.01
        result = fit_dihedral_workflow(bins, hist, plot=False)
        for key in ("mults", "amps", "potential", "smooth_potential", "report"):
            assert key in result

    def test_report_is_list_of_strings(self):
        bins = np.linspace(-180, 180, 72, endpoint=False)
        hist = np.exp(-0.5 * ((bins) / 30) ** 2) + 0.01
        result = fit_dihedral_workflow(bins, hist, plot=False)
        assert isinstance(result["report"], list)
        assert all(isinstance(r, str) for r in result["report"])

    def test_smooth_potential_same_length(self):
        bins = np.linspace(-180, 180, 72, endpoint=False)
        hist = np.exp(-0.5 * ((bins) / 30) ** 2) + 0.01
        result = fit_dihedral_workflow(bins, hist, plot=False)
        assert len(result["smooth_potential"]) == len(bins)
