"""Tests for shaker/helper.py — pure bead classification utilities."""

import pytest
from shaker.helper import (
    _size_from_name,
    _bead_mass_from_type,
    _bead_sizes_dict,
    _bead_masses_dict,
    _BEAD_RATIOS,
    _valid_martini_bead_types,
    _validate_bead_types,
)


class TestSizeFromName:
    def test_regular_bead(self):
        assert _size_from_name(["C1"]) == ["R"]
        assert _size_from_name(["P4"]) == ["R"]
        assert _size_from_name(["N4a"]) == ["R"]

    def test_small_bead(self):
        assert _size_from_name(["SC3"]) == ["S"]
        assert _size_from_name(["SP1"]) == ["S"]
        assert _size_from_name(["SN4a"]) == ["S"]

    def test_tiny_bead(self):
        assert _size_from_name(["TC4"]) == ["T"]
        assert _size_from_name(["TP1"]) == ["T"]

    def test_virtual_bead(self):
        # Virtual beads are identified by a leading 'U' 
        assert _size_from_name(["U1"]) == ["U"]
        assert _size_from_name(["u2"]) == ["U"]

    def test_mixed_list(self):
        result = _size_from_name(["C1", "SC3", "TC4", "U1"])
        assert result == ["R", "S", "T", "U"]

    def test_empty_list(self):
        assert _size_from_name([]) == []

    def test_case_insensitive_first_char(self):
        # Function checks string[0].upper()
        assert _size_from_name(["s1"]) == ["S"]
        assert _size_from_name(["t1"]) == ["T"]
        assert _size_from_name(["u1"]) == ["U"]

    def test_returns_list(self):
        result = _size_from_name(["C1", "SC3"])
        assert isinstance(result, list)


class TestBeadMassFromType:
    def test_regular_mass(self):
        assert _bead_mass_from_type("C1") == _bead_masses_dict["R"]
        assert _bead_mass_from_type("P4") == 72.0
        assert _bead_mass_from_type("N4a") == 72.0

    def test_small_mass(self):
        assert _bead_mass_from_type("SC3") == _bead_masses_dict["S"]
        assert _bead_mass_from_type("SC3") == 54.0

    def test_tiny_mass(self):
        assert _bead_mass_from_type("TC4") == _bead_masses_dict["T"]
        assert _bead_mass_from_type("TC4") == 36.0

    def test_size_ordering(self):
        """Regular > Small > Tiny in mass."""
        assert _bead_mass_from_type("C1") > _bead_mass_from_type("SC3")
        assert _bead_mass_from_type("SC3") > _bead_mass_from_type("TC4")


class TestBeadRatios:
    def test_ratios_defined(self):
        assert _BEAD_RATIOS["regular"] == 4
        assert _BEAD_RATIOS["small"] == 3
        assert _BEAD_RATIOS["tiny"] == 2

    def test_sizes_defined(self):
        assert "R" in _bead_sizes_dict
        assert "S" in _bead_sizes_dict
        assert "T" in _bead_sizes_dict

    def test_size_ordering(self):
        """Regular > Small > Tiny in radius."""
        assert _bead_sizes_dict["R"] > _bead_sizes_dict["S"]
        assert _bead_sizes_dict["S"] > _bead_sizes_dict["T"]


class TestValidMartiniBeadTypes:
    def test_known_types_present(self):
        assert {"SC3", "TC4", "P4", "W"} <= _valid_martini_bead_types()

    def test_bogus_type_absent(self):
        assert "ZZZZ" not in _valid_martini_bead_types()


class TestValidateBeadTypes:
    def test_valid_type_passes(self):
        _validate_bead_types({"B1": {"type": "SC3"}})

    def test_missing_type_passes_by_default(self):
        _validate_bead_types({"B1": {"atoms": ["C1"]}})

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="not a recognized Martini"):
            _validate_bead_types({"B1": {"type": "ZZZZ"}})

    def test_missing_type_raises_when_required(self):
        with pytest.raises(ValueError, match="no 'type' defined"):
            _validate_bead_types({"B1": {"atoms": ["C1"]}}, require_type=True)
