"""Tests for the render_2d module."""

import pytest
from shaker.render_2d import _infer_net_charge, _require_bead_key


class TestRequireBeadKey:
    """Test suite for the _require_bead_key validation function."""

    def test_valid_bead_map_all_keys_present(self):
        """Test that no error is raised when all beads have the required key."""
        bead_map = {
            "R1": {"type": "SX3", "charge": 0, "atoms": ["C1", "C2"]},
            "R2": {"type": "SX3", "charge": 0, "atoms": ["C3", "C4"]},
            "P1": {"type": "SP2", "charge": 0, "atoms": ["P1", "O1"]},
        }
        # Should not raise
        _require_bead_key(bead_map, "atoms", "MOL")

    def test_valid_bead_map_different_key(self):
        """Test validation of different required keys."""
        bead_map = {
            "R1": {"type": "SX3", "charge": 0},
            "R2": {"type": "SX3", "charge": 1},
        }
        # Should not raise
        _require_bead_key(bead_map, "type", "MOL")
        _require_bead_key(bead_map, "charge", "MOL")

    def test_missing_single_key_in_one_bead(self):
        """Test that ValueError is raised when one bead is missing the key."""
        bead_map = {
            "R1": {"type": "SX3", "charge": 0, "atoms": ["C1"]},
            "R2": {"type": "SX3", "charge": 0},  # Missing "atoms"
            "P1": {"type": "SP2", "charge": 0, "atoms": ["P1"]},
        }
        with pytest.raises(ValueError) as exc_info:
            _require_bead_key(bead_map, "atoms", "MOL")
        
        assert "Beads missing a 'atoms' entry" in str(exc_info.value)
        assert "MOL" in str(exc_info.value)
        assert "R2" in str(exc_info.value)

    def test_missing_key_in_multiple_beads(self):
        """Test that all missing beads are listed in the error message."""
        bead_map = {
            "R1": {"type": "SX3", "atoms": ["C1"]},  # Missing "charge"
            "R2": {"type": "SX3", "atoms": ["C2"]},  # Missing "charge"
            "P1": {"type": "SP2", "charge": 0, "atoms": ["P1"]},
        }
        with pytest.raises(ValueError) as exc_info:
            _require_bead_key(bead_map, "charge", "MOL")
        
        error_msg = str(exc_info.value)
        assert "Beads missing a 'charge' entry" in error_msg
        assert "R1" in error_msg
        assert "R2" in error_msg

    def test_all_beads_missing_key(self):
        """Test error message when all beads are missing the key."""
        bead_map = {
            "R1": {"type": "SX3"},
            "R2": {"type": "TC5"},
        }
        with pytest.raises(ValueError) as exc_info:
            _require_bead_key(bead_map, "atoms", "MOL")
        
        error_msg = str(exc_info.value)
        assert "R1" in error_msg
        assert "R2" in error_msg

    def test_empty_bead_map(self):
        """Test that empty bead_map passes validation."""
        bead_map = {}
        # Should not raise
        _require_bead_key(bead_map, "atoms", "MOL")


class TestInferNetCharge:
    """Test suite for the _infer_net_charge helper."""

    def test_returns_integer_sum_for_valid_charges(self):
        """Infer integer net charge from per-bead charges."""
        bead_map = {
            "B1": {"type": "SQ4p", "charge": 1, "atoms": ["N"]},
            "B2": {"type": "SQ5n", "charge": -1, "atoms": ["CA"]},
            "B3": {"type": "U", "charge": 0, "atoms": ["C"]},
        }
        assert _infer_net_charge(bead_map, "MOL") == 0

    def test_returns_integer_when_float_charges_sum_to_whole_number(self):
        """Allow float bead charges as long as the total is a whole number."""
        bead_map = {
            "SC1": {"type": "SC3", "charge": 0.5, "atoms": ["CB", "CG"]},
            "SC2": {"type": "TP1", "charge": 0.5, "atoms": ["CD", "OH"]},
        }
        value = _infer_net_charge(bead_map, "MOL")
        assert value == 1
        assert isinstance(value, int)

    def test_raises_when_charge_key_is_missing(self):
        """Raise a clear error when any bead has no charge entry."""
        bead_map = {
            "N": {"type": "U", "charge": 0, "atoms": ["N"]},
            "CA": {"type": "U", "atoms": ["CA"]},
        }
        with pytest.raises(ValueError) as exc_info:
            _infer_net_charge(bead_map, "MOL")

        error_msg = str(exc_info.value)
        assert "missing a 'charge' entry" in error_msg
        assert "MOL" in error_msg
        assert "CA" in error_msg
        assert "pass net_charge explicitly" in error_msg

    def test_raises_when_total_charge_is_not_integer(self):
        """Raise when summed charge is non-integer."""
        bead_map = {
            "SC1": {"type": "SC3", "charge": 0.3, "atoms": ["CB", "CG"]},
            "SC2": {"type": "TP1", "charge": 0.3, "atoms": ["CD", "OH"]},
        }
        with pytest.raises(ValueError) as exc_info:
            _infer_net_charge(bead_map, "MOL")

        error_msg = str(exc_info.value)
        assert "sum to a non-integer net charge" in error_msg
        assert "MOL" in error_msg
        assert "0.6" in error_msg
