"""Tests for the render_2d module."""

from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from shaker.render_2d import (_infer_net_charge, _isolate_residue, _load_mols,
                              _require_bead_key)


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


class TestIsolateResidue:
    """Bond orders and formal charges must survive residue isolation.

    `_isolate_residue` copies them from the already-perceived full
    structure rather than re-running bond determination on the fragment,
    which loses the context that constrains ambiguous groups.
    """

    @pytest.fixture
    def pdb_path(self):
        path = Path(__file__).parent.parent / "tutorials" / "AA_references" / \
            "BasicParam" / "AA.pdb"
        if not path.exists():
            pytest.skip(f"reference structure not available: {path}")
        return str(path)

    def test_bond_orders_are_copied_not_reperceived(self, pdb_path):
        """The isolated residue keeps the full structure's bond orders."""
        full = Chem.MolFromPDBFile(pdb_path, sanitize=False, removeHs=False)
        rdDetermineBonds.DetermineBonds(full, charge=0)
        target_idx = [a.GetIdx() for a in full.GetAtoms()]

        isolated, _ = _isolate_residue(full, target_idx)

        expected = sorted(str(b.GetBondType()) for b in full.GetBonds())
        actual = sorted(str(b.GetBondType()) for b in isolated.GetBonds())
        assert actual == expected

    def test_isolated_residue_is_not_bond_free(self, pdb_path):
        """Regression: the fragment used to come back with no bonds at all."""
        full = Chem.MolFromPDBFile(pdb_path, sanitize=False, removeHs=False)
        rdDetermineBonds.DetermineBonds(full, charge=0)

        isolated, _ = _isolate_residue(full, [a.GetIdx() for a in full.GetAtoms()])
        assert isolated.GetNumBonds() > 0

    def test_multiple_bond_orders_present(self, pdb_path):
        """Double/aromatic bonds survive, not flattened to all-single."""
        full = Chem.MolFromPDBFile(pdb_path, sanitize=False, removeHs=False)
        rdDetermineBonds.DetermineBonds(full, charge=0)

        isolated, _ = _isolate_residue(full, [a.GetIdx() for a in full.GetAtoms()])
        orders = {str(b.GetBondType()) for b in isolated.GetBonds()}
        assert len(orders) > 1, f"expected mixed bond orders, got {orders}"

    def test_formal_charges_are_copied(self, pdb_path):
        """Net formal charge of the slice matches the source structure."""
        full = Chem.MolFromPDBFile(pdb_path, sanitize=False, removeHs=False)
        rdDetermineBonds.DetermineBonds(full, charge=0)

        isolated, _ = _isolate_residue(full, [a.GetIdx() for a in full.GetAtoms()])
        assert Chem.GetFormalCharge(isolated) == Chem.GetFormalCharge(full)


class TestCrossResidueBondPerception:
    """Regression: a residue bonded to its neighbours must render correctly.

    Bond perception used to run on the capped fragment, which strips away
    the context constraining ambiguous groups. For a mid-chain residue
    carrying a nitro-type group this failed outright at the correct
    charge, and "succeeded" at a wrong one by flattening the nitro group
    to three single bonds.
    """

    @pytest.fixture
    def peptide_pdb(self):
        path = Path(__file__).parent / "data" / "PPN.pdb"
        if not path.exists():
            pytest.skip(f"fixture not available: {path}")
        return str(path)

    def test_midchain_residue_loads_at_its_own_charge(self, peptide_pdb):
        """Used to raise ValueError from RDKit at the correct charge."""
        molH, mol = _load_mols(peptide_pdb, "PPN", 0)
        assert Chem.GetFormalCharge(molH) == 0

    def test_cross_residue_bonds_are_capped(self, peptide_pdb):
        """Both peptide bonds become caps, shown as R atoms."""
        molH, mol = _load_mols(peptide_pdb, "PPN", 0)
        caps = [a for a in molH.GetAtoms() if a.GetAtomicNum() == 0]
        assert len(caps) == 2

    def test_nitro_group_keeps_a_double_bond(self, peptide_pdb):
        """The failure mode was a nitro group flattened to all single bonds."""
        molH, mol = _load_mols(peptide_pdb, "PPN", 0)

        def name(atom):
            info = atom.GetPDBResidueInfo()
            return info.GetName().strip() if info else ""

        nitro_bonds = {
            frozenset((name(b.GetBeginAtom()), name(b.GetEndAtom()))):
                str(b.GetBondType())
            for b in molH.GetBonds()
            if {name(b.GetBeginAtom()), name(b.GetEndAtom())} & {"OJ1", "OJ2"}
        }
        assert "DOUBLE" in nitro_bonds.values(), nitro_bonds

    def test_nitro_group_formal_charges(self, peptide_pdb):
        """Standard nitro resonance form: N+ balanced by O-, netting zero."""
        molH, mol = _load_mols(peptide_pdb, "PPN", 0)
        charges = {
            a.GetPDBResidueInfo().GetName().strip(): a.GetFormalCharge()
            for a in molH.GetAtoms()
            if a.GetPDBResidueInfo() and a.GetFormalCharge()
        }
        assert charges.get("NH") == 1
        assert sum(charges.values()) == 0
