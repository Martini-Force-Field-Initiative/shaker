"""Tests for the render_2d module."""

import math
import re
import warnings
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

from shaker.render_2d import (_bead_heavy_names, _infer_net_charge,
                              _isolate_residue, _load_mols, _orient_2d,
                              _require_bead_key, render_2dMapping)


def _structure(*parts):
    path = Path(__file__).parent.joinpath(*parts)
    if not path.exists():
        pytest.skip(f"structure not available: {path}")
    return str(path)


@pytest.fixture
def tutorial_pdb():
    """Single-residue molecule — the common case, no cross-residue bonds."""
    return _structure("..", "tutorials", "AA_references", "BasicParam", "AA.pdb")


@pytest.fixture
def peptide_pdb():
    """Mid-chain residue with a nitro-type sidechain, bonded on both sides."""
    return _structure("data", "PPN.pdb")


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

    def test_bonds_and_charges_are_copied_not_reperceived(self, tutorial_pdb):
        full = Chem.MolFromPDBFile(tutorial_pdb, sanitize=False, removeHs=False)
        rdDetermineBonds.DetermineBonds(full, charge=0)

        isolated, _ = _isolate_residue(full, [a.GetIdx() for a in full.GetAtoms()])

        assert (sorted(str(b.GetBondType()) for b in isolated.GetBonds())
                == sorted(str(b.GetBondType()) for b in full.GetBonds()))
        assert Chem.GetFormalCharge(isolated) == Chem.GetFormalCharge(full)


class TestCrossResidueBondPerception:
    """Regression: a residue bonded to its neighbours must render correctly.

    Perception used to run on the capped fragment, which strips away the
    context constraining ambiguous groups. For this mid-chain residue and
    its nitro-type sidechain that failed outright at the correct charge,
    and "succeeded" at a wrong one by flattening the nitro group to three
    single bonds.
    """

    def test_midchain_residue_loads_at_its_own_charge(self, peptide_pdb):
        """Used to raise ValueError from RDKit at the correct charge."""
        molH, _ = _load_mols(peptide_pdb, "PPN", 0)
        assert Chem.GetFormalCharge(molH) == 0

    def test_nitro_group_keeps_a_double_bond(self, peptide_pdb):
        """The wrong-charge fix flattened the nitro group to all singles."""
        molH, _ = _load_mols(peptide_pdb, "PPN", 0)

        def name(atom):
            info = atom.GetPDBResidueInfo()
            return info.GetName().strip() if info else ""

        orders = [str(b.GetBondType()) for b in molH.GetBonds()
                  if {name(b.GetBeginAtom()), name(b.GetEndAtom())} & {"OJ1", "OJ2"}]
        assert "DOUBLE" in orders, orders


class TestBeadHeavyNames:
    def test_hydrogens_fold_onto_heavy_atoms_and_deduplicate(self):
        hmap = {"HB1": "CB", "HB2": "CB"}
        assert _bead_heavy_names(["CB", "HB1", "HB2", "CB", "CG"], hmap) == ["CB", "CG"]


class TestOrient2D:
    """rotate/mirror let related molecules be drawn in a consistent frame."""

    @pytest.fixture
    def mol(self):
        m = Chem.MolFromSmiles("c1ccccc1C(=O)N")
        AllChem.Compute2DCoords(m)
        return m

    @staticmethod
    def _coords(m):
        conf = m.GetConformer()
        return [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y)
                for i in range(m.GetNumAtoms())]

    def test_mirror_negates_x_about_centroid(self, mol):
        before = self._coords(mol)
        ox = sum(x for x, _ in before) / len(before)
        _orient_2d(mol, mirror=True)
        for (x0, y0), (x1, y1) in zip(before, self._coords(mol)):
            assert x1 == pytest.approx(2 * ox - x0, abs=1e-6)
            assert y1 == pytest.approx(y0, abs=1e-6)

    def test_rotation_preserves_distances(self, mol):
        before = self._coords(mol)
        _orient_2d(mol, rotate=37.5)
        after = self._coords(mol)
        assert math.dist(after[0], after[1]) == pytest.approx(
            math.dist(before[0], before[1]), abs=1e-6)

    def test_mirror_warns_on_stereocentre(self):
        chiral = Chem.MolFromSmiles("C[C@H](N)C(=O)O")
        AllChem.Compute2DCoords(chiral)
        with pytest.warns(UserWarning, match="stereocentre"):
            _orient_2d(chiral, mirror=True)

    def test_no_warning_without_stereocentres(self, mol):
        """Guards the includeUnassigned=True choice from warning on everything."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _orient_2d(mol, mirror=True)


class TestRenderedSvg:
    """End-to-end properties of the emitted SVG."""

    MAPPING = {"MOL": {
        "R1": {"type": "SX3", "charge": 0, "atoms": ["Cl1", "C0B", "C0A", "C05"]},
        "R2": {"type": "SX3", "charge": 0, "atoms": ["C06", "C05", "C08", "Cl0"]},
        "P1": {"type": "SP2", "charge": 0, "atoms": ["O04", "C03", "N02", "H0U"]},
    }}

    def _render(self, pdb, tmp_path, mapping=None, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return render_2dMapping(pdb, "MOL", mapping or self.MAPPING,
                                    out_svg=str(tmp_path / "out.svg"), **kwargs)

    def test_mask_references_resolve_within_the_figure(self, tutorial_pdb, tmp_path):
        """Ids are per-render, so a mask reference must not dangle."""
        svg = self._render(tutorial_pdb, tmp_path)
        used = set(re.findall(r'mask="url\(#([^)]+)\)"', svg))
        assert used <= set(re.findall(r'<mask id="([^"]+)"', svg))

    def test_transparent_by_default_and_opaque_covers_canvas(self, tutorial_pdb, tmp_path):
        assert not re.search(r"<rect style='opacity:1.0;fill:#",
                             self._render(tutorial_pdb, tmp_path))

        svg = self._render(tutorial_pdb, tmp_path, transparent=False)
        view = re.search(r"viewBox='([-\d.\s]+)'", svg).group(1).split()
        bg = re.search(r"<rect style='opacity:1.0;fill:#[0-9A-Fa-f]{6};stroke:none' "
                       r"width='([\d.]+)' height='([\d.]+)'", svg)
        assert bg, "expected an opaque background rect"
        assert float(bg.group(1)) == pytest.approx(float(view[2]), abs=0.01)
        assert float(bg.group(2)) == pytest.approx(float(view[3]), abs=0.01)

    def test_canvas_grows_to_fit_long_labels(self, tutorial_pdb, tmp_path):
        """Labels overhanging the panel widen the canvas instead of clipping.

        The width comes from extents recorded at every drawing site, so this
        also catches a new overlay element that forgets to record its own.
        """
        def view_width(mapping):
            svg = self._render(tutorial_pdb, tmp_path, mapping, size=(950, 480))
            return float(re.search(r"viewBox='[-\d.]+ [-\d.]+ ([\d.]+)", svg).group(1))

        # Long enough to overhang from anywhere on the canvas, so the test
        # doesn't depend on where these particular beads happen to sit.
        long_names = {"MOL": {f"{k}_{'X' * 60}": v
                              for k, v in self.MAPPING["MOL"].items()}}
        assert view_width(self.MAPPING) == pytest.approx(950, abs=0.01)
        assert view_width(long_names) > 950
