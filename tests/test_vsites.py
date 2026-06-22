"""
Tests for shaker/vsites.py — virtual site generation and molecule alignment.

Geometric tests use in-memory MDAnalysis universes built directly from known
atom positions (Å), chosen so that the resulting nm-space frame basis and/or
virtual-site weights are exact and easy to predict by hand. Alignment tests
use small multi-residue GRO/XTC fixtures written to tmp_path.
"""

import os

import numpy as np
import pytest
import MDAnalysis as mda

from conftest import make_universe
from shaker.vsites import (
    _atom_label,
    _resolve_frame_atoms,
    _solve_vsiten_weights,
    _add_masses_to_mapping,
    align_mol_to_single_traj,
    generate_virtual_sites3,
    generate_virtual_sitesN,
)

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:Reader has no dt information, set to 1\.0 ps:UserWarning:MDAnalysis\.coordinates\.XTC"
    ),
    pytest.mark.filterwarnings(
        r"ignore:Empty box \[0\., 0\., 0\.\] found - treating as missing unit cell.*:UserWarning:MDAnalysis\.coordinates\.GRO"
    ),
    pytest.mark.filterwarnings(
        r"ignore:missing dimension - setting unit cell to zeroed box.*:UserWarning:MDAnalysis\.coordinates\.GRO"
    ),
    pytest.mark.filterwarnings(
        r"ignore:Unknown masses are set to 0\.0.*:PendingDeprecationWarning:MDAnalysis\.guesser\.default_guesser"
    ),
]


# ---------------------------------------------------------------------------
# Helpers for multi-residue GRO/XTC fixtures (align_mol_to_single_traj tests)
# ---------------------------------------------------------------------------

def _gro_line(resid, resname, name, index, pos):
    return (f"{resid:>5}{resname:<5}{name:>5}{index:>5}"
            f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}\n")


def _write_gro(path, atoms):
    """atoms: sequence of (resid, resname, name, pos_nm)."""
    lines = ["Test system\n", f"{len(atoms):>5}\n"]
    for i, (resid, resname, name, pos) in enumerate(atoms, start=1):
        lines.append(_gro_line(resid, resname, name, i, pos))
    lines.append("   5.00000   5.00000   5.00000\n")
    path.write_text("".join(lines))


def _write_xtc_from_universe(path, universe, frames):
    """Write one XTC frame per position array in ``frames`` (Å)."""
    with mda.Writer(str(path), n_atoms=len(universe.atoms)) as w:
        for positions in frames:
            universe.atoms.positions = np.asarray(positions, dtype=np.float32)
            w.write(universe.atoms)


@pytest.fixture
def two_residue_gro_xtc(tmp_path):
    """
    Two MOL residues, each a right triangle A-B-C.
    Residue 2 is residue 1 translated by [50, 0, 0] Å.
    Two trajectory frames: in frame 2, residue 2 is additionally rotated
    90 deg about z, to exercise the per-residue alignment.
    """
    names = ["A", "B", "C"]
    ref_tri = np.array([[0., 0., 0.], [10., 0., 0.], [0., 10., 0.]], dtype=np.float32)
    translate = np.array([50., 0., 0.], dtype=np.float32)

    frame1 = np.vstack([ref_tri, ref_tri + translate])

    theta = np.pi / 2
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]], dtype=np.float32)
    res2_rotated = ref_tri @ rot.T + translate
    frame2 = np.vstack([ref_tri, res2_rotated])

    u = mda.Universe.empty(
        n_atoms=6, n_residues=2,
        atom_resindex=np.repeat([0, 1], 3),
        trajectory=True,
    )
    u.add_TopologyAttr("name", names * 2)
    u.add_TopologyAttr("resname", ["MOL", "MOL"])
    u.add_TopologyAttr("resid", [1, 2])

    gro = tmp_path / "two_res.gro"
    xtc = tmp_path / "two_res.xtc"

    u.atoms.positions = frame1
    u.atoms.write(str(gro))
    _write_xtc_from_universe(xtc, u, [frame1, frame2])

    return str(gro), str(xtc), ref_tri


@pytest.fixture
def mismatched_atom_count_gro_xtc(tmp_path):
    """Residue 1 has 3 atoms (A,B,C), residue 2 has only 2 (A,B)."""
    atoms = [
        (1, "MOL", "A", (0., 0., 0.)),
        (1, "MOL", "B", (1., 0., 0.)),
        (1, "MOL", "C", (0., 1., 0.)),
        (2, "MOL", "A", (5., 0., 0.)),
        (2, "MOL", "B", (6., 0., 0.)),
    ]
    gro = tmp_path / "mismatch.gro"
    _write_gro(gro, atoms)
    u = mda.Universe(str(gro))
    xtc = tmp_path / "mismatch.xtc"
    with mda.Writer(str(xtc), n_atoms=len(u.atoms)) as w:
        w.write(u.atoms)
    return str(gro), str(xtc)


@pytest.fixture
def mismatched_align_selection_gro_xtc(tmp_path):
    """Same atom count (3) per residue, but residue 2 has D instead of C."""
    atoms = [
        (1, "MOL", "A", (0., 0., 0.)),
        (1, "MOL", "B", (1., 0., 0.)),
        (1, "MOL", "C", (0., 1., 0.)),
        (2, "MOL", "A", (5., 0., 0.)),
        (2, "MOL", "B", (6., 0., 0.)),
        (2, "MOL", "D", (5., 1., 0.)),
    ]
    gro = tmp_path / "selmismatch.gro"
    _write_gro(gro, atoms)
    u = mda.Universe(str(gro))
    xtc = tmp_path / "selmismatch.xtc"
    with mda.Writer(str(xtc), n_atoms=len(u.atoms)) as w:
        w.write(u.atoms)
    return str(gro), str(xtc)


# ---------------------------------------------------------------------------
# _atom_label
# ---------------------------------------------------------------------------

class TestAtomLabel:
    def test_index_output(self):
        u = make_universe([[0, 0, 0], [1, 0, 0]], ["A", "B"])
        assert _atom_label(u.atoms, 0, "index") == str(u.atoms.indices[0] + 1)
        assert _atom_label(u.atoms, 1, "index") == str(u.atoms.indices[1] + 1)

    def test_name_output(self):
        u = make_universe([[0, 0, 0], [1, 0, 0]], ["A", "B"])
        assert _atom_label(u.atoms, 0, "name") == "A"
        assert _atom_label(u.atoms, 1, "name") == "B"

    def test_invalid_output_raises(self):
        u = make_universe([[0, 0, 0]], ["A"])
        with pytest.raises(ValueError, match="output must be"):
            _atom_label(u.atoms, 0, "bogus")


# ---------------------------------------------------------------------------
# _resolve_frame_atoms
# ---------------------------------------------------------------------------

class TestResolveFrameAtoms:
    def test_resolves_in_requested_order(self):
        u = make_universe([[0, 0, 0], [1, 0, 0], [2, 0, 0]], ["A", "B", "C"])
        frame, frame_set, frame_pos, frame_labels = _resolve_frame_atoms(
            u.atoms, ["B", "A"], "index")
        assert list(frame) == [1, 0]
        assert frame_set == {0, 1}
        assert frame_labels == [str(u.atoms.indices[1] + 1), str(u.atoms.indices[0] + 1)]
        assert frame_pos == pytest.approx(np.array([[1, 0, 0], [0, 0, 0]]))

    def test_missing_atom_raises(self):
        u = make_universe([[0, 0, 0], [1, 0, 0]], ["A", "B"])
        with pytest.raises(ValueError, match="No atom named 'Z'"):
            _resolve_frame_atoms(u.atoms, ["Z"], "name")

    def test_duplicate_name_raises(self):
        u = make_universe([[0, 0, 0], [1, 0, 0], [2, 0, 0]], ["A", "A", "B"])
        with pytest.raises(ValueError, match="not unique"):
            _resolve_frame_atoms(u.atoms, ["A"], "name")


# ---------------------------------------------------------------------------
# _solve_vsiten_weights
# ---------------------------------------------------------------------------

class TestSolveVsitenWeights:
    def test_exact_line_solution(self):
        """Two frame atoms on a line; target at 30% along → weights [0.7, 0.3]."""
        frame_pos = np.array([[0., 0, 0], [1., 0, 0]])
        A = np.vstack([frame_pos.T, np.ones(2)])
        target = np.array([0.3, 0., 0.])
        w = _solve_vsiten_weights(A, frame_pos, target=target, original=target,
                                   weight_cutoff=1e-6, reconstruction_cutoff=1e-3,
                                   atom_name_for_error="V")
        assert w == pytest.approx([0.7, 0.3], abs=1e-4)

    def test_weights_below_cutoff_are_zeroed_and_renormalized(self):
        frame_pos = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0]])
        A = np.vstack([frame_pos.T, np.ones(3)])
        # Barycentric target: w0=0.3999996, w1=0.6, w2=4e-7 (below default cutoff)
        target = 0.6 * frame_pos[1] + 4e-7 * frame_pos[2]
        w = _solve_vsiten_weights(A, frame_pos, target=target, original=target,
                                   weight_cutoff=1e-6, reconstruction_cutoff=1e-3,
                                   atom_name_for_error="V")
        assert w[2] == 0.0
        assert w.sum() == pytest.approx(1.0)
        assert w[1] == pytest.approx(0.6 / (0.6 + 0.3999996), abs=1e-4)

    def test_reconstruction_error_raises(self):
        """Frame atoms collinear along x; target has a y-component that can't be reproduced."""
        frame_pos = np.array([[0., 0, 0], [1., 0, 0]])
        A = np.vstack([frame_pos.T, np.ones(2)])
        original = np.array([0.5, 1.0, 0.0])
        with pytest.raises(ValueError, match="'V1'"):
            _solve_vsiten_weights(A, frame_pos, target=original, original=original,
                                   weight_cutoff=1e-6, reconstruction_cutoff=1e-3,
                                   atom_name_for_error="V1")


# ---------------------------------------------------------------------------
# generate_virtual_sites3
# ---------------------------------------------------------------------------

class TestGenerateVirtualSites3:
    """
    Frame A=(0,0,0), B=(10,0,0), C=(0,10,0) Å  →  nm basis is the identity,
    so factors (a,b,c) read off directly as the virtual site's nm position.
    D=(3,4,0) Å  → (a,b,c)=(0.3,0.4,0)   → |c| below default cutoff → funct 1.
    E=(2,3,5) Å  → (a,b,c)=(0.2,0.3,0.5) → |c| above default cutoff → funct 4.
    """

    @pytest.fixture
    def universe(self):
        positions = [(0, 0, 0), (10, 0, 0), (0, 10, 0), (3, 4, 0), (2, 3, 5)]
        return make_universe(positions, ["A", "B", "C", "D", "E"])

    def test_funct1_used_below_cutoff(self, universe):
        lines, _ = generate_virtual_sites3(universe, ["A", "B", "C"], output="name")
        d_line = next(l for l in lines if l.strip().startswith("D"))
        assert "1  0.30000  0.40000" in d_line

    def test_funct4_used_above_cutoff(self, universe):
        lines, _ = generate_virtual_sites3(universe, ["A", "B", "C"], output="name")
        e_line = next(l for l in lines if l.strip().startswith("E"))
        assert "4  0.20000  0.30000  0.50000" in e_line

    def test_c_cutoff_can_force_funct1(self, universe):
        """Raising c_cutoff above |c|=0.5 should switch E to funct 1 as well."""
        lines, _ = generate_virtual_sites3(universe, ["A", "B", "C"], output="name",
                                            c_cutoff=0.6)
        e_line = next(l for l in lines if l.strip().startswith("E"))
        assert "1  0.20000  0.30000" in e_line

    def test_constraints_distances(self, universe):
        lines, _ = generate_virtual_sites3(universe, ["A", "B", "C"], output="name")
        assert "A      B   1  1.00000" in lines[1]
        assert "A      C   1  1.00000" in lines[2]
        assert "B      C   1  1.41421" in lines[3]

    def test_constraints_omitted_when_disabled(self, universe):
        lines, _ = generate_virtual_sites3(universe, ["A", "B", "C"], output="name",
                                            include_constraints=False)
        assert not any("constraints" in l for l in lines)

    def test_exclusions_present_by_default(self, universe):
        lines, _ = generate_virtual_sites3(universe, ["A", "B", "C"], output="name")
        assert "[ exclusions ]" in lines

    def test_exclusions_omitted_when_disabled(self, universe):
        lines, _ = generate_virtual_sites3(universe, ["A", "B", "C"], output="name",
                                            include_exclusions=False)
        assert not any("exclusions" in l for l in lines)

    def test_frame_atoms_excluded_from_virtual_sites_block(self, universe):
        lines, _ = generate_virtual_sites3(universe, ["A", "B", "C"], output="name")
        vsite_start = lines.index("[ virtual_sites3 ]")
        vsite_block = lines[vsite_start:vsite_start + 3]
        assert not any(l.strip().startswith(("A ", "B ", "C ")) for l in vsite_block)


class TestGenerateVirtualSites3Errors:
    def test_empty_selection_raises(self):
        u = make_universe([(0, 0, 0)], ["A"])
        with pytest.raises(ValueError, match="No atoms matched"):
            generate_virtual_sites3(u, ["A", "B", "C"], selection="resname XXX")

    def test_wrong_frame_count_raises(self):
        u = make_universe([(0, 0, 0), (1, 0, 0)], ["A", "B"])
        with pytest.raises(ValueError, match="exactly 3 atom names"):
            generate_virtual_sites3(u, ["A", "B"])

    def test_collinear_frame_raises(self):
        u = make_universe([(0, 0, 0), (1, 0, 0), (2, 0, 0)], ["A", "B", "C"])
        with pytest.raises(ValueError, match="collinear"):
            generate_virtual_sites3(u, ["A", "B", "C"])


class TestGenerateVirtualSites3MassSplit:
    MAPPING = {
        "MOL": {
            "A": {"type": "SC3", "atoms": ["x1"]},
            "B": {"type": "SC3", "atoms": ["x2"]},
            "C": {"type": "SC3", "atoms": ["x3"]},
            "D": {"type": "TC4", "atoms": ["x4"]},
            "E": {"type": "TC4", "atoms": ["x5"]},
        }
    }

    @pytest.fixture
    def universe(self):
        positions = [(0, 0, 0), (10, 0, 0), (0, 10, 0), (3, 4, 0), (2, 3, 5)]
        return make_universe(positions, ["A", "B", "C", "D", "E"])

    def test_frame_beads_get_equal_share(self, universe):
        import copy
        mapping = copy.deepcopy(self.MAPPING)
        _, updated = generate_virtual_sites3(universe, ["A", "B", "C"], output="name",
                                              mass_split="equal", mapping=mapping, resname="MOL")
        total = 3 * 54.0 + 2 * 36.0  # 3 S beads + 2 T beads
        for bead in ("A", "B", "C"):
            assert updated["MOL"][bead]["mass"] == pytest.approx(total / 3)

    def test_non_frame_beads_get_zero_mass(self, universe):
        import copy
        mapping = copy.deepcopy(self.MAPPING)
        _, updated = generate_virtual_sites3(universe, ["A", "B", "C"], output="name",
                                              mass_split="equal", mapping=mapping, resname="MOL")
        assert updated["MOL"]["D"]["mass"] == 0.0
        assert updated["MOL"]["E"]["mass"] == 0.0

    def test_no_mapping_returns_none(self, universe):
        _, updated = generate_virtual_sites3(universe, ["A", "B", "C"], output="name")
        assert updated is None

    def test_missing_mapping_raises(self, universe):
        with pytest.raises(ValueError, match="mapping must be provided"):
            generate_virtual_sites3(universe, ["A", "B", "C"], output="name",
                                     mass_split="equal", resname="MOL")

    def test_missing_resname_raises(self, universe):
        import copy
        mapping = copy.deepcopy(self.MAPPING)
        with pytest.raises(ValueError, match="resname must be provided"):
            generate_virtual_sites3(universe, ["A", "B", "C"], output="name",
                                     mass_split="equal", mapping=mapping)


# ---------------------------------------------------------------------------
# generate_virtual_sitesN
# ---------------------------------------------------------------------------

class TestGenerateVirtualSitesNErrors:
    def test_too_few_frame_atoms_raises(self):
        u = make_universe([(0, 0, 0), (1, 0, 0)], ["A", "B"])
        with pytest.raises(ValueError, match="at least 2"):
            generate_virtual_sitesN(u, ["A"])

    def test_too_many_frame_atoms_raises(self):
        u = make_universe([(0, 0, 0)] * 5, ["A", "B", "C", "D", "E"])
        with pytest.raises(ValueError, match="underdetermined"):
            generate_virtual_sitesN(u, ["A", "B", "C", "D", "E"])

    def test_empty_selection_raises(self):
        u = make_universe([(0, 0, 0), (1, 0, 0)], ["A", "B"])
        with pytest.raises(ValueError, match="No atoms matched"):
            generate_virtual_sitesN(u, ["A", "B"], selection="resname XXX")


class TestGenerateVirtualSitesNTwoFrame:
    """F0=(0,0,0) A, F1=(10,0,0) Å B. V=(3,0,0) Å → nm (0.3,0,0) → weights [0.7, 0.3]."""

    @pytest.fixture
    def universe(self):
        return make_universe([(0, 0, 0), (10, 0, 0), (3, 0, 0)], ["A", "B", "V"])

    def test_weights_in_output(self, universe):
        lines, _ = generate_virtual_sitesN(universe, ["A", "B"], output="name")
        v_line = next(l for l in lines if l.strip().startswith("V"))
        assert "A  0.70000" in v_line
        assert "B  0.30000" in v_line

    def test_frame_atoms_excluded_from_vsite_block(self, universe):
        lines, _ = generate_virtual_sitesN(universe, ["A", "B"], output="name")
        start = lines.index("[ virtual_sitesn ]") + 2  # skip header + comment
        end = lines.index("")
        block = lines[start:end]
        assert len(block) == 1
        assert block[0].strip().startswith("V")


class TestGenerateVirtualSitesNThreeFrame:
    """Triangle A=(0,0,0) B=(10,0,0) C=(0,10,0) Å; V at the centroid → equal weights."""

    @pytest.fixture
    def universe(self):
        return make_universe(
            [(0, 0, 0), (10, 0, 0), (0, 10, 0), (10 / 3, 10 / 3, 0)],
            ["A", "B", "C", "V"])

    def test_centroid_gets_equal_weights(self, universe):
        lines, _ = generate_virtual_sitesN(universe, ["A", "B", "C"], output="name")
        v_line = next(l for l in lines if l.strip().startswith("V"))
        for label in ("A", "B", "C"):
            assert f"{label}  0.33333" in v_line


class TestGenerateVirtualSitesNFourFrame:
    """Four coplanar frame atoms forming a square in the z=0 plane."""

    SQUARE = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)]

    def test_tetrahedral_centroid_gets_equal_weights(self):
        """Non-coplanar (tetrahedral) frame: exact square system, no projection."""
        tetra = [(0, 0, 0), (10, 0, 0), (0, 10, 0), (0, 0, 10)]
        centroid_A = np.mean(tetra, axis=0)
        u = make_universe(tetra + [tuple(centroid_A)], ["A", "B", "C", "D", "V"])
        lines, _ = generate_virtual_sitesN(u, ["A", "B", "C", "D"], output="name")
        v_line = next(l for l in lines if l.strip().startswith("V"))
        for label in ("A", "B", "C", "D"):
            assert f"{label}  0.25000" in v_line

    def test_coplanar_frame_projects_small_out_of_plane_noise(self):
        """
        V is 0.005 Å (0.0005 nm) out of the frame's plane — below the default
        1e-3 nm reconstruction_cutoff, so projection should let this succeed
        with sane in-plane weights, instead of the unprojected system
        amplifying the noise into a spurious error.
        """
        u = make_universe(self.SQUARE + [(5, 5, 0.005)],
                           ["A", "B", "C", "D", "V"])
        lines, _ = generate_virtual_sitesN(u, ["A", "B", "C", "D"], output="name")
        v_line = next(l for l in lines if l.strip().startswith("V"))
        for label in ("A", "B", "C", "D"):
            assert f"{label}  0.25000" in v_line

    def test_coplanar_frame_still_rejects_large_out_of_plane_error(self):
        """V genuinely 0.2 nm off the frame's plane must still raise."""
        u = make_universe(self.SQUARE + [(5, 5, 2.0)],
                           ["A", "B", "C", "D", "V"])
        with pytest.raises(ValueError, match="'V'"):
            generate_virtual_sitesN(u, ["A", "B", "C", "D"], output="name")


class TestGenerateVirtualSitesNMassSplit:
    MAPPING = {
        "MOL": {
            "A": {"type": "SC3", "atoms": ["x1"]},
            "B": {"type": "SC3", "atoms": ["x2"]},
            "V": {"type": "TC4", "atoms": ["x3"]},
        }
    }

    @pytest.fixture
    def universe(self):
        return make_universe([(0, 0, 0), (10, 0, 0), (3, 0, 0)], ["A", "B", "V"])

    def test_frame_beads_get_equal_share(self, universe):
        import copy
        mapping = copy.deepcopy(self.MAPPING)
        _, updated = generate_virtual_sitesN(universe, ["A", "B"], output="name",
                                              mass_split="equal", mapping=mapping, resname="MOL")
        total = 2 * 54.0 + 36.0
        assert updated["MOL"]["A"]["mass"] == pytest.approx(total / 2)
        assert updated["MOL"]["B"]["mass"] == pytest.approx(total / 2)
        assert updated["MOL"]["V"]["mass"] == 0.0

    def test_no_mapping_returns_none(self, universe):
        _, updated = generate_virtual_sitesN(universe, ["A", "B"], output="name")
        assert updated is None


# ---------------------------------------------------------------------------
# _add_masses_to_mapping
# ---------------------------------------------------------------------------

class TestAddMassesToMapping:
    def test_equal_split_among_frame_beads(self):
        mapping = {"MOL": {
            "A": {"type": "SC3"}, "B": {"type": "SC3"}, "V": {"type": "TC4"},
        }}
        updated = _add_masses_to_mapping(mapping, "MOL", ["A", "B", "V"], ["A", "B"], "equal")
        total = 54.0 + 54.0 + 36.0
        assert updated["MOL"]["A"]["mass"] == pytest.approx(total / 2)
        assert updated["MOL"]["B"]["mass"] == pytest.approx(total / 2)
        assert updated["MOL"]["V"]["mass"] == 0.0

    def test_modifies_in_place(self):
        mapping = {"MOL": {"A": {"type": "SC3"}, "B": {"type": "SC3"}}}
        updated = _add_masses_to_mapping(mapping, "MOL", ["A", "B"], ["A"], "equal")
        assert updated is mapping

    def test_missing_bead_in_mapping_raises(self):
        mapping = {"MOL": {"A": {"type": "SC3"}}}
        with pytest.raises(ValueError, match="not found in mapping"):
            _add_masses_to_mapping(mapping, "MOL", ["A", "Z"], ["A"], "equal")

    def test_frame_bead_outside_selection_raises(self):
        mapping = {"MOL": {"A": {"type": "SC3"}, "B": {"type": "SC3"}}}
        with pytest.raises(ValueError, match="must be part of the selection"):
            _add_masses_to_mapping(mapping, "MOL", ["A"], ["A", "B"], "equal")

    def test_unsupported_scheme_raises(self):
        mapping = {"MOL": {"A": {"type": "SC3"}}}
        with pytest.raises(ValueError, match="Unsupported mass_split"):
            _add_masses_to_mapping(mapping, "MOL", ["A"], ["A"], "bogus")


# ---------------------------------------------------------------------------
# align_mol_to_single_traj
# ---------------------------------------------------------------------------

class TestAlignMolToSingleTraj:
    def test_output_files_created(self, two_residue_gro_xtc, tmp_path):
        gro, xtc, _ = two_residue_gro_xtc
        out_xtc, out_gro = tmp_path / "aligned.xtc", tmp_path / "ref.gro"
        align_mol_to_single_traj(gro, xtc, selection="resname MOL",
                                  align_selection="name A B C",
                                  output_xtc=str(out_xtc), output_gro=str(out_gro),
                                  return_average=False)
        assert out_xtc.exists()
        assert out_gro.exists()

    def test_one_aligned_frame_per_residue_per_input_frame(self, two_residue_gro_xtc, tmp_path):
        gro, xtc, _ = two_residue_gro_xtc
        out_xtc, out_gro = tmp_path / "aligned.xtc", tmp_path / "ref.gro"
        align_mol_to_single_traj(gro, xtc, selection="resname MOL",
                                  align_selection="name A B C",
                                  output_xtc=str(out_xtc), output_gro=str(out_gro),
                                  return_average=False)
        aligned = mda.Universe(str(out_gro), str(out_xtc))
        # 2 residues x 2 input frames = 4 output frames
        assert len(aligned.trajectory) == 4

    def test_alignment_recovers_reference_geometry(self, two_residue_gro_xtc, tmp_path):
        """
        Residue 2 is a rigid copy of residue 1 (translated, and rotated in
        frame 2). Aligning on the full atom set should map every molecule,
        in every frame, back onto the reference triangle.
        """
        gro, xtc, ref_tri = two_residue_gro_xtc
        out_xtc, out_gro = tmp_path / "aligned.xtc", tmp_path / "ref.gro"
        align_mol_to_single_traj(gro, xtc, selection="resname MOL",
                                  align_selection="name A B C",
                                  output_xtc=str(out_xtc), output_gro=str(out_gro),
                                  return_average=False)
        aligned = mda.Universe(str(out_gro), str(out_xtc))
        for _ in aligned.trajectory:
            assert aligned.atoms.positions == pytest.approx(ref_tri, abs=1e-3)

    def test_average_gro_written_when_requested(self, two_residue_gro_xtc, tmp_path):
        gro, xtc, _ = two_residue_gro_xtc
        out_xtc, out_gro = tmp_path / "aligned.xtc", tmp_path / "ref.gro"
        avg_gro = tmp_path / "avg.gro"
        align_mol_to_single_traj(gro, xtc, selection="resname MOL",
                                  align_selection="name A B C",
                                  output_xtc=str(out_xtc), output_gro=str(out_gro),
                                  return_average=True, average_gro=str(avg_gro))
        assert avg_gro.exists()

    def test_average_not_written_when_disabled(self, two_residue_gro_xtc, tmp_path):
        gro, xtc, _ = two_residue_gro_xtc
        out_xtc, out_gro = tmp_path / "aligned.xtc", tmp_path / "ref.gro"
        avg_gro = tmp_path / "avg.gro"
        align_mol_to_single_traj(gro, xtc, selection="resname MOL",
                                  align_selection="name A B C",
                                  output_xtc=str(out_xtc), output_gro=str(out_gro),
                                  return_average=False, average_gro=str(avg_gro))
        assert not avg_gro.exists()


class TestAlignMolToSingleTrajErrors:
    def test_no_residues_matched_raises(self, two_residue_gro_xtc, tmp_path):
        gro, xtc, _ = two_residue_gro_xtc
        with pytest.raises(ValueError, match="No residues matched"):
            align_mol_to_single_traj(gro, xtc, selection="resname XXX",
                                      align_selection="name A B C",
                                      output_xtc=str(tmp_path / "o.xtc"),
                                      output_gro=str(tmp_path / "o.gro"),
                                      return_average=False)

    def test_reference_residue_out_of_range_raises(self, two_residue_gro_xtc, tmp_path):
        gro, xtc, _ = two_residue_gro_xtc
        with pytest.raises(ValueError, match="out of range"):
            align_mol_to_single_traj(gro, xtc, selection="resname MOL",
                                      align_selection="name A B C",
                                      reference_residue=5,
                                      output_xtc=str(tmp_path / "o.xtc"),
                                      output_gro=str(tmp_path / "o.gro"),
                                      return_average=False)

    def test_mismatched_atom_counts_raises(self, mismatched_atom_count_gro_xtc, tmp_path):
        gro, xtc = mismatched_atom_count_gro_xtc
        with pytest.raises(ValueError, match="must have the same number of atoms"):
            align_mol_to_single_traj(gro, xtc, selection="resname MOL",
                                      align_selection="name A B",
                                      output_xtc=str(tmp_path / "o.xtc"),
                                      output_gro=str(tmp_path / "o.gro"),
                                      return_average=False)

    def test_empty_align_selection_raises(self, mismatched_align_selection_gro_xtc, tmp_path):
        gro, xtc = mismatched_align_selection_gro_xtc
        with pytest.raises(ValueError, match="selected 0 atoms"):
            align_mol_to_single_traj(gro, xtc, selection="resname MOL",
                                      align_selection="name ZZZZ",
                                      output_xtc=str(tmp_path / "o.xtc"),
                                      output_gro=str(tmp_path / "o.gro"),
                                      return_average=False)

    def test_mismatched_align_selection_across_residues_raises(
            self, mismatched_align_selection_gro_xtc, tmp_path):
        gro, xtc = mismatched_align_selection_gro_xtc
        with pytest.raises(ValueError, match="Alignment selection must match"):
            align_mol_to_single_traj(gro, xtc, selection="resname MOL",
                                      align_selection="name A B C",
                                      output_xtc=str(tmp_path / "o.xtc"),
                                      output_gro=str(tmp_path / "o.gro"),
                                      return_average=False)
