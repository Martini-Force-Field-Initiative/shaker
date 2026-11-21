"""
Tests for shaker/mapper.py — AA→CG trajectory mapping.

Uses minimal GRO + XTC files written to a tmp_path fixture (see conftest.py).
Positions are in nm in the GRO file; MDAnalysis exposes them in Å internally.
"""

import warnings
from pathlib import Path

import MDAnalysis as mda
import pytest

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:Reader has no dt information, set to 1\.0 ps:UserWarning:MDAnalysis\.coordinates\.XTC"
    ),
    pytest.mark.filterwarnings(
        r"ignore:Empty box \[0\., 0\., 0\.\] found - treating as missing unit cell.*:UserWarning:MDAnalysis\.coordinates\.GRO"
    ),
]

from shaker.mapper import bead_neighbours, map_aa2cg

# ---------------------------------------------------------------------------
# Mapping definitions reused across tests
# ---------------------------------------------------------------------------

SINGLE_BEAD = {"MOL": {"B1": {"type": "SC3", "charge": 0, "atoms": ["C1", "C2"]}}}

TWO_BEADS = {
    "MOL": {
        "B1": {"type": "SC3", "charge": 0, "atoms": ["C1"]},
        "B2": {"type": "SC3", "charge": 0, "atoms": ["C2"]},
    }
}


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------


class TestOutputFiles:
    def test_gro_created(self, two_atom_gro_xtc, tmp_path):
        gro, xtc = two_atom_gro_xtc
        map_aa2cg(
            gro, xtc, SINGLE_BEAD, outname="out", outdir=str(tmp_path), report=False
        )
        assert (tmp_path / "out.gro").exists()

    def test_xtc_created(self, two_atom_gro_xtc, tmp_path):
        gro, xtc = two_atom_gro_xtc
        map_aa2cg(
            gro, xtc, SINGLE_BEAD, outname="out", outdir=str(tmp_path), report=False
        )
        assert (tmp_path / "out.xtc").exists()

    def test_log_file_created_when_report_true(self, two_atom_gro_xtc, tmp_path):
        gro, xtc = two_atom_gro_xtc
        map_aa2cg(
            gro,
            xtc,
            SINGLE_BEAD,
            outname="out",
            outdir=str(tmp_path),
            report=True,
            verbose=False,
        )
        assert (tmp_path / "out_mapping.log").exists()

    def test_no_log_when_report_false(self, two_atom_gro_xtc, tmp_path):
        gro, xtc = two_atom_gro_xtc
        map_aa2cg(
            gro, xtc, SINGLE_BEAD, outname="out", outdir=str(tmp_path), report=False
        )
        assert not (tmp_path / "out_mapping.log").exists()

    def test_outdir_created_if_missing(self, two_atom_gro_xtc, tmp_path):
        gro, xtc = two_atom_gro_xtc
        new_dir = tmp_path / "subdir" / "nested"
        map_aa2cg(
            gro, xtc, SINGLE_BEAD, outname="out", outdir=str(new_dir), report=False
        )
        assert new_dir.exists()


# ---------------------------------------------------------------------------
# Bead positions
# ---------------------------------------------------------------------------


class TestBeadPositions:
    def test_single_bead_is_cog(self, two_atom_gro_xtc, tmp_path):
        """
        Both atoms in one bead: COG should be the midpoint.
        C1=[0,0,0] nm, C2=[0.3,0,0] nm  →  midpoint = [1.5, 0, 0] Å.
        """
        gro, xtc = two_atom_gro_xtc
        map_aa2cg(
            gro, xtc, SINGLE_BEAD, outname="out", outdir=str(tmp_path), report=False
        )
        cg = mda.Universe(str(tmp_path / "out.gro"))
        assert cg.atoms.positions[0] == pytest.approx([1.5, 0.0, 0.0], abs=0.1)

    def test_two_beads_each_on_own_atom(self, two_atom_gro_xtc, tmp_path):
        """One atom per bead: each bead should sit exactly on its atom."""
        gro, xtc = two_atom_gro_xtc
        map_aa2cg(
            gro, xtc, TWO_BEADS, outname="out", outdir=str(tmp_path), report=False
        )
        cg = mda.Universe(str(tmp_path / "out.gro"))
        assert cg.atoms.positions[0] == pytest.approx([0.0, 0.0, 0.0], abs=0.1)
        assert cg.atoms.positions[1] == pytest.approx([3.0, 0.0, 0.0], abs=0.1)


# ---------------------------------------------------------------------------
# Topology / naming
# ---------------------------------------------------------------------------


class TestTopology:
    def test_bead_names_preserved(self, two_atom_gro_xtc, tmp_path):
        gro, xtc = two_atom_gro_xtc
        map_aa2cg(
            gro, xtc, TWO_BEADS, outname="out", outdir=str(tmp_path), report=False
        )
        cg = mda.Universe(str(tmp_path / "out.gro"))
        assert list(cg.atoms.names) == ["B1", "B2"]

    def test_resname_preserved(self, two_atom_gro_xtc, tmp_path):
        gro, xtc = two_atom_gro_xtc
        map_aa2cg(
            gro, xtc, TWO_BEADS, outname="out", outdir=str(tmp_path), report=False
        )
        cg = mda.Universe(str(tmp_path / "out.gro"))
        assert set(cg.atoms.resnames) == {"MOL"}

    def test_correct_bead_count(self, two_atom_gro_xtc, tmp_path):
        gro, xtc = two_atom_gro_xtc
        map_aa2cg(
            gro, xtc, TWO_BEADS, outname="out", outdir=str(tmp_path), report=False
        )
        cg = mda.Universe(str(tmp_path / "out.gro"))
        assert len(cg.atoms) == 2


# ---------------------------------------------------------------------------
# Bead neighbour detection
# ---------------------------------------------------------------------------


class TestBeadNeighbours:
    def test_warns_for_atom_names_absent_from_structure(self, two_atom_gro_xtc):
        gro, _ = two_atom_gro_xtc
        gro_path = Path(gro)
        gro_path.write_text(gro_path.read_text().replace("0.300", "0.150"))
        mapping = {
            "MOL": {
                "B1": {"type": "SC3", "atoms": ["C1", "C99"]},
                "B2": {"type": "SC3", "atoms": ["C2", "C98", "C99"]},
                "B3": {"type": "SC3", "atoms": ["C97"]},
            }
        }

        with pytest.warns(UserWarning, match="Atom names in the mapping") as recorded:
            result = bead_neighbours(gro, mapping, "MOL", verbose=False)

        assert len(recorded) == 1
        message = str(recorded[0].message)
        assert "residue 'MOL'" in message
        assert "bead 'B1': C99" in message
        assert "bead 'B2': C98, C99" in message
        assert "bead 'B3': C97" in message
        assert result == {"B1": ["B2"], "B2": ["B1"], "B3": []}

    def test_no_missing_atom_warning_for_complete_mapping(self, two_atom_gro_xtc):
        gro, _ = two_atom_gro_xtc
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            bead_neighbours(gro, TWO_BEADS, "MOL", verbose=False)

        assert not any(
            "Atom names in the mapping are absent" in str(warning.message)
            for warning in recorded
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_atom_raises_value_error(self, two_atom_gro_xtc, tmp_path):
        gro, xtc = two_atom_gro_xtc
        bad_mapping = {"MOL": {"B1": {"type": "SC3", "charge": 0, "atoms": ["C99"]}}}
        with pytest.raises(ValueError, match="not found"):
            map_aa2cg(
                gro, xtc, bad_mapping, outname="out", outdir=str(tmp_path), report=False
            )

    def test_no_matching_residues_raises(self, two_atom_gro_xtc, tmp_path):
        gro, xtc = two_atom_gro_xtc
        bad_mapping = {"XXX": {"B1": {"type": "SC3", "charge": 0, "atoms": ["C1"]}}}
        with pytest.raises(ValueError):
            map_aa2cg(
                gro, xtc, bad_mapping, outname="out", outdir=str(tmp_path), report=False
            )
