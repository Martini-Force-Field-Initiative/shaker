"""
Tests for shaker/mapper.py — AA→CG trajectory mapping.

Uses minimal GRO + XTC files written to a tmp_path fixture (see conftest.py).
Positions are in nm in the GRO file; MDAnalysis exposes them in Å internally.
"""

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


@pytest.fixture
def make_structure(tmp_path):
    """Write residue specifications as GRO and single-frame XTC files."""

    def make(residues):
        gro = tmp_path / "input.gro"
        xtc = tmp_path / "input.xtc"
        lines = []
        for resname, resid, atoms in residues:
            for name, x in atoms:
                lines.append(
                    f"{resid:5d}{resname:<5}{name:>5}{len(lines) + 1:5d}"
                    f"{x / 10:8.3f}{0:8.3f}{0:8.3f}\n"
                )
        gro.write_text(
            f"Test system\n{len(lines):5d}\n"
            + "".join(lines)
            + "   5.00000   5.00000   5.00000\n"
        )
        u = mda.Universe(str(gro))
        with mda.Writer(str(xtc), n_atoms=len(u.atoms)) as writer:
            writer.write(u.atoms)
        return gro, xtc

    return make


def call_mapping_function(function, gro, xtc, mapping, tmp_path, resname="MOL"):
    if function == "map":
        return map_aa2cg(gro, xtc, mapping, outdir=tmp_path, report=False)
    return bead_neighbours(gro, mapping, resname, verbose=False)


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

    def test_repeated_mapping_entries_weight_center(self, two_atom_gro_xtc, tmp_path):
        gro, xtc = two_atom_gro_xtc
        mapping = {
            "MOL": {"B1": {"atoms": ["C1", "C1", "C2"]}, "B2": {"atoms": ["C2"]}}
        }
        map_aa2cg(gro, xtc, mapping, outdir=tmp_path, report=False)
        cg = mda.Universe(str(tmp_path / "cg_mapped.gro"))
        assert cg.atoms.positions[0] == pytest.approx([1.0, 0.0, 0.0], abs=0.01)
        assert cg.atoms.positions[1] == pytest.approx([3.0, 0.0, 0.0], abs=0.01)


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
    def test_shared_atom_connects_beads(self, two_atom_gro_xtc):
        gro, _ = two_atom_gro_xtc
        mapping = {"MOL": {"B1": {"atoms": ["C1", "C1"]}, "B2": {"atoms": ["C1"]}}}
        assert bead_neighbours(gro, mapping, "MOL", verbose=False) == {
            "B1": ["B2"],
            "B2": ["B1"],
        }

    def test_only_first_instance_and_requested_mapping_are_validated(
        self, make_structure
    ):
        gro, _ = make_structure(
            [
                ("MOL", 1, [("C1", 0), ("C2", 1.5)]),
                ("MOL", 2, [("C1", 10), ("C1", 11.5)]),
            ]
        )
        mapping = {**TWO_BEADS, "OTHER": {}}
        assert bead_neighbours(gro, mapping, "MOL", verbose=False) == {
            "B1": ["B2"],
            "B2": ["B1"],
        }


class TestMultiResidueMapping:
    def test_maps_all_requested_types_and_instances(self, make_structure, tmp_path):
        gro, xtc = make_structure(
            [
                ("MOL", 1, [("C1", 0), ("C2", 1.5)]),
                ("SOL", 2, [("C1", 4)]),
                ("MOL", 3, [("C1", 8), ("C2", 9.5)]),
            ]
        )
        mapping = {
            "SOL": {"S": {"atoms": ["C1"]}},
            "MOL": {"M": {"atoms": ["C2"]}},
        }
        source = mda.Universe(str(gro))
        with mda.Writer(str(xtc), n_atoms=len(source.atoms)) as writer:
            writer.write(source.atoms)
            source.atoms.positions += [1, 0, 0]
            writer.write(source.atoms)
        map_aa2cg(gro, xtc, mapping, outdir=tmp_path, report=False)
        cg = mda.Universe(
            str(tmp_path / "cg_mapped.gro"), str(tmp_path / "cg_mapped.xtc")
        )
        assert list(cg.atoms.resnames) == ["SOL", "MOL", "MOL"]
        assert list(cg.atoms.resids) == [2, 1, 3]
        assert list(cg.atoms.names) == ["S", "M", "M"]
        assert len(cg.trajectory) == 2
        assert cg.atoms.positions[:, 0] == pytest.approx([4, 1.5, 9.5], abs=0.01)
        cg.trajectory[1]
        assert cg.atoms.positions[:, 0] == pytest.approx([5, 2.5, 10.5], abs=0.01)

    @pytest.mark.parametrize(
        "second_atoms, message",
        [
            ([("C1", 10), ("C1", 11.5)], "Duplicate atom names"),
            ([("C1", 10)], "Atom 'C2' not found"),
        ],
    )
    def test_invalid_later_instance_does_not_write_output(
        self, make_structure, tmp_path, second_atoms, message
    ):
        gro, xtc = make_structure(
            [("MOL", 1, [("C1", 0), ("C2", 1.5)]), ("MOL", 2, second_atoms)]
        )
        output = tmp_path / "cg_mapped.gro"
        output.write_text("existing output")
        with pytest.raises(ValueError, match=message):
            map_aa2cg(gro, xtc, TWO_BEADS, outdir=tmp_path)
        assert output.read_text() == "existing output"
        assert not (tmp_path / "cg_mapped.xtc").exists()
        assert not (tmp_path / "cg_mapped_mapping.log").exists()


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

    def test_missing_residue_among_valid_entries_raises(
        self, two_atom_gro_xtc, tmp_path
    ):
        gro, xtc = two_atom_gro_xtc
        mapping = {**TWO_BEADS, "XXX": {"B": {"atoms": ["C1"]}}}
        with pytest.raises(ValueError, match="No residue named 'XXX'"):
            map_aa2cg(gro, xtc, mapping, outdir=tmp_path, report=False)
        assert not (tmp_path / "cg_mapped.xtc").exists()

    @pytest.mark.parametrize("function", ["map", "neighbours"])
    @pytest.mark.parametrize("duplicate", ["C1", "C3"])
    def test_duplicate_atom_names_rejected_even_if_unmapped(
        self, make_structure, tmp_path, function, duplicate
    ):
        atoms = [("C1", 0), ("C2", 1.5), (duplicate, 3)]
        if duplicate == "C3":
            atoms.append(("C3", 4.5))
        gro, xtc = make_structure([("MOL", 7, atoms)])
        with pytest.raises(
            ValueError, match=f"Duplicate atom names.*MOL.*7.*{duplicate}"
        ):
            call_mapping_function(function, gro, xtc, TWO_BEADS, tmp_path)

    @pytest.mark.parametrize("function", ["map", "neighbours"])
    @pytest.mark.parametrize(
        "mapping, message",
        [
            ({"MOL": {}}, "No beads defined.*MOL"),
            ({"MOL": {"B1": {}}}, "No atoms defined.*B1.*MOL"),
            ({"MOL": {"B1": {"atoms": []}}}, "No atoms defined.*B1.*MOL"),
            ({"MOL": {"B1": {"atoms": ["C99"]}}}, "Atom 'C99' not found.*B1.*MOL"),
            ({"MOL": {"B1": {"atoms": ["C*"]}}}, r"Atom 'C\*' not found"),
        ],
    )
    def test_invalid_selected_mapping_rejected(
        self, two_atom_gro_xtc, tmp_path, function, mapping, message
    ):
        gro, xtc = two_atom_gro_xtc
        with pytest.raises(ValueError, match=message):
            call_mapping_function(function, gro, xtc, mapping, tmp_path)

    @pytest.mark.parametrize("function", ["map", "neighbours"])
    def test_atom_list_cannot_be_a_string(self, two_atom_gro_xtc, tmp_path, function):
        gro, xtc = two_atom_gro_xtc
        mapping = {"MOL": {"B1": {"atoms": "C1"}}}
        with pytest.raises(TypeError, match="Atom names.*B1.*MOL.*list"):
            call_mapping_function(function, gro, xtc, mapping, tmp_path)

    def test_empty_mapping_rejected(self, two_atom_gro_xtc, tmp_path):
        gro, xtc = two_atom_gro_xtc
        with pytest.raises(ValueError, match="at least one residue"):
            map_aa2cg(gro, xtc, {}, outdir=tmp_path, report=False)
        with pytest.raises(ValueError, match="No mapping found"):
            bead_neighbours(gro, {}, "MOL", verbose=False)

    @pytest.mark.parametrize("function", ["map", "neighbours"])
    def test_residue_name_is_literal(self, two_atom_gro_xtc, tmp_path, function):
        gro, xtc = two_atom_gro_xtc
        mapping = {"M*": {"B1": {"atoms": ["C1"]}}}
        with pytest.raises(ValueError, match=r"No residue named 'M\*'"):
            call_mapping_function(function, gro, xtc, mapping, tmp_path, resname="M*")

    def test_literal_pattern_names_and_mapping_report(self, make_structure, tmp_path):
        gro, xtc = make_structure(
            [("MOL", 2, [("C1", 4), ("C2", 5.5)]), ("M*", 1, [("C*", 0)])]
        )
        mapping = {"M*": {"B": {"atoms": ["C*"]}}}
        map_aa2cg(gro, xtc, mapping, outdir=tmp_path, verbose=False)
        cg = mda.Universe(str(tmp_path / "cg_mapped.gro"))
        assert list(cg.atoms.resnames) == ["M*"]
        report = (tmp_path / "cg_mapped_mapping.log").read_text()
        assert "Residue : M*" in report
        assert "Heavy atoms in residue   : 1" in report

    def test_literal_pattern_names_in_neighbours(self, make_structure):
        gro, _ = make_structure(
            [
                ("MOL", 2, [("C1", 4), ("C1", 5.5)]),
                ("M*", 1, [("C*", 0), ("C2", 1.5)]),
            ]
        )
        mapping = {"M*": {"B1": {"atoms": ["C*"]}, "B2": {"atoms": ["C2"]}}}
        assert bead_neighbours(gro, mapping, "M*", verbose=False) == {
            "B1": ["B2"],
            "B2": ["B1"],
        }
