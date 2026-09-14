import MDAnalysis as mda
import numpy as np
import pytest


def make_universe(positions, names, resname="MOL", resid=1):
    """Build a single-residue in-memory universe at fixed positions (in Å)."""
    n = len(positions)
    u = mda.Universe.empty(
        n_atoms=n,
        n_residues=1,
        atom_resindex=np.zeros(n, dtype=int),
        trajectory=True,
    )
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("resname", [resname])
    u.add_TopologyAttr("resid", [resid])
    u.atoms.positions = np.array(positions, dtype=np.float32)
    return u


@pytest.fixture
def two_atom_gro_xtc(tmp_path):
    """
    Minimal two-atom GRO + XTC for mapper tests.
    C1 at [0, 0, 0] nm, C2 at [0.3, 0, 0] nm  →  3 Å apart.
    """
    gro = tmp_path / "ref.gro"
    xtc = tmp_path / "ref.xtc"

    gro.write_text(
        "Test system\n"
        "    2\n"
        "    1MOL     C1    1   0.000   0.000   0.000\n"
        "    1MOL     C2    2   0.300   0.000   0.000\n"
        "   5.00000   5.00000   5.00000\n"
    )

    u = mda.Universe(str(gro))
    with mda.Writer(str(xtc), n_atoms=2) as W:
        W.write(u.atoms)

    return str(gro), str(xtc)


@pytest.fixture
def three_atom_gro_xtc(tmp_path):
    """
    Three collinear atoms for angle/distance mapper tests.
    C1=[0,0,0], C2=[0.3,0,0], C3=[0.7,0,0] nm.
    """
    gro = tmp_path / "ref3.gro"
    xtc = tmp_path / "ref3.xtc"

    gro.write_text(
        "Test system\n"
        "    3\n"
        "    1MOL     C1    1   0.000   0.000   0.000\n"
        "    1MOL     C2    2   0.300   0.000   0.000\n"
        "    1MOL     C3    3   0.700   0.000   0.000\n"
        "   5.00000   5.00000   5.00000\n"
    )

    u = mda.Universe(str(gro))
    with mda.Writer(str(xtc), n_atoms=3) as W:
        W.write(u.atoms)

    return str(gro), str(xtc)
