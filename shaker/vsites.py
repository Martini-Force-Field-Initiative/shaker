"""Virtual site generation and molecule alignment."""

import numpy as np
import MDAnalysis as md
from MDAnalysis.analysis import align
import itertools
from tqdm.autonotebook import tqdm
from .helper import _bead_mass_from_type

def align_mol_to_single_traj(topology, trajectory,
                             selection="resname MOL",
                             align_selection="name A B C",
                             reference_residue=0,
                             output_xtc="aligned_molecules.xtc",
                             output_gro="reference_molecule.gro",
                             return_average=True,
                             average_gro="average_molecule.gro"):
    """
    Align N molecules over X frames into a single-molecule trajectory with N*X frames.

    Parameters
    ----------
    topology : str
        Topology/structure file readable by MDAnalysis (e.g. .gro, .pdb).
    trajectory : str
        Trajectory file (e.g. .xtc, .trr).
    selection : str, optional
        Atom selection that identifies the molecules of interest.
        The selected atoms are split by residues.
    align_selection : str, optional
        Atom selection applied within each residue to define the alignment frame.
        Must select the same number of atoms in every residue.
    reference_residue : int, optional
        Index of the selected residue to use as structural reference.
    output_xtc : str, optional
        Output trajectory containing one aligned molecule per frame.
    output_gro : str, optional
        Output structure of the reference molecule.
    return_average : bool, optional
        If True, compute the average coordinates over the aligned trajectory
        and write them to ``average_gro``.
    average_gro : str, optional
        Output structure containing the average coordinates of the aligned
        trajectory. Only written if ``return_average=True``.
    """
    
    u = md.Universe(topology, trajectory)
    residues = u.select_atoms(selection).residues
    
    if len(residues) == 0:
        raise ValueError(f"No residues matched selection: {selection!r}")

    if not (0 <= reference_residue < len(residues)):
        raise ValueError(f"reference_residue={reference_residue} is out of range for "
                         f"{len(residues)} selected residues.")

    # Check that all residues contain the same number of atoms
    n_atoms_ref = len(residues[reference_residue].atoms)
    for i, res in enumerate(residues):
        if len(res.atoms) != n_atoms_ref:
            raise ValueError(
                f"Residue {i} ({res.resname} {res.resid}) has {len(res.atoms)} atoms, "
                f"but reference residue has {n_atoms_ref}. "
                "All molecules must have the same number of atoms.")

    # Build reference molecule from the chosen residue
    ref_atoms = residues[reference_residue].atoms
    ref = md.Merge(ref_atoms)
    ref_sel = ref.select_atoms(align_selection)

    if len(ref_sel) == 0:
        raise ValueError(f"align_selection {align_selection!r} selected 0 atoms in the reference residue.")

    # Check that alignment selection is valid and consistent across all residues
    for i, res in enumerate(residues):
        mob_sel = res.atoms.select_atoms(align_selection)
        if len(mob_sel) != len(ref_sel):
            raise ValueError(
                f"Residue {i} ({res.resname} {res.resid}) selected {len(mob_sel)} atoms "
                f"with align_selection, but reference selected {len(ref_sel)}. "
                "Alignment selection must match across all molecules.")

    # Save the reference molecule structure
    ref.atoms.write(output_gro)

    n_frames = len(u.trajectory)
    n_molecules = len(residues)

    # Reusable mobile universe with same topology as one molecule
    mobile = md.Merge(ref_atoms)
    with md.Writer(output_xtc, n_atoms=len(ref.atoms)) as writer:
        for ts in tqdm(u.trajectory, desc="Aligning molecules"):
            for res in residues:
                # Copy current residue coordinates into mobile universe
                mobile.atoms.positions = res.atoms.positions.copy()
                mob_sel = mobile.select_atoms(align_selection)
                align.alignto(mobile, ref, select=align_selection)
                writer.write(mobile.atoms)

    # Load aligned trajectory and compute average structure
    if return_average:
        aligned = md.Universe(output_gro, output_xtc)
        allpos = np.empty((len(aligned.trajectory), len(aligned.atoms), 3),
                          dtype=np.float32)
    
        for i, ts in tqdm(enumerate(aligned.trajectory),
                          total=len(aligned.trajectory),
                          desc="Averaging aligned trajectory",):
            allpos[i] = aligned.atoms.positions
    
        avg_pos = allpos.mean(axis=0)
        aligned.atoms.positions = avg_pos
        aligned.atoms.write(average_gro)


def generate_virtual_sites3(universe, frame_names,
                            selection="all", output="index", c_cutoff=0.1,
                            include_constraints=True, include_exclusions=True,
                            mass_split=None, mapping=None, resname=None):
    
    """
    Build GROMACS [constraints], [virtual_sites3], and [exclusions] entries
    from an existing CG structure.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe containing the CG model.
    frame_names : sequence of str
        Names of the three atoms defining the virtual-site frame, in order.
    selection : str, optional
        Atom selection defining which atoms participate in the virtual-site
        construction. Default is "all".
    output : {"index", "name"}, optional
        Whether to label atoms using 1-based atom indices ("index") or
        atom names ("name") in the generated topology entries.
    c_cutoff : float, optional
        Threshold for switching between GROMACS virtual_sites3 function
        types 3 and 4. If ``abs(c) < c_cutoff``, the site is written as
        function type 3 using only ``a`` and ``b``. Otherwise function
        type 4 is used with ``a``, ``b``, and ``c``.
    include_constraints : bool, optional
        Whether to include a [constraints] section for the three frame atoms.
    include_exclusions : bool, optional
        Whether to include an [exclusions] section excluding all atoms in
        the selected group from one another.
    mass_split : {None, "equal"}, optional
        Optional mass redistribution scheme. If ``None`` (default), no
        mass handling is performed. If ``"equal"``, the total mass of the
        beads defined in ``mapping[resname]`` is redistributed evenly among
        the three frame beads and all other beads are assigned mass 0.0.
    mapping : dict, optional
        Mapping dictionary describing bead types for the residue. Required
        if ``mass_split`` is not ``None``. The bead types are used to infer
        default Martini bead masses.
    resname : str, optional
        Residue name key used to access the bead definitions in ``mapping``.
        Required if ``mass_split`` is not ``None``.
    
    Returns
    -------
    dict
        Dictionary containing the following keys:

        topology_lines : list of str
            Lines for the generated [constraints], [virtual_sites3], and
            [exclusions] topology sections.

        mapping : dict or None
            Updated mapping dictionary containing bead masses if
            ``mass_split`` was applied, otherwise ``None``.
    
    Notes
    -----
    Coordinates are internally converted from Å to nm to match GROMACS
    topology units.
    
    Each site position is expressed in the local frame defined by the
    three constructing atoms as

    .. math::

        r = a * v1 + b * v2 + c * (v1 \\times v2)

    where
    
        v1 = r(frame[1]) − r(frame[0])
        v2 = r(frame[2]) − r(frame[0]).
    """

    cgats = universe.select_atoms(selection)

    if len(cgats) == 0:
        raise ValueError(f"No atoms matched selection {selection!r}")

    if len(frame_names) != 3:
        raise ValueError("frame_names must contain exactly 3 atom names")

    frame_universe_indices = []
    for name in frame_names:
        matches = cgats.select_atoms(f"name {name}")
        if len(matches) == 0:
            raise ValueError(f"No atom named {name!r} found in selection")
        if len(matches) > 1:
            raise ValueError(f"Atom name {name!r} is not unique in selection "
                             f"({len(matches)} matches found)")
        frame_universe_indices.append(matches.indices[0])

    # Map universe indices to local cgats indices
    index_map = {idx: i for i, idx in enumerate(cgats.indices)}
    frame = np.array([index_map[idx] for idx in frame_universe_indices], dtype=int)

    # Work entirely in nm
    pos = cgats.positions.copy() * 0.1

    # Put first frame atom at the origin
    pos -= pos[frame[0]]
    v1 = pos[frame[1]]
    v2 = pos[frame[2]]
    cross = np.cross(v1, v2)
    basis = np.vstack((v1, v2, cross))
    det = np.linalg.det(basis)
    if abs(det) < 1e-12:
        raise ValueError("Chosen frame atoms are collinear or nearly collinear; "
                         "cannot define a virtual-site basis.")

    inv_basis = np.linalg.inv(basis)
    factors = pos @ inv_basis

    def atom_label(i):
        if output == "index":
            return str(cgats.indices[i] + 1)  # GROMACS 1-based
        if output == "name":
            return str(cgats.names[i])
        raise ValueError("output must be 'index' or 'name'")

    lines = []
    frame_labels = [atom_label(i) for i in frame]
    frame_txt = "  ".join(frame_labels)
    
    if include_constraints:
        lines.append("[ constraints ]")
        for i, j in itertools.combinations(frame, 2):
            dist_nm = np.linalg.norm(pos[i] - pos[j])
            lines.append(f"{atom_label(i):>6} {atom_label(j):>6}   1  {dist_nm:.5f}")
        lines.append("")

    lines.append("[ virtual_sites3 ]")
    for i in range(len(cgats)):
        if i in frame:
            continue

        a, b, c = factors[i]
        if abs(c) < c_cutoff:
            lines.append(f"{atom_label(i):>6}   {frame_txt}   3  {a:.5f}  {b:.5f}")
        else:
            lines.append(f"{atom_label(i):>6}   {frame_txt}   4  {a:.5f}  {b:.5f}  {c:.5f}")
    lines.append("")

    if include_exclusions:
        lines.append("[ exclusions ]")
        all_labels = [atom_label(i) for i in range(len(cgats))]
        for i in range(len(all_labels) - 1):
            others = all_labels[i + 1 :]
            lines.append(f"{all_labels[i]:>6}   " + "  ".join(others))
        lines.append("")


    ## Now we deal with the annoying masses. For now only even split
    if mass_split is not None:
        if mapping is None:
            raise ValueError("mapping must be provided when mass_split is used")
        if resname is None:
            raise ValueError("resname must be provided when mass_split is used")
        
        updated_mapping = _add_masses_to_mapping( mapping=mapping, resname=resname,
                                                  selected_names=list(cgats.names),
                                                  frame_names=frame_names,
                                                  mass_split=mass_split )
        return lines, mapping
    else:
        return lines


def _add_masses_to_mapping(mapping, resname, selected_names, frame_names, mass_split):
    """
    Add bead masses to the selected beads in ``mapping[resname]`` according
    to the chosen scheme.

    Parameters
    ----------
    mapping : dict
        Mapping dictionary.
    resname : str
        Residue name key in ``mapping``.
    selected_names : sequence of str
        Bead names included in the virtual-site scheme.
    frame_names : sequence of str
        Names of the three constructing beads.
    mass_split : {"equal"}
        Mass redistribution scheme.

    Returns
    -------
    dict
        Updated mapping dictionary.
    """
    
    bead_map = mapping[resname]
    selected_names = set(selected_names)
    frame_names = set(frame_names)

    ## check all beads make sense...
    missing = selected_names - set(bead_map)
    if missing:
        raise ValueError(f"Selected beads not found in mapping[{resname!r}]: {sorted(missing)}")

    missing_frame = frame_names - selected_names
    if missing_frame:
        raise ValueError(f"Frame beads must be part of the selection: {sorted(missing_frame)}")

    ## Do the thang...
    base_masses = {bead: _bead_mass_from_type(bead_map[bead]["type"])
                   for bead in selected_names}

    if mass_split == "equal":
        total_mass = sum(base_masses.values())
        frame_mass = total_mass / 3.0

        for bead in selected_names:
            bead_map[bead]["mass"] = frame_mass if bead in frame_names else 0.0
    else:
        raise ValueError("Unsupported mass_split type")

    return mapping