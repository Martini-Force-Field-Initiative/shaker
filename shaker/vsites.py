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

    # Reusable mobile universe with same topology as one molecule
    mobile = md.Merge(ref_atoms)
    with md.Writer(output_xtc, n_atoms=len(ref.atoms)) as writer:
        for ts in tqdm(u.trajectory, desc="Aligning molecules"):
            for res in residues:
                mobile.atoms.positions = res.atoms.positions.copy()
                align.alignto(mobile, ref, select=align_selection)
                writer.write(mobile.atoms)

    # Load aligned trajectory and compute average structure
    if return_average:
        aligned = md.Universe(output_gro, output_xtc)
        allpos = np.empty((len(aligned.trajectory), len(aligned.atoms), 3),
                          dtype=np.float32)

        for i, ts in tqdm(enumerate(aligned.trajectory),
                          total=len(aligned.trajectory),
                          desc="Averaging aligned trajectory"):
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
        types 1 and 4. If ``abs(c) < c_cutoff``, the site is written as
        function type 1 using only ``a`` and ``b``. Otherwise function
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
    tuple
        A 2-tuple containing:

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
    frame_set = set(frame.tolist())

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

    lines = []
    frame_labels = [_atom_label(cgats, i, output) for i in frame]
    frame_txt = "  ".join(frame_labels)

    if include_constraints:
        lines.append("[ constraints ]")
        for i, j in itertools.combinations(frame, 2):
            dist_nm = np.linalg.norm(pos[i] - pos[j])
            lines.append(f"{_atom_label(cgats, i, output):>6} "
                         f"{_atom_label(cgats, j, output):>6}   1  {dist_nm:.5f}")
        lines.append("")

    lines.append("[ virtual_sites3 ]")
    for i in range(len(cgats)):
        if i in frame_set:
            continue

        a, b, c = factors[i]
        if abs(c) < c_cutoff:
            lines.append(f"{_atom_label(cgats, i, output):>6}   {frame_txt}   1  {a:.5f}  {b:.5f}")
        else:
            lines.append(f"{_atom_label(cgats, i, output):>6}   {frame_txt}   4  {a:.5f}  {b:.5f}  {c:.5f}")
    lines.append("")

    if include_exclusions:
        lines.append("[ exclusions ]")
        all_labels = [_atom_label(cgats, i, output) for i in range(len(cgats))]
        for i in range(len(all_labels) - 1):
            lines.append(f"{all_labels[i]:>6}   " + "  ".join(all_labels[i + 1:]))
        lines.append("")

    if mass_split is not None:
        if mapping is None:
            raise ValueError("mapping must be provided when mass_split is used")
        if resname is None:
            raise ValueError("resname must be provided when mass_split is used")

        updated_mapping = _add_masses_to_mapping(mapping=mapping, resname=resname,
                                                 selected_names=list(cgats.names),
                                                 frame_names=frame_names,
                                                 mass_split=mass_split)
        return lines, updated_mapping
    else:
        return lines, None


def generate_virtual_sitesN(universe, frame_names,
                             selection="all", output="index",
                             include_exclusions=True,
                             weight_cutoff=1e-6,
                             mass_split=None, mapping=None, resname=None):
    """
    Build GROMACS [virtual_sitesn] and [exclusions] entries from an existing
    CG structure using a linear combination of N frame atoms with relative
    weights (center of weights, function type 3).

    For each non-frame atom in the selection, the normalized weights are found
    analytically by solving:

    .. math::

        \\mathbf{r}_s = \\sum_{i=1}^N w_i \\, \\mathbf{r}_i,
        \\quad \\sum_{i=1}^N w_i = 1

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe containing the CG model. Should contain a single averaged
        structure (this construction is only meaningful for rigid molecules).
    frame_names : sequence of str
        Names of the N atoms defining the constructing frame, in order.
        Must contain at least 2 atoms and at most 4 atoms (3 spatial dimensions
        limit the system to N-1 <= 3 independent weights).
    selection : str, optional
        Atom selection defining which atoms participate. Frame atoms and
        virtual sites must all be within this selection. Default is "all".
    output : {"index", "name"}, optional
        Whether to label atoms using 1-based atom indices ("index") or
        atom names ("name") in the generated topology entries.
    include_exclusions : bool, optional
        Whether to include an [exclusions] section excluding all atoms in
        the selected group from one another.
    weight_cutoff : float, optional
        Weights below this threshold are set to zero. Default is 1e-6.
    mass_split : {None, "equal"}, optional
        Optional mass redistribution scheme. If ``None`` (default), no
        mass handling is performed. If ``"equal"``, the total mass of the
        beads defined in ``mapping[resname]`` is redistributed evenly among
        the N frame beads and all other beads are assigned mass 0.0.
    mapping : dict, optional
        Mapping dictionary describing bead types for the residue. Required
        if ``mass_split`` is not ``None``.
    resname : str, optional
        Residue name key used to access the bead definitions in ``mapping``.
        Required if ``mass_split`` is not ``None``.

    Returns
    -------
    tuple
        A 2-tuple containing:

        topology_lines : list of str
            Lines for the generated [virtual_sitesn] and [exclusions]
            topology sections.

        mapping : dict or None
            Updated mapping dictionary containing bead masses if
            ``mass_split`` was applied, otherwise ``None``.

    Raises
    ------
    ValueError
        If any frame atom is not found or not unique in the selection,
        if fewer than 2 or more than 4 frame atoms are provided, if the
        frame atoms are collinear (underdetermined system), or if any solved
        weight is negative beyond ``weight_cutoff``.

    Notes
    -----
    Coordinates are internally converted from Å to nm to match GROMACS
    topology units.

    The topology block format is:

    .. code-block:: text

        [ virtual_sitesn ]
        ; site   funct   constructing_atom_1  weight_1  constructing_atom_2  weight_2 ...
        s        3       i  w_i  j  w_j  k  w_k  ...
    """

    if len(frame_names) < 2:
        raise ValueError("frame_names must contain at least 2 atom names.")

    if len(frame_names) > 4:
        raise ValueError(
            f"frame_names contains {len(frame_names)} atoms, but the system is "
            "underdetermined with more than 4 frame atoms in 3 spatial dimensions "
            "(at most N-1=3 independent weights can be solved).")

    cgats = universe.select_atoms(selection)
    if len(cgats) == 0:
        raise ValueError(f"No atoms matched selection {selection!r}")

    index_map = {idx: i for i, idx in enumerate(cgats.indices)}

    frame_local_indices = []
    for name in frame_names:
        matches = cgats.select_atoms(f"name {name}")
        if len(matches) == 0:
            raise ValueError(f"No atom named {name!r} found in selection")
        if len(matches) > 1:
            raise ValueError(f"Atom name {name!r} is not unique in selection "
                             f"({len(matches)} matches found)")
        frame_local_indices.append(index_map[matches.indices[0]])

    frame = np.array(frame_local_indices, dtype=int)
    frame_set = set(frame.tolist())

    # Work in nm
    pos = cgats.positions.copy() * 0.1
    frame_pos = pos[frame]  # shape (N, 3)

    # Build the least-squares system enforcing sum(w) = 1
    # Substituting w_N = 1 - sum(w_1..w_{N-1}):
    # r_s - r_N = sum_{i=0}^{N-2} w_i * (r_i - r_N)
    # Shape: (3, N-1)
    A = (frame_pos[:-1] - frame_pos[-1]).T

    frame_labels = [_atom_label(cgats, i, output) for i in frame]

    lines = []
    lines.append("[ virtual_sitesn ]")
    lines.append("; site   funct   constructing_atom_1  weight_1  constructing_atom_2  weight_2 ...")

    for i in range(len(cgats)):
        if i in frame_set:
            continue

        rhs = pos[i] - frame_pos[-1]  # shape (3,)
        w_partial, _, rank, _ = np.linalg.lstsq(A, rhs, rcond=None)

        if rank < A.shape[1]:
            raise ValueError(
                f"Frame atoms are collinear or coplanar in a degenerate way; "
                f"cannot solve for weights of atom {cgats.names[i]!r}.")

        w_last = 1.0 - w_partial.sum()
        w_full = np.append(w_partial, w_last)
        w_full[np.abs(w_full) < weight_cutoff] = 0.0

        if np.any(w_full < 0):
            raise ValueError(
                f"Atom {cgats.names[i]!r} produced negative weights {w_full}. "
                "This suggests the virtual site lies outside the convex hull "
                "of the frame atoms. Check your frame selection or structure.")

        w_full /= w_full.sum()  # renormalize after zeroing

        pairs_str = "  ".join(f"{lbl}  {w:.5f}"
                               for lbl, w in zip(frame_labels, w_full))
        lines.append(f"{_atom_label(cgats, i, output):>6}   3   {pairs_str}")

    lines.append("")

    if include_exclusions:
        lines.append("[ exclusions ]")
        all_labels = [_atom_label(cgats, i, output) for i in range(len(cgats))]
        for i in range(len(all_labels) - 1):
            lines.append(f"{all_labels[i]:>6}   {'  '.join(all_labels[i + 1:])}")
        lines.append("")

    if mass_split is not None:
        if mapping is None:
            raise ValueError("mapping must be provided when mass_split is used")
        if resname is None:
            raise ValueError("resname must be provided when mass_split is used")

        updated_mapping = _add_masses_to_mapping(mapping=mapping, resname=resname,
                                                 selected_names=list(cgats.names),
                                                 frame_names=frame_names,
                                                 mass_split=mass_split)
        return lines, updated_mapping
    else:
        return lines, None


def _atom_label(atom_group, i, output):
    """
    Return a GROMACS-compatible label for atom at local index ``i``.

    Parameters
    ----------
    atom_group : MDAnalysis.AtomGroup
        The atom group containing the atom.
    i : int
        Local index within ``atom_group``.
    output : {"index", "name"}
        Label type: 1-based integer index or atom name.

    Returns
    -------
    str
        Atom label string.
    """
    if output == "index":
        return str(atom_group.indices[i] + 1)
    if output == "name":
        return str(atom_group.names[i])
    raise ValueError("output must be 'index' or 'name'")

def _add_masses_to_mapping(mapping, resname, selected_names, frame_names, mass_split):
    """
    Add bead masses to the selected beads in ``mapping[resname]`` according
    to the chosen scheme.

    Parameters
    ----------
    mapping : dict
        Mapping dictionary. Modified in-place.
    resname : str
        Residue name key in ``mapping``.
    selected_names : sequence of str
        Bead names included in the virtual-site scheme.
    frame_names : sequence of str
        Names of the N constructing beads.
    mass_split : {"equal"}
        Mass redistribution scheme.

    Returns
    -------
    dict
        Updated mapping dictionary (same object as input, modified in-place).

    Notes
    -----
    The mapping dictionary is modified in-place and also returned for
    convenience. If you need to preserve the original, pass a deep copy.
    """

    bead_map = mapping[resname]
    selected_names = set(selected_names)
    frame_names = set(frame_names)

    missing = selected_names - set(bead_map)
    if missing:
        raise ValueError(f"Selected beads not found in mapping[{resname!r}]: {sorted(missing)}")

    missing_frame = frame_names - selected_names
    if missing_frame:
        raise ValueError(f"Frame beads must be part of the selection: {sorted(missing_frame)}")

    base_masses = {bead: _bead_mass_from_type(bead_map[bead]["type"])
                   for bead in selected_names}

    if mass_split == "equal":
        total_mass = sum(base_masses.values())
        frame_mass = total_mass / len(frame_names)

        for bead in selected_names:
            bead_map[bead]["mass"] = frame_mass if bead in frame_names else 0.0
    else:
        raise ValueError("Unsupported mass_split type")

    return mapping