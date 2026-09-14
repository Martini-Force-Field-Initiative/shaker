"""Map AA/QM trajectories to CG representations."""

import warnings
from pathlib import Path

import MDAnalysis as md
import numpy as np
from tqdm.autonotebook import tqdm

from .helper import _BEAD_RATIOS, _size_from_name, _size_label


def _write_mapping_report(u, mapping, outdir, outname, verbose=True):
    """
    Write a mapping quality report to a log file.

    For each residue in the mapping, reports the number of heavy atoms,
    beads produced (with type breakdown if bead types are available), and
    the bead-to-atom mismatch evaluated against the Martini 3 tolerance
    (±1 per 10 heavy atoms).

    Bead size classes are inferred from the bead type string via
    `_size_from_name` (R=regular, S=small, T=tiny, U=virtual). Virtual
    beads contribute 0 to the expected heavy atom count. If no ``type``
    fields are present in the mapping, an unweighted 4:1 ratio is assumed.
    If only some beads have a type field, a warning is emitted and the
    unweighted ratio is used.

    Parameters
    ----------
    u : MDAnalysis.Universe
        The atomistic universe.
    mapping : dict
        The atom-to-bead mapping dictionary (same format as in `map_aa2cg`).
    outdir : Path
        Directory where the log file will be written.
    outname : str
        Base name used for the log file ({outname}_mapping.log).
    verbose : bool, optional
        If ``True`` (default), print the report to the terminal in addition
        to writing it to the log file.
    """
    log_path = outdir / f"{outname}_mapping.log"

    lines = []
    lines.append("Mapping summary")
    lines.append("=" * 50)
    lines.append("")

    for resname, beads in mapping.items():
        residues = u.select_atoms(f"resname {resname}").residues
        if len(residues) == 0:
            continue

        ref_res = residues[0]
        n_heavy = len(ref_res.atoms.select_atoms("not name H*"))
        n_beads = len(beads)

        all_typed = all("type" in bead_def for bead_def in beads.values())
        any_typed = any("type" in bead_def for bead_def in beads.values())

        if any_typed and not all_typed:
            missing = [
                name for name, bead_def in beads.items() if "type" not in bead_def
            ]
            lines.append(
                f"  WARNING: mixed typing for {resname} — "
                f"beads missing type: {missing}. "
                f"Falling back to unweighted ratio."
            )
            lines.append("")

        if all_typed:
            bead_types = [bead_def["type"] for bead_def in beads.values()]
            size_classes = _size_from_name(bead_types)
            type_counts = {"R": 0, "S": 0, "T": 0, "U": 0}
            for sc in size_classes:
                type_counts[sc] += 1

            type_str = (
                f"[{type_counts['R']} regular, "
                f"{type_counts['S']} small, "
                f"{type_counts['T']} tiny"
                + (f", {type_counts['U']} virtual]" if type_counts["U"] else "]")
            )

            expected_heavy = sum(
                _BEAD_RATIOS.get(_size_label.get(sc, "regular"), 4)
                for sc in size_classes
                if sc != "U"
            )
            ratio_label = "Bead/atom mismatch (weighted)  "
        else:
            type_str = "[bead types not specified]"
            expected_heavy = n_beads * 4
            ratio_label = "Bead/atom mismatch (unweighted)"

        ratio_diff = n_heavy - expected_heavy
        tolerance = max(1, round(n_heavy / 10))
        abs_diff = abs(ratio_diff)

        if abs_diff == 0:
            ratio_status = "OK"
        elif abs_diff <= tolerance:
            ratio_status = f"ACCEPTABLE (within Martini 3 tolerance of ±{tolerance})"
        elif ratio_diff > tolerance:
            ratio_status = f"UNDER-MAPPED (exceeds tolerance of ±{tolerance})"
        else:
            ratio_status = f"OVER-MAPPED  (exceeds tolerance of ±{tolerance})"

        lines.append(f"Residue : {resname}")
        lines.append(f"  Heavy atoms in residue   : {n_heavy}")
        lines.append(f"  Beads produced           : {n_beads} {type_str}")
        lines.append(f"  {ratio_label}  : {ratio_diff:+d}  [{ratio_status}]")
        lines.append("")

    report_str = "\n".join(lines)

    with open(log_path, "w") as f:
        f.write(report_str)

    print(f"Mapping log written to {log_path}\n")

    if verbose:
        print(report_str)


def map_aa2cg(
    gro, xtc, mapping, outname="cg_mapped", outdir=".", report=True, verbose=True
):
    """
    Map an atomistic trajectory to a coarse-grained representation.

    This function constructs coarse-grained beads from atomistic coordinates by
    computing the center of geometry of user-defined groups of atoms for each
    residue. The mapping is applied to every frame of the input trajectory,
    producing a coarse-grained trajectory and a representative structure.

    Parameters
    ----------
    gro : str or Path
        Atomistic structure file (.gro or .pdb) used to initialize the system.
    xtc : str or Path
        Atomistic trajectory file (.xtc or .trr) containing the coordinates
        to be mapped.
    mapping : dict
        Dictionary defining the atom-to-bead mapping. The expected format is::

            mapping = {
                "RESNAME": {
                    "BEAD1": {
                        "type": "BEADTYPE",
                        "charge": 0,
                        "atoms": ["ATOM1", "ATOM2", ...],
                    },
                    "BEAD2": {
                        "type": "BEADTYPE",
                        "charge": 0,
                        "atoms": ["ATOM3", "ATOM4", ...],
                    },
                }
            }

        The top-level keys correspond to residue names present in the atomistic
        system. Each bead entry is keyed by the desired CG bead name and must
        contain an ``atoms`` field listing the atom names contributing to that
        bead. Optional ``type`` and ``charge`` fields may be provided; the
        ``type`` field is used to infer bead size class (R/S/T/U) for
        type-weighted mismatch reporting.
    outname : str, optional
        Base name of the output files. Default is ``"cg_mapped"``.
    outdir : str or Path, optional
        Directory where the mapped structure and trajectory will be written.
        Created if it does not exist. Default is ``"."``.
    report : bool, optional
        If ``True`` (default), write a mapping quality report to
        ``{outdir}/{outname}_mapping.log`` after the trajectory is written.
        The report lists for each residue: the number of heavy atoms, the
        number and type breakdown of beads produced, and the bead-to-atom
        mismatch evaluated against the Martini 3 tolerance (±1 per 10 heavy
        atoms). When bead ``type`` fields are present the mismatch is
        type-weighted (regular=4, small=3, tiny=2, virtual=0); otherwise an
        unweighted 4:1 ratio is assumed.
    verbose : bool, optional
        If ``True`` (default), print the mapping report to the terminal in
        addition to writing it to the log file. Has no effect if ``report``
        is ``False``.

    Returns
    -------
    None

    Notes
    -----
    - Bead coordinates are computed as the center of geometry (COG) of the
      atoms assigned to each bead.
    - Atom names may appear multiple times in a bead definition to apply
      weighting when computing the bead center.
    - The output trajectory contains one particle per bead in the order defined
      by the mapping dictionary.
    - The output structure corresponds to the first frame of the mapped
      trajectory.
    - Atom selections that do not match any atoms will raise a ``ValueError``.
    """

    gro = Path(gro).resolve()
    xtc = Path(xtc).resolve()
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Element information is missing, elements attribute will not be populated.*",
            category=UserWarning,
            module=r"MDAnalysis\.topology\.PDBParser",
        )
        u = md.Universe(gro, xtc)

    # Build bead AtomGroups in the exact order we want
    bead_agg = []
    out_resids = []
    out_resnames = []
    out_atomnames = []

    for resname, beads in mapping.items():
        for res in u.select_atoms(f"resname {resname}").residues:
            atoms = res.atoms
            for bead_name, bead_def in beads.items():
                agg_whole = u.select_atoms("name empty")
                for atom in bead_def["atoms"]:
                    agg = atoms.select_atoms(f"name {atom}")
                    if len(agg) == 0:
                        raise ValueError(
                            f"Atom '{atom}' not found in resname "
                            f"'{resname}' (resid {res.resid})"
                        )
                    agg_whole += agg

                bead_agg.append(agg_whole)
                out_resids.append(res.resid)
                out_resnames.append(res.resname)
                out_atomnames.append(bead_name)

    n_beads = len(bead_agg)
    if n_beads == 0:
        raise ValueError(
            "No beads were generated. Check the mapping and residue names."
        )

    # Create an empty CG Universe with n_beads atoms
    cg = md.Universe.empty(
        n_atoms=n_beads,
        n_residues=n_beads,
        atom_resindex=np.arange(n_beads),
        trajectory=True,
    )

    cg.add_TopologyAttr("name", out_atomnames)
    cg.add_TopologyAttr("resname", out_resnames)
    cg.add_TopologyAttr("resid", out_resids)

    coords = np.zeros((n_beads, 3), dtype=np.float32)
    out_gro = outdir / f"{outname}.gro"
    out_xtc = outdir / f"{outname}.xtc"

    wrote_gro = False
    with md.Writer(out_xtc.as_posix(), n_beads) as W:
        for ts in tqdm(u.trajectory, desc="Mapping the trajectory"):
            for k, ag in enumerate(bead_agg):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*duplicates.*")
                    coords[k] = ag.center_of_geometry()
            cg.atoms.positions = coords
            W.write(cg.atoms)
            if not wrote_gro:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"missing dimension - setting unit cell to zeroed box.*",
                        category=UserWarning,
                    )
                    cg.atoms.write(out_gro.as_posix())
                wrote_gro = True

    if report:
        _write_mapping_report(u, mapping, outdir, outname, verbose=verbose)
