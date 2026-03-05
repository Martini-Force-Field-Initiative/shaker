import MDAnalysis as md
from pathlib import Path
import numpy as np
from tqdm import tqdm
'''
Functions and tools to process and map AA/QM trajectories to CG.
'''
    
def size_from_name(bead_types):
    '''
    Function to convert list of bead types into a list of bead sizes (R,S,T).
    '''
    # Initialize the extra list
    extra_list = []
    # Iterate through the strings and add the corresponding letter to the extra list
    for string in bead_types:
        if string[0].upper() == 'S':  # Check if the first letter is S
            extra_list.append('S')
        elif string[0].upper() == 'T':  # Check if the first letter is T
            extra_list.append('T')
        else:  # For all other cases
            extra_list.append('R')
    return extra_list


def map_aa2cg(gro, xtc, resnames,
             bead_assignments, bead_names,
             outname="cg_mapped", outdir="."):
    
    '''
    Map an atomistic trajectory to a coarse-grained representation.

    This function constructs coarse-grained beads from atomistic coordinates by
    computing the center of geometry of user-defined groups of atoms for each residue. 
    The mapping is applied to every frame of the input trajectory, producing a 
    coarse-grained trajectory and a representative structure.

    Parameters
    ----------
    gro : str or Path
        Atomistic structure file (.gro or .pdb) used to initialize the system.
    xtc : str or Path
        Atomistic trajectory file (.xtc or .trr) containing the coordinates
        to be mapped.
    resnames : list of str
        Residue names to be mapped. Each entry must correspond to a mapping
        definition in `bead_assignments` and `bead_names`.
    bead_assignments : list of list of list of str
        Atom-to-bead mapping definitions. For each residue type in `resnames`,
        this contains a list of beads, where each bead is defined by a list of
        atom names contributing to that bead.
    bead_names : list of list of str
        Names assigned to the coarse-grained beads for each residue type.
        The number of bead names must match the number of bead definitions
        in `bead_assignments`.
    outname : str, optional
        Base name of the output files. Default is "cg_mapped".
    outdir : str or Path, optional
        Directory where the mapped structure and trajectory will be written.
        Created if it does not exist.

    Notes
    -----
    - Bead coordinates are computed as the center of geometry (COG) of the
      atoms assigned to each bead.
    - The output trajectory contains one particle per bead in the order
      defined by `resnames` and `bead_assignments`.
    - The output structure corresponds to the first frame of the mapped
      trajectory.
    - Atom selections that do not match any atoms will raise a `ValueError`.
    '''
    
    gro = Path(gro).resolve()
    xtc = Path(xtc).resolve()
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    u = md.Universe(gro, xtc)

    # Build bead AtomGroups in the exact order we want 
    bead_agg = []
    out_resids = []
    out_resnames = []
    out_atomnames = []

    for i, resname in enumerate(resnames):
        for res in u.select_atoms(f"resname {resname}").residues:
            atoms = res.atoms
            for bead_idx, bead in enumerate(bead_assignments[i]):
                agg_whole = u.select_atoms("name empty")
                for atom in bead:
                    agg = atoms.select_atoms(f"name {atom}")
                    if len(agg) == 0:
                        raise ValueError(f"Atom '{atom}' not found in resname '{resname}' (resid {res.resid})")
                    agg_whole += agg
                bead_agg.append(agg_whole)

                # metadata for output topology
                out_resids.append(res.resid)
                out_resnames.append(res.resname)
                out_atomnames.append(bead_names[i][bead_idx])

    n_beads = len(bead_agg)

    # Create an empty CG Universe with n_beads atoms
    cg = md.Universe.empty(
        n_atoms=n_beads,
        n_residues=n_beads,          
        atom_resindex=np.arange(n_beads),
        trajectory=True,)

    cg.add_TopologyAttr("name", out_atomnames)
    cg.add_TopologyAttr("resname", out_resnames)
    cg.add_TopologyAttr("resid", out_resids)

    # Fill coordinates frame-by-frame...
    coords = np.zeros((n_beads, 3), dtype=np.float32)

    out_gro = outdir / f"{outname}.gro"
    out_xtc = outdir / f"{outname}.xtc"
    
    wrote_gro = False
    with md.Writer(out_xtc.as_posix(), n_beads) as W:
        for ts in tqdm(u.trajectory):
            for k, ag in enumerate(bead_agg):
                ## we could add a com flag here.
                coords[k] = ag.center_of_geometry()   
            cg.atoms.positions = coords
            W.write(cg.atoms)
            if not wrote_gro:              # write gro for first frame.
                cg.atoms.write(out_gro.as_posix())
                wrote_gro = True