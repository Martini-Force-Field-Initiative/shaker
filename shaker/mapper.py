import MDAnalysis as md
from pathlib import Path
import numpy as np
from tqdm.autonotebook import tqdm
'''
Functions and tools to process and map AA/QM trajectories to CG.
'''

def map_aa2cg(gro, xtc, mapping,
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
        bead. Optional ``type`` and ``charge`` fields may also be provided for
        use in topology generation or bookkeeping.
    outname : str, optional
        Base name of the output files. Default is "cg_mapped".
    outdir : str or Path, optional
        Directory where the mapped structure and trajectory will be written.
        Created if it does not exist.

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

    for resname, beads in mapping.items():
        for res in u.select_atoms(f"resname {resname}").residues:
            atoms = res.atoms
            for bead_name, bead_def in beads.items():
                agg_whole = u.select_atoms("name empty")
                for atom in bead_def["atoms"]:
                    agg = atoms.select_atoms(f"name {atom}")
                    if len(agg) == 0:
                        raise ValueError(f"Atom '{atom}' not found in resname '{resname}' (resid {res.resid})")
                    agg_whole += agg
    
                bead_agg.append(agg_whole)
                out_resids.append(res.resid)
                out_resnames.append(res.resname)
                out_atomnames.append(bead_name)
                
    n_beads = len(bead_agg)
    if n_beads == 0:
        raise ValueError("No beads were generated. Check the mapping and residue names.")
        
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
        for ts in tqdm(u.trajectory, desc="Mapping the trajectory"):
            for k, ag in enumerate(bead_agg):
                ## we could add a com flag here.
                coords[k] = ag.center_of_geometry()   
            cg.atoms.positions = coords
            W.write(cg.atoms)
            if not wrote_gro:              # write gro for first frame.
                cg.atoms.write(out_gro.as_posix())
                wrote_gro = True