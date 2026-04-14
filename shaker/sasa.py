import MDAnalysis as md
from importlib.resources import files
import os 
import subprocess
from .helper import _bead_sizes_dict, _size_from_name
from pathlib import Path

'''
Functions, tools and workflows to calculate SASA & Connely surfaces.
Mostly wrappers for GROMACS' `gmx sasa` tool.
'''


def run_SASA(name, 
             gro, xtc, 
             resname, 
             isCG=False, mapping=None, 
             dir_out='.', 
             selection='all',
             gmx_loc=''):
    
    '''
    This is a wrapper around `gmx sasa` that simplifies SASA analysis for
    both atomistic and coarse-grained systems. For coarse-grained systems,
    a custom van der Waals radii file is generated from the supplied bead
    names and bead types.
    
    Parameters
    ----------
    name : str
            Name/handle for this analysis. Used to create the output directory
            `dir_out/SASA/<name>/`.
    gro : str
        Directory to .gro structure file for analysis.
    xtc : str
        Directory to .xtc structure file for analysis.
    resname : str
        Resname of target molecule to analyse.
    isCG : bool, optional
        If True, prepare a CG-specific van der Waals radii file. Requires
        `bead_names` and `bead_types`. Default is False.
    mapping : dict, optional
        Mapping dictionary in SHAKER format. Required when `isCG=True`.
        The bead names and bead types are extracted from `mapping[resname]`
        to generate the CG van der Waals radii file.
    selection : str, optional
        Additional atom selection applied within the first residue matching
        `resname`. Can be used to exclude parts of the molecule from analysis.
        Default is "all".
    gmx_loc : str, optional
        Prefix/path to the GROMACS executable directory.
    
    Notes
    -----
    - This function currently analyzes only the first residue matching `resname`.
    '''
    ## normalize paths
    gro = Path(gro).resolve()
    xtc = Path(xtc).resolve()
    dir_out = Path(dir_out).resolve()
    
    ## Directory handling.
    dir_writing = f'{dir_out}/SASA/{name}'
    os.makedirs(dir_writing, exist_ok=True)

    ## Create index file and spit out a gro.
    u = md.Universe(gro, xtc)
    tgt = u.select_atoms(f'resname {resname}').residues[0].atoms.select_atoms(selection)
    tgt.write(f"{dir_writing}/index.ndx", mode="w", name= 'TGT')
    tgt.atoms.write(f"{dir_writing}/gro.gro")

    ## Prepare vdw radii file.
    if isCG:
        if mapping is None:
            raise ValueError("When isCG=True, a mapping dictionary must be provided.")
        if resname not in mapping:
            raise ValueError(f"No mapping found for resname '{resname}'")
    
        bead_names = list(mapping[resname].keys())
        bead_types = [bead["type"] for bead in mapping[resname].values()]
        bead_sizes = _size_from_name(bead_types)
        _write_cg_vdw(dir_writing, bead_names, bead_sizes)
    else: # Most likely AA.
        vdwloc = files("shaker.data.vdw") / "vdwradii_AA.dat"
        subprocess.call(f'cp {vdwloc} {dir_writing}/vdwradii.dat'
                    , shell=True)

    ## Calculate SASA & connoly surface
    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"   # disable #file.1# backups

    cmd1 = [f"{gmx_loc}gmx", "sasa",
            "-f", xtc, "-s", gro,
            "-n", "index.ndx",
            "-ndots", "4800", "-probe", "0.191",
            "-or", "resarea_SASA.xvg",
            "-o", "SASA.xvg",
            "-tv", "vol.xvg",]

    cmd2 = [f"{gmx_loc}gmx", "sasa",
            "-s", "gro.gro",
            "-o", "temp.xvg",
            "-probe", "0.191",
            "-ndots", "240",
            "-q", "surface.pdb",]

    logfile = f"{dir_writing}/gmx_sasa.log"
    with open(logfile, "w") as log:
        subprocess.run(cmd1, input="TGT\n", cwd=dir_writing,
                       stdout=log, stderr=subprocess.STDOUT,
                       env=env, text=True, check=True)

        subprocess.run(cmd2, input="System\n", cwd=dir_writing,
                       stdout=log, stderr=subprocess.STDOUT,
                       env=env, text=True, check=True)


def _write_cg_vdw (dir_out, bead_names, bead_sizes):
    '''
    Write a CG vdwradii.dat file for SASA calculations.

    Bead sizes are mapped using `_bead_sizes_dict`. If a bead type is "U",
    its radius is set to 0.
    '''
    out = Path(dir_out) / "vdwradii.dat"
    with open(out, "w") as sasa:
        sasa.write("; CG van der Waals radii :)\n")
        for bead, btype in zip(bead_names, bead_sizes):
            if btype == "U":
                size = 0.0
            else:
                if btype not in _bead_sizes_dict:
                    raise ValueError(f"Unknown bead size type '{btype}'")
                size = _bead_sizes_dict[btype]

            sasa.write(f"???  {bead:4}  {size:.3f}\n")
