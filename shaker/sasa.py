import MDAnalysis as md
from importlib.resources import files
import os 
import subprocess
from .helper import bead_sizes_dict
from .mapper import size_from_name
'''
Functions, tools and workflows to calculate SASA & Connely surfaces.
Mostly wrappers for GROMACS' `gmx sasa` tool.
'''


def run_SASA(name, 
             gro, xtc, 
             resname, 
             isCG=False, bead_names=None, bead_types=None, 
             dir_out='.', 
             selection='all',
             gmx_loc=''):
    
    '''
    Wrapper for gmx SASA. Simplifies handling of CG sims and vdw radii files.
        ----------
    name : str
        Handle for particular analysis.
    gro : str
        Directory to .gro structure file for analysis.
    xtc : str
        Directory to .xtc structure file for analysis.
    resname : str
        Resname of target molecule to analyse.
    isCG : Bool
        True if analysing CG molecule. Requires `bead_names` and `bead_types`.
    bead_names : list
        List of CG bead names for the target molecule.
    bead_types : list
        List of CG bead types for the target molecule. Used to define bead size.
    selection : str
        Selection string for the target molecule. Used to exclude parts of the 
        molecule from analysis.
    gmx_loc : str
        Directory to gmx compilation.
    '''

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
        if bead_names is None or bead_types is None:
            raise ValueError("When isCG=True, both 'bead_names' and 'bead_types' must be provided.")
        bead_sizes = size_from_name(bead_types)
        write_cg_vdw(dir_writing, bead_names, bead_sizes)
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


def write_cg_vdw (dir_out, bead_names, bead_sizes):
    '''fuction to write the CG vdwradii.dat file for calculating SASA.'''
    
    sasa = open(dir_out+'/vdwradii.dat', 'w')
    sasa.write(';CG van der walls radii\n')
    for idx, bead in enumerate(bead_names):
        size = bead_sizes_dict[bead_sizes[idx]]
        sasa.write('???  {}    {}\n'.format(bead, size))
    sasa.close()
