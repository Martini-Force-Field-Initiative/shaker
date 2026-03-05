import MDAnalysis as md
import subprocess
from importlib.resources import files
import numpy as np
import glob
import os
from pathlib import Path
import shlex

def prepare_setup_water(initial_structure, structure_itp='initial_CG.itp',
                        n_mols=1, box_size=4, 
                        FFitp=None, SolvITP=None, IonsITP=None,
                        NaCL_Conc=0.15, gmx_loc=''):
    '''
    Prepare a solvated Martini 3 simulation system for a coarse-grained molecule.
    
    This function builds a solvent box, inserts `n_mols` copies of the solute,
    solvates with water, neutralizes the system, and adds NaCl to reach
    a target concentration. It writes a minimal `topol.top` and produces GROMACS
    coordinate/tpr suitable for subsequent MD.
    
    Parameters
    ----------
    initial_structure : str or Path
        Solute structure file (typically a CG .gro or .pdb) used as the template
        for insertion.
    structure_itp : str, optional
        ITP filename (or path) describing the solute topology included in `topol.top`.
        Default is "initial_CG.itp".
    n_mols : int, optional
        Number of solute copies to insert into the simulation box. Default is 1.
    box_size : float, optional
        Cubic box edge length in nm passed to `gmx insert-molecules -box`.
        Default is 4.
    FFitp : str or Path, optional
        Martini force-field ITP file. If None, the SHAKER-distributed default is used.
    SolvITP : str or Path, optional
        Martini solvent ITP file. If None, the SHAKER-distributed default is used.
    IonsITP : str or Path, optional
        Martini ions ITP file. If None, the SHAKER-distributed default is used.
    NaCL_Conc : float, optional
        Target NaCl concentration (mol/L). Default is 0.15.
    gmx_loc : str, optional
        Prefix/path to the GROMACS executable (e.g. "/usr/local/bin/" or "").
    
    Notes
    -----
    The setup procedure performs the following steps:
    
    1. Generate a topology file.
    2. Insert `n_mols` copies of `initial_structure` into a cubic box (`gmx insert-molecules`).
    3. Solvate the box with Martini water using ``gmx solvate``.
    4. Neutralize the system with counterions using ``gmx genion``.
    5. Add additional NaCl to reach the target concentration.
    
    All GROMACS stdout/stderr are appended to `gmx_setup_water.log`.

    Wishlist
    --------
    - Support for multiple solvent models.
    '''
    
    ## normalize paths
    initial_structure = Path(initial_structure).resolve()
    structure_itp     = Path(structure_itp).resolve()
    gmx = str(Path(gmx_loc) / "gmx") if gmx_loc else "gmx"
    
    ## Pick itps and mdps.
    if FFitp is None: FFitp = files("shaker.data.itps") / "martini_v3.0.0.itp"
    if SolvITP is None: SolvITP = files("shaker.data.itps") / "martini_v3.0.0_solvents_v1.itp"
    if IonsITP is None: IonsITP = files("shaker.data.itps") / "martini_v3.0.0_ions_v1.itp"

    minMDP = files("shaker.data.mdps") / "min.mdp"
    waterbox = files("shaker.data.itps") / "water.gro"

    ## Prepare top file
    resname = np.unique(md.Universe(initial_structure).residues.resnames)[0]
    
    topinput = open('topol.top', 'w')
    topinput.write(f'#include "{FFitp}"\n')
    topinput.write(f'#include "{structure_itp}"\n')
    topinput.write(f'#include "{SolvITP}"\n')
    topinput.write(f'#include "{IonsITP}"\n')
    topinput.write('[system]\n')
    topinput.write('Shaker approved!\n')
    topinput.write("\n")
    topinput.write("[ molecules ]\n")
    topinput.write(f'{resname}   {n_mols}\n')    
    topinput.close()

    ## Run gmx
    env = os.environ.copy()
    ## Get rid of the old runs when rerunning.
    env["GMX_MAXBACKUP"] = "-1"
    with open("gmx_setup_water.log", "w") as log:
        ## Create box with molecules
        _run([gmx, "insert-molecules",
            "-ci", str(initial_structure),
            "-box", str(box_size), str(box_size), str(box_size),
            "-nmol", str(n_mols),
            "-o", "box.gro"],
            log=log, env=env)

        ## Solvate the box
        _run([gmx, "solvate",
            "-cp", "box.gro", "-cs", str(waterbox),
            "-o", "watered.gro", "-p", "topol.top"],
            log=log, env=env)

        ## Neutralize
        _run([gmx, "grompp",
            "-f", str(minMDP), "-c", "watered.gro",
            "-p", "topol.top", "-o", "memion.tpr",
            "-maxwarn", "1"],
            log=log, env=env)

        _run([gmx, "genion",
              "-s", "memion.tpr", "-o", "memion.gro",
              "-p", "topol.top",
              "-pname", "NA", "-pq", "+1",
              "-nname", "CL", "-nq", "-1",
              "-neutral"],
             log=log, env=env, input_text="W\n")

        ## Add NaCl
        Waternumber = len(md.Universe('memion.gro').select_atoms('resname W').residues)
        naclNUM = int((NaCL_Conc * Waternumber * 4) / 55.5)

        _run([gmx, "grompp",
            "-f", str(minMDP), "-c", "memion.gro",
            "-p", "topol.top", "-o", "memion_2.tpr",
            "-maxwarn", "4"],
            log=log, env=env)

        _run([gmx, "genion",
            "-s", "memion_2.tpr", "-o", "memion_2.gro",
            "-p", "topol.top", 
            "-pname", "NA", "-pq", "+1",
            "-nname", "CL", "-nq", "-1",
            "-np", str(naclNUM), "-nn", str(naclNUM)],
            log=log, env=env, input_text="W\n")

    
def runSim (minMDP=None, relMDP=None, prodMDP=None,
            minOPT='-pin on -nt 8', 
            relOPT='-pin on -nt 8', 
            prodOPT='-pin on -nt 8', 
            maxwarn=1, gmx_loc='',
            cleanTraj=True, cleanOldRun=True,):
    '''
    Run a standard Martini simulation pipeline (minimization → relaxation → production).

    By default, the MDP files distributed with SHAKER are used.

    Parameters
    ----------
    minMDP, relMDP, prodMDP : str or Path, optional
        MDP parameter files for minimization, relaxation, and production MD.
        If None, the defaults distributed with SHAKER are used.
    minOPT, relOPT, prodOPT : str, optional
        Extra command-line arguments passed to `gmx mdrun` for each stage
        (e.g. "-pin on -nt 8").
    maxwarn : int, optional
        Value passed to `gmx grompp -maxwarn`. Default is 1.
    gmx_loc : str, optional
        Prefix/path to the GROMACS executable (e.g. "/usr/local/bin/" or "").
    cleanTraj : bool, optional
        If True, post-process the production trajectory using `_traj_cleanup`.
    cleanOldRun : bool, optional
        If True, remove files from previous runs using `_clean_old_files()`.

    Notes
    -----
    The function assumes the following files exist in the working directory:

    - `memion_2.gro` : starting coordinates
    - `topol.top`    : system topology

    These are the final outputs of `prepare_setup_water`.
    
    Output files use the prefixes `m`, `r`, and `p` corresponding to
    minimization, relaxation, and production stages.
    '''

    if minMDP is None: minMDP = files("shaker.data.mdps") / "min.mdp"
    if relMDP is None: relMDP = files("shaker.data.mdps") / "rel.mdp"
    if prodMDP is None: prodMDP = files("shaker.data.mdps") / "prod.mdp"

    if cleanOldRun:
        _clean_old_files()

    gmx = str(Path(gmx_loc) / "gmx") if gmx_loc else "gmx"
    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"
    
    ### Minimize the system
    _run([gmx, "grompp",
        "-f", str(minMDP), "-c", "memion_2.gro",
        "-p", "topol.top", "-o", "m.tpr",
        "-maxwarn", str(maxwarn)], env=env)
    _run([gmx, "mdrun", "-v", "-deffnm", "m",
        *shlex.split(minOPT)], env=env)
    
    ### Relax the system
    _run([gmx, "grompp",
        "-f", str(relMDP), "-c", "m.gro",
        "-p", "topol.top", "-o", "r.tpr",
        "-maxwarn", str(maxwarn)], env=env)
    _run([gmx, "mdrun", "-v", "-deffnm", "r",
        *shlex.split(relOPT)], env=env)
    
    ### Production Sim
    _run([gmx, "grompp",
        "-f", str(prodMDP), "-c", "r.gro",
        "-p", "topol.top", "-o", "p.tpr",
        "-maxwarn", str(maxwarn)], env=env)
    _run([gmx, "mdrun", "-v", "-deffnm", "p",
        *shlex.split(prodOPT)], env=env)

    if cleanTraj:
        _traj_cleanup('p.gro', 'p.xtc', 'p.tpr')


def _run(cmd, *, log=None, env=None, input_text=None, cwd=None):
    '''
    Short helper to assist when using subprocess to run gmx.
    '''
    subprocess.run(cmd, input=input_text,
        text=True if input_text is not None else False,
        stdout=log, stderr=subprocess.STDOUT, #if log is None goes to term.
        env=env, cwd=cwd, check=True,)

def _traj_cleanup(gro, xtc, tpr,
                 outname='pbc', gmx_loc=''):
    '''
    Remove water and fix PBC after running the production.
    '''
    u=md.Universe(gro)
    clean = u.select_atoms('not resname W ION NA CL')
    clean.write("index_clean.ndx", mode="w", name= 'clean')
    sel_in = "clean\n"

    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"
    gmx = str(Path(gmx_loc) / "gmx") if gmx_loc else "gmx"

    with open("gmx_traj_cleanup.log", "w") as log:
        # PBC-fixed trajectory
        _run([gmx, "trjconv",
            "-f", str(xtc), "-s", str(tpr),
            "-o", f"{outname}.xtc",
            "-pbc", "mol", "-n", "index_clean.ndx"],
            log=log, env=env, input_text=sel_in)
        
        # First frame PDB w/ connects
        _run([gmx, "trjconv",
            "-f", str(xtc), "-s", str(tpr),
            "-o", f"{outname}.pdb", "-pbc", "mol",
            "-b", "0", "-e", "0", "-conect",
            "-n", "index_clean.ndx"],
            log=log, env=env, input_text=sel_in)

    # Remove ENDMDL
    pdb_path = Path(f"{outname}.pdb")
    lines = pdb_path.read_text().splitlines(True)
    pdb_path.write_text("".join(l for l in lines if "ENDMDL" not in l))
    
    
def _clean_old_files():
    '''
    Small function to clean old simulation files from previous iterations.
    '''
    for pattern in ("m.*", "r.*", "p.*", "pbc.*", "step*", "crash*"):
        for f in glob.glob(pattern):
            if os.path.isfile(f):
                os.remove(f)