import MDAnalysis as md
import subprocess
from importlib.resources import files
import numpy as np
import glob
import os
from pathlib import Path

def prepare_setup_water(initial_structure,
                        box_size=4, box_type='dodecahedron',
                        FFitp=None, SolvITP=None, IonsITP=None,
                        NaCL_Conc=0.15, gmx_loc=''):
    '''
    Function to setup a simulation of the target molecule inside
    a water box.

    WISHLIST: 
    - Allow multiple molecules per simbox.
    - Shoot output to a log file?
    - Allow multiple solvents.
    '''
    ## normalize paths
    initial_structure = Path(initial_structure).resolve()
    
    if FFitp is None: FFitp = files("shaker.data.itps") / "martini_v3.0.0.itp"
    if SolvITP is None: SolvITP = files("shaker.data.itps") / "martini_v3.0.0_solvents_v1.itp"
    if IonsITP is None: IonsITP = files("shaker.data.itps") / "martini_v3.0.0_ions_v1.itp"

    minMDP = files("shaker.data.mdps") / "min.mdp"
    waterbox = files("shaker.data.itps") / "water.gro"

    resname = np.unique(md.Universe(initial_structure).residues.resnames)[0]
    
    topinput = open('topol.top', 'w')
    topinput.write(f'#include "{FFitp}"\n')
    topinput.write('#include "initial_CG.itp"\n')
    topinput.write(f'#include "{SolvITP}"\n')
    topinput.write(f'#include "{IonsITP}"\n')
    topinput.write('[system]\n')
    topinput.write('one molecule\n')
    topinput.write("\n")
    topinput.write("[ molecules ]\n")
    topinput.write(f'{resname}   1\n')    
    topinput.close()
        
    subprocess.call(f'{gmx_loc}gmx editconf -f {initial_structure} -o box.gro -box {box_size} {box_size} {box_size} -bt {box_type}'
                , shell=True)
    subprocess.call(f'{gmx_loc}gmx solvate -cp box.gro -cs {waterbox} -o watered.gro -p topol.top'
                , shell=True)

    u = md.Universe('watered.gro')
    Waternumber = len(u.select_atoms('resname W PW').residues)
    subprocess.call(f'{gmx_loc}gmx grompp -f {minMDP} -c watered.gro -p topol.top -o memion.tpr -maxwarn 1'
                , shell=True)
    p = subprocess.Popen(f'{gmx_loc}gmx genion -s memion.tpr -o memion.gro -p topol.top -pname NA -pq +1 -nname CL -nq -1 -neutral'
                         , stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    p.communicate('W')
    p.wait()
   
    naclNUM = int((NaCL_Conc * Waternumber*4)/55.5)
    subprocess.call(f'{gmx_loc}gmx grompp -f {minMDP} -c memion.gro -p topol.top -o memion_2.tpr -maxwarn 4'
                , shell=True)
    p = subprocess.Popen(f'{gmx_loc}gmx genion -s memion_2.tpr -o memion_2.gro -p topol.top -pname NA -pq +1 -nname CL -nq -1 -np {naclNUM} -nn {naclNUM}'
                         , stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    p.communicate('W')
    p.wait()


def traj_cleanup(gro, xtc, tpr,
                 outname='pbc', gmx_loc=''):
    '''
    Remove water and fix PBC after running the production.
    '''
    u=md.Universe(gro)
    clean = u.select_atoms('not resname W ION NA CL SOL SOD CLO')
    clean.write("index_clean.ndx", mode="w", name= 'clean')

    p = subprocess.Popen(f'{gmx_loc}gmx trjconv -f {xtc} -s {tpr} -o {outname}.xtc -pbc mol -n index_clean.ndx'
                         , stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    p.communicate('clean')
    p.wait()

    p = subprocess.Popen(f'{gmx_loc}gmx trjconv -f {xtc} -s {tpr} -o {outname}.pdb -pbc mol -b 0 -e 0 -conect -n index_clean.ndx'
                         , stdin=subprocess.PIPE, shell=True, universal_newlines=True)
    p.communicate('clean')
    p.wait()
    subprocess.call(f'sed "/ENDMDL/d" -i {outname}.pdb'
                    , shell = True)
    
def clean_old_files():
    '''
    Small function to clean old simulation files from previous iterations.
    '''
    for pattern in ("m.*", "r.*", "p.*", "pbc.*", "step*", "crash*"):
        for f in glob.glob(pattern):
            if os.path.isfile(f):
                os.remove(f)

def runSim (minMDP=None, relMDP=None, prodMDP=None,
            minOPT='-pin on -nt 8', 
            relOPT='-pin on -nt 8', 
            prodOPT='-pin on -nt 8', 
            maxwarn=1, gmx_loc='',
            cleanTraj=True, cleanOldRun=True,
            ):
    '''
    Re-runs min, and short production run. By default it uses the mdp files
    distributed with the package.
    '''

    if minMDP is None: minMDP = files("shaker.data.mdps") / "min.mdp"
    if relMDP is None: relMDP = files("shaker.data.mdps") / "rel.mdp"
    if prodMDP is None: prodMDP = files("shaker.data.mdps") / "prod.mdp"

    if cleanOldRun:
        clean_old_files()
        
    ### Minimize the system
    subprocess.call(f"{gmx_loc}gmx grompp -f {minMDP} -c memion_2.gro -p topol.top -o m.tpr -maxwarn {maxwarn}", shell = True)
    subprocess.call(f"{gmx_loc}gmx mdrun -v -deffnm m {minOPT}", shell = True)
    
    ### Relax the system
    subprocess.call(f"{gmx_loc}gmx grompp -f {relMDP} -c m.gro -p topol.top -o r.tpr -maxwarn {maxwarn}", shell = True)
    subprocess.call(f"{gmx_loc}gmx mdrun -v -deffnm r {relOPT}", shell = True)
    
    ### Production Sim
    subprocess.call(f"{gmx_loc}gmx grompp -f {prodMDP} -c r.gro -p topol.top -o p.tpr -maxwarn {maxwarn}", shell = True)
    subprocess.call(f"{gmx_loc}gmx mdrun -v -deffnm p {prodOPT}", shell = True)

    if cleanTraj:
        traj_cleanup('p.gro', 'p.xtc', 'p.tpr')
