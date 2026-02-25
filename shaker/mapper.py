import MDAnalysis as md
from importlib.resources import files
import os
import subprocess

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
                    
def map_aa2cg (gro, xtc, resnames,
               bead_assignments, bead_names,
               gmx_loc=''):
    '''Provide the universe along with the lists containing the resnames of eg the lipids,
    the bead_assignements (which atoms goes into which bead), and a list of the bead_names'''

    # Initializing the universe
    u = md.Universe(gro,xtc)
    
    # Generating the mapping index files, for each bead in each residue.
    # Not the fastest but allows differential weighing of beads.
    bead_agg = []
    for i, molecule in enumerate(resnames):
        targets = u.select_atoms(f'resname {molecule}').residues
        for j in range(len(targets)):
            target = targets[j].atoms
            for bead in bead_assignments[i]:
                agg_whole = u.select_atoms('name empty')
                for atom in bead:
                    agg = target.select_atoms(f'name {atom}')
                    if len(agg) == 0:
                        raise ValueError(f"Atom '{atom}' not found in resname '{molecule}' "
                                         f"(resid {targets[j].resid})")
                    agg_whole += agg
                bead_agg.append(agg_whole)
    with md.selections.gromacs.SelectionWriter('MappingIndex.ndx', mode='w') as ndx:
        for idx, agg in enumerate(bead_agg):
            ndx.write(agg)

    ## Create fake AA structure where all atoms are hydrogens.
    ## Turns Gromacs' c.o.m. into c.o.g.
    for atom in u.atoms:
        atom.name = 'H' ## WARNING. After this point any name based selection will be fucked.
    u.atoms.write('all_hydrogen.gro')
    
    #Using gromacs and the fancy index we wrote map the AA simulation into CG
    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"   # disable #file.1# backups
    
    groups_in = "\n".join(map(str, range(len(bead_agg)))) + "\n"
    cmd_gro = [f"{gmx_loc}gmx", "traj",
               "-f", gro, "-s", "all_hydrogen.gro",
               "-n", "MappingIndex.ndx", "-com",
               "-ng", str(len(bead_agg)),
               "-oxt", "cg_mapped.gro",]
    
    cmd_xtc = [f"{gmx_loc}gmx", "traj",
               "-f", xtc, "-s", "all_hydrogen.gro",
               "-n", "MappingIndex.ndx", "-com",
               "-ng", str(len(bead_agg)),
               "-oxt", "cg_mapped.xtc",]
    
    with open("gmx_mapping.log", "w") as log:
        subprocess.run(cmd_gro, input=groups_in,
                       text=True, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True,)
    
        subprocess.run(cmd_xtc, input=groups_in,
                       text=True, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True,)

    ### Fix names in GRO for easier parameterization
    names = []
    for i, molecule in enumerate(resnames):
        targets = u.select_atoms('resname {}'.format(molecule)).residues
        gro_names    = bead_names[i]*len(targets)
        names += gro_names

    with open('cg_mapped.gro') as cggro_in:
        gro_file = cggro_in.readlines()
    for idx, line in enumerate(gro_file[2:-1]):
        gro_file[2+idx]=line.replace(line[12:15],names[idx])
    with open('cg_mapped.gro', 'w+') as file_out:
        for line in gro_file:
            file_out.write(line)
