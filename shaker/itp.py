

def write_initial_CGitp (resname, bead_names, bead_types, bead_charges):
    '''Function to write the inital CG itp file.'''
    
    initial = open('initial_CG.itp', 'w')
    initial.write('[ moleculetype ]\n')
    initial.write('{}  1'.format(resname))
    initial.write('\n')
    initial.write('[ atoms ]\n')
    initial.write('; nr type resnr residue atom cgnr charge mass\n')
    for idx, bead in enumerate(bead_names):
        initial.write("{:4}    {:4}    {:4}    {:4}    {:4}    {:4}    {:4}    \n"
                      .format(idx+1, bead_types[idx], 0, resname, bead, idx+1, bead_charges[idx]))
    initial.write('\n')
    initial.write('[ bonds ]\n')
    initial.write('; i  j  funct length\n')
    initial.write('\n')
    initial.write('[ angles ]\n')
    initial.write(';  i  j  k   funct   angle   force.c.\n')
    initial.write('\n')
    initial.close()
