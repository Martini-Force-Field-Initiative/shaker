
_bead_sizes_dict = {'R':0.264,
                   'S':0.230,
                   'T':0.191}

_bead_masses_dict = {'R': 72.,
                   'S': 54.,
                   'T': 36.}

def _size_from_name(bead_types):
    '''
    Function to convert list of bead types into a list of bead sizes (R,S,T).
    '''
    extra_list = []
    for string in bead_types:
        if string[0].upper() == 'S':  
            extra_list.append('S')
        elif string[0].upper() == 'T': 
            extra_list.append('T')
        else:  
            extra_list.append('R')
    return extra_list

def _bead_mass_from_type(bead_type):
    '''
    Function to convertbead types into a mass value.
    '''
    if bead_type.startswith("S"):
        return _bead_masses_dict["S"]
    if bead_type.startswith("T"):
        return _bead_masses_dict["T"]
    return _bead_masses_dict["R"]