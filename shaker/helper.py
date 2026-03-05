
bead_sizes_dict = {'R':0.264,
                   'S':0.230,
                   'T':0.191}

bead_masses_dict = {'R': 72.,
                   'S': 54.,
                   'T': 36.}

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