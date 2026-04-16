"""Utility helper functions used across SHAKER modules."""

_bead_sizes_dict = {'R':0.264,
                   'S':0.230,
                   'T':0.191}

_bead_masses_dict = {'R': 72.,
                   'S': 54.,
                   'T': 36.}

_BEAD_RATIOS = {"regular": 4,
                "small":   3,
                "tiny":    2,}

_size_label = {"R": "regular", "S": "small", "T": "tiny", "U": "virtual"}

def _size_from_name(bead_types):
    '''
    Convert bead types into size classes (R, S, T, U).

    U is treated as a special case (radius = 0).
    '''
    sizes = []
    for string in bead_types:
        t = string[0].upper()
        if t == 'U':
            sizes.append('U')
        elif t == 'S':
            sizes.append('S')
        elif t == 'T':
            sizes.append('T')
        else:
            sizes.append('R')
    return sizes

def _bead_mass_from_type(bead_type):
    '''
    Function to convertbead types into a mass value.
    '''
    if bead_type.startswith("S"):
        return _bead_masses_dict["S"]
    if bead_type.startswith("T"):
        return _bead_masses_dict["T"]
    return _bead_masses_dict["R"]