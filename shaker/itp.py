from pathlib import Path

def write_initial_CGitp (resname, filename="initial_CG.itp",
                         bead_names, bead_types, bead_charges, 
                         header=''):
    
    """
    Write an initial CG topology file (.itp).

    This function generates a minimal Martini-compatible `.itp` file
    describing a single CG molecule. The topology contains a `[ moleculetype ]`
    section and an `[ atoms ]` section populated from the provided bead
    definitions. Empty `[ bonds ]` and `[ angles ]` sections are included as
    placeholders for subsequent parameterization.

    Parameters
    ----------
    resname : str
        Residue name of the CG molecule.
    bead_names : sequence of str
        Names of the CG beads in the order they appear in the topology.
    bead_types : sequence of str
        Martini bead types corresponding to each bead.
    bead_charges : sequence of float
        Partial charges assigned to each bead.
    header : str, optional
        Optional text written at the top of the `.itp` file (e.g. comments,
        metadata, or parameterization notes).

    """

    ## Run some quick checks.
    if not (len(bead_names) == len(bead_types) == len(bead_charges)):
        raise ValueError("bead_names, bead_types and bead_charges must have the same length")
    filename = Path(filename).resolve()

    ## Write the file
    with open(filename, "w") as f:
        f.write(f"{header}\n\n")
        f.write("[ moleculetype ]\n")
        f.write(f"{resname}  1\n\n")

        f.write("[ atoms ]\n")
        f.write("; nr type resnr residue atom cgnr charge mass\n")

        for i, (name, typ, charge) in enumerate(zip(bead_names, bead_types, bead_charges), 1):
            f.write(f"{i:4} {typ:4} {0:4} {resname:4} {name:4} {i:4} {charge:6}\n")

        f.write("\n[ bonds ]\n")
        f.write("; i  j  funct length\n\n")

        f.write("[ angles ]\n")
        f.write("; i  j  k  funct  angle  force.c.\n\n")