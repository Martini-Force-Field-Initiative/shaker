from pathlib import Path

def write_initial_CGitp (resname, mapping, 
                         filename="initial_CG.itp",
                         header=None, footer=None):    
    """
    Write an initial CG topology file (.itp).

    This function generates a minimal Martini-compatible `.itp` file
    describing a single CG molecule. The topology contains a `[ moleculetype ]`
    section and an `[ atoms ]` section populated from the provided bead
    definitions. 

    Parameters
    ----------
    resname : str
        Residue name of the CG molecule.
    mapping : dict
        Mapping dictionary in SHAKER format. The expected structure is::

            mapping = {
                "RESNAME": {
                    "BEAD1": {"type": "TC1", "charge": 0, "atoms": [...]},
                    "BEAD2": {"atoms": [...]},
                }
            }

        Only `mapping[resname]` is used by this function.
        If a bead does not define a `"type"` entry, the placeholder `"TYPe"`
        is used. If a bead does not define a `"charge"` entry, the value `0`
        is used.
    filename : str or Path, optional
        Output filename of the generated `.itp` topology file.
        The file will be overwritten if it already exists.
        Default is `"initial_CG.itp"`.
    header : str or sequence of str, optional
        Text written at the top of the file.
    footer : str or sequence of str, optional
        Text written at the end of the file.

    """

    if resname not in mapping:
        raise ValueError(f"No mapping found for resname '{resname}'")
        
    filename = Path(filename).resolve()
    beads = mapping[resname]
    
    def _normalize(x):
        if x is None:
            return []
        if isinstance(x, str):
            return [x]
        return list(x)

    header_lines = _normalize(header)
    footer_lines = _normalize(footer)
    
    with open(filename, "w") as f:

        # header
        for line in header_lines:
            f.write(f"{line}\n")
        if header_lines:
            f.write("\n")

        f.write("[ moleculetype ]\n")
        f.write(f"{resname}  1\n\n")

        f.write("[ atoms ]\n")
        f.write("; nr type resnr residue atom cgnr charge [mass]\n")

        for i, (bead_name, bead_def) in enumerate(beads.items(), 1):
            bead_type = bead_def.get("type", "TYPe")
            bead_charge = bead_def.get("charge", 0)
            bead_mass = bead_def.get("mass")

            if bead_mass is None:
                f.write(f"{i:4} {bead_type:4} {0:4} {resname:4} {bead_name:4} "
                        f"{i:4} {bead_charge:6}\n")
            else:
                f.write(f"{i:4} {bead_type:4} {0:4} {resname:4} {bead_name:4} "
                        f"{i:4} {bead_charge:6} {bead_mass:8.1f}\n")

        # footer
        if footer_lines:
            f.write("\n")
            for line in footer_lines:
                f.write(f"{line}\n")