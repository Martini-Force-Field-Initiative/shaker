"""Utility helper functions used across SHAKER modules."""

import warnings
from functools import lru_cache
from importlib.resources import files

import numpy as np

_bead_sizes_dict = {"R": 0.264, "S": 0.230, "T": 0.191}

_bead_masses_dict = {"R": 72.0, "S": 54.0, "T": 36.0}

_BEAD_RATIOS = {
    "regular": 4,
    "small": 3,
    "tiny": 2,
}

_size_label = {"R": "regular", "S": "small", "T": "tiny", "U": "virtual"}

# Default histogram bin edges. Distances (Å) go to 50 Å so large molecules
# are not truncated; angles and dihedrals (°) cover their full range exactly.
_BINS_DIST = np.linspace(0, 50, 251)  # 0.2 Å
_BINS_ANGL = np.linspace(0, 180, 91)  # 2°
_BINS_DIHED = np.linspace(-180, 180, 181)  # 2°, the full circle once


def _density_histograms(values, edges, names, arg):
    """
    One `density=True` histogram per list in `values`, warning when values
    fall outside `edges`: np.histogram drops them silently and rescales the
    rest, which skews the distribution. `names` labels each list in the
    warning; `arg` is the bins argument to point the user to.
    """
    hists = []
    for vals, name in zip(values, names):
        vals = np.asarray(vals, float)
        outside = np.count_nonzero((vals < edges[0]) | (vals > edges[-1]))
        if outside:
            warnings.warn(
                f"{name}: {outside / vals.size:.1%} of values fall outside the bins "
                f"({edges[0]:g} to {edges[-1]:g}) and are ignored; pass {arg}= "
                "to cover them.",
                stacklevel=3,
            )
        # All values outside (already warned): numpy's own 0/0 warning would
        # only repeat it; the histogram is left as NaN.
        with np.errstate(invalid="ignore", divide="ignore"):
            hists.append(np.histogram(vals, bins=edges, density=True)[0])
    return np.array(hists)


def _size_from_name(bead_types):
    """
    Convert bead types into size classes (R, S, T, U).

    U is treated as a special case (radius = 0).
    """
    sizes = []
    for string in bead_types:
        t = string[0].upper()
        if t == "U":
            sizes.append("U")
        elif t == "S":
            sizes.append("S")
        elif t == "T":
            sizes.append("T")
        else:
            sizes.append("R")
    return sizes


def _bead_mass_from_type(bead_type):
    """
    Function to convertbead types into a mass value.
    """
    if bead_type.startswith("S"):
        return _bead_masses_dict["S"]
    if bead_type.startswith("T"):
        return _bead_masses_dict["T"]
    return _bead_masses_dict["R"]


def _category_from_type(bead_type):
    """
    Return the Martini chemical-category letter (e.g. C, P, N, X, Q, D, W)
    for a bead type string, stripping the optional small/tiny size prefix
    ('S'/'T') and any trailing polarity/H-bonding suffix letters. 'U'
    (virtual) has no size-prefixed variants and is returned as-is.
    """
    if not bead_type:
        return ""
    t = bead_type[0].upper()
    if t == "U":
        return "U"
    if t in ("S", "T") and len(bead_type) > 1:
        t = bead_type[1].upper()
    return t


@lru_cache(maxsize=1)
def _valid_martini_bead_types():
    """
    Parse the bundled Martini 3 force field and return the set of valid
    bead type names, taken from the [ atomtypes ] section of
    martini_v3.0.0.itp.
    """
    itp_path = files("shaker.data.itps") / "martini_v3.0.0.itp"
    types = set()
    in_section = False
    with itp_path.open() as f:
        for line in f:
            line = line.split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("["):
                in_section = line.strip("[] ").strip() == "atomtypes"
                continue
            if in_section:
                types.add(line.split()[0])
    return types


def _validate_bead_types(bead_map, require_type=False):
    """
    Check that bead types in ``bead_map`` are recognized Martini 3 atom
    types.

    Parameters
    ----------
    bead_map : dict
        Bead definitions for a single residue, e.g. ``mapping[resname]``.
    require_type : bool, optional
        If True, every bead must define a "type" field. If False
        (default), beads without one are skipped.

    Raises
    ------
    ValueError
        If a type is required but missing, or a given type is not a
        recognized Martini 3 bead type.
    """
    valid_types = _valid_martini_bead_types()
    for bead_name, bead_def in bead_map.items():
        bead_type = bead_def.get("type")
        if bead_type is None:
            if require_type:
                raise ValueError(f"Bead {bead_name!r} has no 'type' defined.")
            continue
        if bead_type not in valid_types:
            raise ValueError(
                f"Bead {bead_name!r} has type {bead_type!r}, which is not a "
                "recognized Martini 3 bead type."
            )
