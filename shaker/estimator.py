import numpy as np
from .measurer import measure_bonded_terms
from .dihedral_fitting import fit_dihedral_workflow

_KB = 0.008314462618  # kJ mol⁻¹ K⁻¹

_SECTION_HEADERS = {
    "bonds": (
        ";  i    j  funct   length      force.c. ; beads",
        ";  i      j    funct   length      force.c."),
    "constraints": (
        ";  i    j  funct   length                ; beads",
        ";  i      j    funct   length"),
    "angles": (
        ";  i    j    k  funct   angle       force.c. ; beads",
        ";  i      j      k    funct   angle       force.c."),
    "proper_dihedrals": (
        ";  i    j    k    l  funct     phase         k     mult ; beads",
        ";  i      j      k      l    funct     phase         k     mult"),
    "improper_dihedrals": (
        ";  i    j    k    l  funct     angle         k           ; beads",
        ";  i      j      k      l    funct     angle         k"),
}


def bonded_estimator(universe, resname,
                     dist_tgts=None, ang_tgts=None,
                     harm_dihed_tgts=None, imp_dihed_tgts=None, plot_dihed=False,
                     T=300,
                     constraint_threshold=25000,
                     angle_cap=250.0, improper_cap=250.0, res_bend_angle=140.0,
                     start=0, stop=None, stride=1,
                     use_indices=False):
    """
    Measure bonded distributions from a mapped reference trajectory and estimate
    initial GROMACS bonded parameters.

    This is a convenience wrapper around `measure_bonded_terms` and
    `_estimate_bonded_from_dict`. It measures the requested bonded targets,
    estimates parameters from the resulting distributions, and returns
    GROMACS-formatted topology text.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe containing the mapped reference structure and trajectory.
    resname : str
        Residue name of the mapped molecule to analyze.
    dist_tgts : sequence of sequence of str, optional
        Bond targets to estimate. Each entry is a pair of bead names.
    ang_tgts : sequence of sequence of str, optional
        Angle targets to estimate. Each entry is a triplet of bead names.
    harm_dihed_tgts : sequence of sequence of str, optional
        Proper dihedral targets to fit. Each entry is a quartet of bead names.
        Fitted using the full dihedral workflow (inverted Boltzmann -> smooth -> fit)
        and written as type 9 (multiple terms per quartet).
    imp_dihed_tgts : sequence of sequence of str, optional
        Improper dihedral targets to estimate. Each entry is a quartet of bead names.
        Parameters estimated from the angle distribution via the equipartition
        theorem and written as type 2 (harmonic improper).
    T : float, optional
        Temperature in Kelvin. Default is 300.
    constraint_threshold : float, optional
        Bond force constant threshold above which a bond is written as a
        constraint instead of a harmonic bond. Default is 25000.
    angle_cap : float, optional
        Maximum allowed force constant for angles in kJ mol⁻¹ rad⁻². If the
        estimated force constant exceeds this value, it is capped and the
        original value is written as a comment. Default is 250.
    improper_cap : float, optional
        Maximum allowed force constant for improper dihedrals in kJ mol⁻¹ rad⁻².
        If the estimated force constant exceeds this value, it is capped and the
        original value is written as a comment. Default is 250.
    start : int, optional
        First frame to analyze. Default is 0.
    stop : int or None, optional
        Last frame to analyze. Default is None.
    stride : int, optional
        Frame stride used during measurement. Default is 1.
    use_indices : bool, optional
        If True, write bead indices instead of bead names in the output.
        Bead names are kept as comments. Default is False.

    Returns
    -------
    str
        GROMACS-formatted topology text containing `[ bonds ]`,
        `[ constraints ]`, `[ angles ]`, and `[ dihedrals ]` sections,
        depending on which targets were provided.
    """

    dist_tgts = [] if dist_tgts is None else dist_tgts
    ang_tgts = [] if ang_tgts is None else ang_tgts
    harm_dihed_tgts = [] if harm_dihed_tgts is None else harm_dihed_tgts
    imp_dihed_tgts = [] if imp_dihed_tgts is None else imp_dihed_tgts

    bead_index = None
    if use_indices:
        residues = universe.select_atoms(f"resname {resname}").residues
        if len(residues) == 0:
            raise ValueError(f"No residues found with resname '{resname}'")
        bead_index = {atom.name: i + 1 for i, atom in enumerate(residues[0].atoms)}

    bonded_dist = measure_bonded_terms(universe, resname,
                                       dist_tgts, ang_tgts, harm_dihed_tgts + imp_dihed_tgts,
                                       start=start, stop=stop, stride=stride)

    return _estimate_bonded_from_dict(
        bonded_dist,
        dist_tgts=dist_tgts,
        ang_tgts=ang_tgts,
        harm_dihed_tgts=harm_dihed_tgts, imp_dihed_tgts=imp_dihed_tgts,
        plot_dihed=plot_dihed,
        T=T,
        constraint_threshold=constraint_threshold,
        angle_cap=angle_cap, improper_cap=improper_cap, res_bend_angle=res_bend_angle,
        bead_index=bead_index,
        use_indices=use_indices)


def _estimate_bonded_from_dict(AA_bonded,
                               dist_tgts=None, ang_tgts=None,
                               harm_dihed_tgts=None, imp_dihed_tgts=None, plot_dihed=False,
                               T=300, dist_units="A", ang_units="deg",
                               constraint_threshold=25000,
                               angle_cap=250.0, improper_cap=250.0, res_bend_angle=140.0,
                               bead_index=None, use_indices=False):
    """
    Estimate initial bonded parameters from measured distributions and return
    GROMACS topology-formatted text.

    Parameters
    ----------
    AA_bonded : dict
        Output dictionary from `measure_bonded_terms`.
    dist_tgts : sequence of sequence of str, optional
        Bond targets to estimate. Each entry is a pair of bead names.
    ang_tgts : sequence of sequence of str, optional
        Angle targets to estimate. Each entry is a triplet of bead names.
    harm_dihed_tgts : sequence of sequence of str, optional
        Proper dihedral targets to fit. Each entry is a quartet of bead names.
        Fitted using the full dihedral workflow (inverted Boltzmann -> smooth -> fit)
        and written as type 9 (multiple terms per quartet).
    imp_dihed_tgts : sequence of sequence of str, optional
        Improper dihedral targets to estimate. Each entry is a quartet of bead names.
        Parameters estimated from the angle distribution via the equipartition
        theorem and written as type 2 (harmonic improper).
    plot_dihed : bool, optional
        If True, plot the raw potential, smoothed potential, and fit for each
        proper dihedral. Default is False.
    T : float, optional
        Temperature in Kelvin. Default is 300.
    dist_units : {"nm", "A"}, optional
        Units of the distance bins. Default is "A".
    ang_units : {"deg", "rad"}, optional
        Units of the angle bins. Default is "deg".
    constraint_threshold : float, optional
        Bond force constant threshold above which a bond is written as a
        constraint instead of a harmonic bond.
    angle_cap : float, optional
        Maximum allowed force constant for angles in kJ mol⁻¹ rad⁻². If the
        estimated force constant exceeds this value, it is capped and the
        original value is written as a comment. Default is 250.
    improper_cap : float, optional
        Maximum allowed force constant for improper dihedrals in kJ mol⁻¹ rad⁻².
        If the estimated force constant exceeds this value, it is capped and the
        original value is written as a comment. Default is 250.
    res_bend_angle : float, optional
        Maximum allowed bending angle in degrees. 
        If the estimated angle exceeds this value, a resticted bending potential is used 
        (angle type 10 in GROMACS). Default is 250.
    bead_index : dict, optional
        Mapping from bead name to bead index. Required only if `use_indices=True`.
    use_indices : bool, optional
        If True, write bead indices instead of bead names in the output.
        Bead names are kept as comments.

    Returns
    -------
    str
        GROMACS-formatted topology text containing `[ bonds ]`,
        `[ constraints ]`, `[ angles ]`, and `[ dihedrals ]` sections,
        depending on which targets were provided.
    """

    dist_tgts = [list(t) for t in (dist_tgts or [])]
    ang_tgts = [list(t) for t in (ang_tgts or [])]
    harm_dihed_tgts = [list(t) for t in (harm_dihed_tgts or [])]
    imp_dihed_tgts = [list(t) for t in (imp_dihed_tgts or [])]

    if use_indices and bead_index is None:
        raise ValueError("bead_index must be provided when use_indices=True")

    def _lookup(name):
        if name not in bead_index:
            raise ValueError(f"Bead name '{name}' not found in bead_index")
        return bead_index[name]

    def _fmt_bond(a, b):
        if use_indices:
            return f"{_lookup(a):4d} {_lookup(b):4d}", f"{a}-{b}"
        return f"{a:>6} {b:>6}", None

    def _fmt_angle(a, b, c):
        if use_indices:
            return f"{_lookup(a):4d} {_lookup(b):4d} {_lookup(c):4d}", f"{a}-{b}-{c}"
        return f"{a:>6} {b:>6} {c:>6}", None

    def _fmt_dihedral(a, b, c, d):
        if use_indices:
            return (f"{_lookup(a):4d} {_lookup(b):4d} "
                    f"{_lookup(c):4d} {_lookup(d):4d}"), f"{a}-{b}-{c}-{d}"
        return f"{a:>6} {b:>6} {c:>6} {d:>6}", None

    def _get_hist(term_type, tgt):
        targets = [list(t) for t in AA_bonded[term_type]["targets"]]
        if tgt not in targets:
            raise ValueError(f"Target {tgt} not found in AA_bonded['{term_type}']['targets']")
        idx = targets.index(tgt)
        return AA_bonded[term_type]["bins"], AA_bonded[term_type]["hist"][idx]

    def _build_comment(line, *parts):
        comments = [p for p in parts if p]
        if comments:
            line += " ; " + ", ".join(comments)
        return line

    def _header(key):
        return _SECTION_HEADERS[key][0 if use_indices else 1]

    bond_lines = []
    constraint_lines = []
    angle_lines = []
    harm_dihed_lines = []
    imp_dihed_lines = []

    for tgt in dist_tgts:
        bins, hist = _get_hist("distances", tgt)
        r0, k = _estimate_bond_params_from_hist(bins, hist, T=T, units=dist_units)
        ij, comment = _fmt_bond(tgt[0], tgt[1])

        if k > constraint_threshold:
            line = _build_comment(
                f"{ij}   1   {r0:8.3f}",
                comment, f"turned into constraint because k={k:.1f}")
            constraint_lines.append(line)
        else:
            line = _build_comment(
                f"{ij}   1   {r0:8.3f}   {k:10.1f}",
                comment)
            bond_lines.append(line)

    for tgt in ang_tgts:
        bins, hist = _get_hist("angles", tgt)
        theta0, ktheta = _estimate_angle_params_from_hist(bins, hist, T=T, units=ang_units)
        ijk, comment = _fmt_angle(tgt[0], tgt[1], tgt[2])

        k_write = min(ktheta, angle_cap)
        cap_comment = f"capped from k={ktheta:.1f}" if ktheta > angle_cap else None
        res_bend_comment = f"angle type set to 10 since θ>{res_bend_angle:.1f}" if theta0 > res_bend_angle else None
        angle_type = "10" if theta0 > res_bend_angle else " 1"
        line = _build_comment(
            f"{ijk}   {angle_type}   {theta0:8.2f}   {k_write:10.1f}",
            cap_comment, res_bend_comment, comment)
        angle_lines.append(line)

    for tgt in harm_dihed_tgts:
        bins, hist = _get_hist("dihedrals", tgt)
        ijkl, comment = _fmt_dihedral(tgt[0], tgt[1], tgt[2], tgt[3])
        model = fit_dihedral_workflow(bins, hist, tgt=tgt, plot=plot_dihed)

        harm_dihed_lines.append(f"; {comment or '-'.join(tgt)}")
        for i, term in enumerate(model['report']):
            tokens = term.split()
            if len(tokens) < 3:
                raise ValueError(
                    f"Unexpected format in dihedral report for {tgt}: {term!r}")
            try:
                phase, k, mult = float(tokens[0]), float(tokens[1]), int(tokens[2])
            except ValueError as e:
                raise ValueError(
                    f"Could not parse dihedral report term for {tgt}: {term!r}") from e
            line = f"{ijkl}   9   {phase:8.2f}   {k:10.4f}   {mult}"
            if i == 0:
                line = _build_comment(line, comment)
            harm_dihed_lines.append(line)
        harm_dihed_lines.append("")

    for tgt in imp_dihed_tgts:
        bins, hist = _get_hist("dihedrals", tgt)
        ijkl, comment = _fmt_dihedral(tgt[0], tgt[1], tgt[2], tgt[3])
        theta0, ktheta = _estimate_improper_params_from_hist(bins, hist, T=T, units="deg")

        k_write = min(ktheta, improper_cap)
        cap_comment = f"capped from k={ktheta:.4f}" if ktheta > improper_cap else None
        line = _build_comment(
            f"{ijkl}   2   {theta0:8.2f}   {k_write:10.4f}",
            cap_comment, comment)
        imp_dihed_lines.append(line)

    lines = []
    if bond_lines:
        lines += ["[ bonds ]", _header("bonds")] + bond_lines + [""]

    if constraint_lines:
        lines += ["[ constraints ]", _header("constraints")] + constraint_lines + [""]

    if angle_lines:
        lines += ["[ angles ]", _header("angles")] + angle_lines + [""]

    if harm_dihed_lines or imp_dihed_lines:
        lines.append("[ dihedrals ]")
        if harm_dihed_lines:
            lines += [_header("proper_dihedrals")] + harm_dihed_lines
        if imp_dihed_lines:
            lines += ["; improper dihedrals", _header("improper_dihedrals")] + imp_dihed_lines + [""]

    return "\n".join(lines)


def _estimate_bond_params_from_hist(bins_x, hist, T=300, units="A"):
    """
    Estimate harmonic bond parameters from a histogrammed bond-length distribution.

    U(r) = ½ k (r − r₀)²  →  σ² = k_B T / k

    Parameters
    ----------
    bins_x : array-like
        Histogram bin centers representing bond distances.
    hist : array-like
        Probability density values corresponding to `bins_x`.
    T : float, optional
        Temperature in Kelvin. Default is 300 K.
    units : {"nm", "A"}, optional
        Units of the bond distances. "A" (Ångström) are converted to nm internally.

    Returns
    -------
    r0 : float
        Estimated equilibrium bond length in nanometers.
    k : float
        Estimated harmonic force constant in kJ mol⁻¹ nm⁻².
    """
    bins_x = np.asarray(bins_x)
    hist = np.asarray(hist)

    mask = np.isfinite(hist) & (hist > 0)
    x = bins_x[mask]
    p = hist[mask]

    r0 = np.sum(x * p) / np.sum(p)
    var = np.sum(p * (x - r0)**2) / np.sum(p)
    k = _KB * T / var

    if units == "A":
        r0 = r0 / 10.0    # Å -> nm
        k = k * 100.0     # kJ/mol/Å² -> kJ/mol/nm²
    elif units != "nm":
        raise ValueError("units must be 'nm' or 'A'")

    return r0, k


def _estimate_angle_params_from_hist(bins_x, hist, T=300, units="deg"):
    """
    Estimate harmonic angle parameters from a histogrammed angle distribution.

    U(θ) = ½ kθ (θ − θ₀)²  →  σ² = k_B T / kθ

    Parameters
    ----------
    bins_x : array-like
        Histogram bin centers representing angles.
    hist : array-like
        Probability density values corresponding to `bins_x`.
    T : float, optional
        Temperature in Kelvin. Default is 300 K.
    units : {"deg", "rad"}, optional
        Units of the angles in `bins_x`. "deg" is converted to radians internally.

    Returns
    -------
    theta0 : float
        Estimated equilibrium angle in degrees.
    ktheta : float
        Estimated harmonic force constant in kJ mol⁻¹ rad⁻².
    """
    bins_x = np.asarray(bins_x)
    hist = np.asarray(hist)

    mask = np.isfinite(hist) & (hist > 0)
    x = bins_x[mask]
    p = hist[mask]

    if units == "deg":
        x = np.deg2rad(x)
    elif units != "rad":
        raise ValueError("units must be 'deg' or 'rad'")

    theta0 = np.sum(x * p) / np.sum(p)
    var = np.sum(p * (x - theta0)**2) / np.sum(p)
    ktheta = _KB * T / var

    return np.rad2deg(theta0), ktheta


def _estimate_improper_params_from_hist(bins_x, hist, T=300, units="deg"):
    """
    Estimate harmonic improper dihedral parameters from a histogrammed
    distribution, with correct handling of periodicity near ±180°.

    U(ξ) = ½ kξ (ξ − ξ₀)²  →  σ² = k_B T / kξ

    Distributions near ±180° wrap around the periodic boundary, inflating the
    variance and underestimating kξ. This is corrected by shifting the
    distribution so the peak sits at 0° before computing statistics, then
    shifting the equilibrium angle back.

    Parameters
    ----------
    bins_x : array-like
        Histogram bin centers representing improper dihedral angles.
    hist : array-like
        Probability density values corresponding to `bins_x`.
    T : float, optional
        Temperature in Kelvin. Default is 300 K.
    units : {"deg", "rad"}, optional
        Units of the angles in `bins_x`. "deg" is converted to radians internally.

    Returns
    -------
    xi0 : float
        Estimated equilibrium improper angle in degrees, in [-180°, 180°].
    kxi : float
        Estimated harmonic force constant in kJ mol⁻¹ rad⁻².
    """
    bins_x = np.asarray(bins_x, dtype=float)
    hist = np.asarray(hist, dtype=float)

    mask = np.isfinite(hist) & (hist > 0)
    x = bins_x[mask]
    p = hist[mask]

    if units == "deg":
        x = np.deg2rad(x)
    elif units != "rad":
        raise ValueError("units must be 'deg' or 'rad'")

    # shift so the peak sits at 0 to avoid wrap-around variance inflation
    peak = x[np.argmax(p)]
    x_shifted = ((x - peak + np.pi) % (2 * np.pi)) - np.pi

    theta0_shifted = np.sum(x_shifted * p) / np.sum(p)
    var = np.sum(p * (x_shifted - theta0_shifted) ** 2) / np.sum(p)

    xi0 = ((theta0_shifted + peak + np.pi) % (2 * np.pi)) - np.pi
    kxi = _KB * T / var

    return np.rad2deg(xi0), kxi