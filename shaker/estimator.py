import numpy as np
from .measurer import measure_bonded_terms
from .dihedral_fitting import fit_dihedral_workflow

def bonded_estimator(universe, resname,
                     dist_tgts=None, ang_tgts=None,
                     harm_dihed_tgts=None, imp_dihed_tgts=None, plot_dihed=False,
                     T=300,
                     constraint_threshold=25000,
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
        Fitted using the full dihedral workflow (inverted Boltzmann → smooth → fit)
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

    if dist_tgts is None:
        dist_tgts = []
    if ang_tgts is None:
        ang_tgts = []
    if harm_dihed_tgts is None:
        harm_dihed_tgts = []
    if imp_dihed_tgts is None:
        imp_dihed_tgts = []
        
    bead_index = None
    if use_indices:
        residues = universe.select_atoms(f"resname {resname}").residues
        if len(residues) == 0:
            raise ValueError(f"No residues found with resname '{resname}'")
        bead_index = {atom.name: i + 1 for i, atom in enumerate(residues[0].atoms)}

    bonded_dist = measure_bonded_terms(universe, resname,
                                        dist_tgts, ang_tgts, harm_dihed_tgts+imp_dihed_tgts,
                                        start=start, stop=stop, stride=stride)

    return _estimate_bonded_from_dict(
        bonded_dist,
        dist_tgts=dist_tgts,
        ang_tgts=ang_tgts,
        harm_dihed_tgts=harm_dihed_tgts, imp_dihed_tgts=imp_dihed_tgts,
        plot_dihed=plot_dihed,
        T=T,
        constraint_threshold=constraint_threshold,
        bead_index=bead_index,
        use_indices=use_indices)


def _estimate_bonded_from_dict(AA_bonded,
                               dist_tgts=None, ang_tgts=None, 
                               harm_dihed_tgts=None, imp_dihed_tgts=None, plot_dihed=True,
                               T=300, dist_units="A", ang_units="deg",
                               constraint_threshold=25000,
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
        Fitted using the full dihedral workflow (inverted Boltzmann → smooth → fit)
        and written as type 9 (multiple terms per quartet).
    imp_dihed_tgts : sequence of sequence of str, optional
        Improper dihedral targets to estimate. Each entry is a quartet of bead names.
        Parameters estimated from the angle distribution via the equipartition
        theorem and written as type 2 (harmonic improper).
    plot_dihed : bool, optional
        If True, plot the raw potential, smoothed potential, and fit for each
        proper dihedral. Default is True.
    T : float, optional
        Temperature in Kelvin. Default is 300.
    dist_units : {"nm", "A"}, optional
        Units of the distance bins. Default is "A".
    ang_units : {"deg", "rad"}, optional
        Units of the angle bins. Default is "deg".
    constraint_threshold : float, optional
        Bond force constant threshold above which a bond is written as a
        constraint instead of a harmonic bond.
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

    if dist_tgts is None:
        dist_tgts = []
    if ang_tgts is None:
        ang_tgts = []
    if harm_dihed_tgts is None:
        harm_dihed_tgts = []
    if imp_dihed_tgts is None:
        imp_dihed_tgts = []

    if use_indices and bead_index is None:
        raise ValueError("bead_index must be provided when use_indices=True")

    def _fmt_bond(a, b):
        if use_indices:
            if a not in bead_index or b not in bead_index:
                raise ValueError(f"Bond target {[a, b]} contains bead names not found in bead_index")
            return f"{bead_index[a]:4d} {bead_index[b]:4d}", f"{a}-{b}"
        return f"{a:>6} {b:>6}", None

    def _fmt_angle(a, b, c):
        if use_indices:
            if a not in bead_index or b not in bead_index or c not in bead_index:
                raise ValueError(f"Angle target {[a, b, c]} contains bead names not found in bead_index")
            return f"{bead_index[a]:4d} {bead_index[b]:4d} {bead_index[c]:4d}", f"{a}-{b}-{c}"
        return f"{a:>6} {b:>6} {c:>6}", None
        
    def _fmt_dihedral(a, b, c, d):
        if use_indices:
            for bead in [a, b, c, d]:
                if bead not in bead_index:
                    raise ValueError(f"Dihedral target {[a,b,c,d]} contains bead name '{bead}' not found in bead_index")
            return (f"{bead_index[a]:4d} {bead_index[b]:4d} "
                    f"{bead_index[c]:4d} {bead_index[d]:4d}"), f"{a}-{b}-{c}-{d}"
        return f"{a:>6} {b:>6} {c:>6} {d:>6}", None
        
    bond_lines = []
    constraint_lines = []
    angle_lines = []
    harm_dihed_lines = []
    imp_dihed_lines = []
    
    if dist_tgts:
        aa_dist_targets = [list(t) for t in AA_bonded["distances"]["targets"]]
        bins_dist = AA_bonded["distances"]["bins"]
        hists_dist = AA_bonded["distances"]["hist"]

        for tgt in dist_tgts:
            tgt = list(tgt)
            if tgt not in aa_dist_targets:
                raise ValueError(f"Bond target {tgt} not found in AA_bonded['distances']['targets']")

            idx = aa_dist_targets.index(tgt)
            r0, k = _estimate_bond_params_from_hist(bins_dist, hists_dist[idx], T=T, units=dist_units)

            ij, comment = _fmt_bond(tgt[0], tgt[1])

            if k > constraint_threshold:
                line = f"{ij}   1   {r0:8.3f}"
                if comment:
                    line += f" ; {comment}, turned into constraint because k={k:.1f}"
                else:
                    line += f" ; turned into constraint because k={k:.1f}"
                constraint_lines.append(line)
            else:
                line = f"{ij}   1   {r0:8.3f}   {k:10.1f}"
                if comment:
                    line += f" ; {comment}"
                bond_lines.append(line)

    if ang_tgts:
        aa_ang_targets = [list(t) for t in AA_bonded["angles"]["targets"]]
        bins_ang = AA_bonded["angles"]["bins"]
        hists_ang = AA_bonded["angles"]["hist"]

        for tgt in ang_tgts:
            tgt = list(tgt)
            if tgt not in aa_ang_targets:
                raise ValueError(f"Angle target {tgt} not found in AA_bonded['angles']['targets']")

            idx = aa_ang_targets.index(tgt)
            theta0, ktheta = _estimate_angle_params_from_hist(
                bins_ang, hists_ang[idx], T=T, units=ang_units)
            ijk, comment = _fmt_angle(tgt[0], tgt[1], tgt[2])
            line = f"{ijk}   1   {theta0:8.2f}   {ktheta:10.1f}"
            
            if comment:
                line += f" ; {comment}"
            angle_lines.append(line)

    
    if harm_dihed_tgts or imp_dihed_tgts:
        aa_dih_targets = [list(t) for t in AA_bonded["dihedrals"]["targets"]]
        bins_dih = AA_bonded["dihedrals"]["bins"]
        hists_dih = AA_bonded["dihedrals"]["hist"]

        if harm_dihed_tgts:
            for tgt in harm_dihed_tgts:
                tgt = list(tgt)
                if tgt not in aa_dih_targets:
                    raise ValueError(f"Dihedral target {tgt} not found in AA_bonded['dihedrals']['targets']")

                idx = aa_dih_targets.index(tgt)
                ijkl, comment = _fmt_dihedral(tgt[0], tgt[1], tgt[2], tgt[3])
                model = fit_dihedral_workflow(bins_dih, hists_dih[idx], tgt=tgt, plot=plot_dihed)

                harm_dihed_lines.append(f"; {comment or '-'.join(tgt)}")
                for i, term in enumerate(model['report']):
                    phase, k, mult = term.split()[:3]
                    line = f"{ijkl}   9   {float(phase):8.2f}   {float(k):10.4f}   {int(mult)}"
                    if i == 0 and comment:
                        line += f" ; {comment}"
                    harm_dihed_lines.append(line)
                harm_dihed_lines.append("")

        if imp_dihed_tgts:
            for tgt in imp_dihed_tgts:
                tgt = list(tgt)
                if tgt not in aa_dih_targets:
                    raise ValueError(f"Improper dihedral target {tgt} not found in AA_bonded['dihedrals']['targets']")

                idx = aa_dih_targets.index(tgt)
                ijkl, comment = _fmt_dihedral(tgt[0], tgt[1], tgt[2], tgt[3])
                theta0, ktheta = _estimate_angle_params_from_hist(
                    bins_dih, hists_dih[idx], T=T, units="deg")

                line = f"{ijkl}   2   {theta0:8.2f}   {ktheta:10.4f}"
                if comment:
                    line += f" ; {comment}"
                imp_dihed_lines.append(line)
    
    lines = []
    if bond_lines:
        lines.append("[ bonds ]")
        if use_indices:
            lines.append(";  i    j  funct   length      force.c. ; beads")
        else:
            lines.append(";  i      j    funct   length      force.c.")
        lines.extend(bond_lines)
        lines.append("")

    if constraint_lines:
        lines.append("[ constraints ]")
        if use_indices:
            lines.append(";  i    j  funct   length                ; beads")
        else:
            lines.append(";  i      j    funct   length")
        lines.extend(constraint_lines)
        lines.append("")

    if angle_lines:
        lines.append("[ angles ]")
        if use_indices:
            lines.append(";  i    j    k  funct   angle       force.c. ; beads")
        else:
            lines.append(";  i      j      k    funct   angle       force.c.")
        lines.extend(angle_lines)
        lines.append("")

    if harm_dihed_lines or imp_dihed_lines:
        lines.append("[ dihedrals ]")
        if harm_dihed_lines:
            if use_indices:
                lines.append(";  i    j    k    l  funct     phase         k     mult ; beads")
            else:
                lines.append(";  i      j      k      l    funct     phase         k     mult")
            lines.extend(harm_dihed_lines)
        if imp_dihed_lines:
            lines.append("; improper dihedrals")
            if use_indices:
                lines.append(";  i    j    k    l  funct     angle         k           ; beads")
            else:
                lines.append(";  i      j      k      l    funct     angle         k")
            lines.extend(imp_dihed_lines)
            lines.append("")
        
    return "\n".join(lines)
    
def _estimate_bond_params_from_hist(bins_x, hist, T=300, units="A"):
    '''
    Estimate harmonic bond parameters from a histogrammed bond-length distribution.

    This function derives an initial estimate of the equilibrium bond length
    (r₀) and force constant (k) directly from a probability distribution of
    bond distances. The method assumes the bond potential is approximately
    harmonic near its minimum:

        U(r) = ½ k (r − r₀)²

    Under this assumption, statistical mechanics gives a simple relationship
    between the variance of the bond-length distribution and the force constant:

        σ² = k_B T / k

    where σ² is the variance of the distribution, k_B is the Boltzmann constant,
    and T is the temperature. The equilibrium bond length is estimated as the
    mean of the distribution, while the force constant is obtained from the
    inverse variance.

    Parameters
    ----------
    bins_x : array-like
        Histogram bin centers representing bond distances.
    hist : array-like
        Probability density values corresponding to `bins_x`. Typically the
        output of `numpy.histogram(..., density=True)` or the distributions
        returned by `shaker.measurer.measure_bonded_terms`.
    T : float, optional
        Temperature of the simulation in Kelvin. Default is 300 K.
    units : {"nm", "A"}, optional
        Units of the bond distances in `bins_x`.
        - "nm": distances are already in nanometers
        - "A": distances are in Ångström and will be converted internally

    Returns
    -------
    r0 : float
        Estimated equilibrium bond length in nanometers.
    k : float
        Estimated harmonic force constant in kJ mol⁻¹ nm⁻² (GROMACS units).

    Notes
    -----
    This estimator assumes the bond distribution is approximately Gaussian,
    which is generally valid for bonded terms near equilibrium. It is therefore
    most useful for generating **initial guesses** for coarse-grained bonded
    parameters, which can later be refined by matching AA and CG distributions.
    '''
    
    kB = 0.008314462618  # kJ mol^-1 K^-1

    bins_x = np.asarray(bins_x)
    hist = np.asarray(hist)

    mask = np.isfinite(hist) & (hist > 0)
    x = bins_x[mask]
    p = hist[mask]

    r0 = np.sum(x * p) / np.sum(p)
    var = np.sum(p * (x - r0)**2) / np.sum(p)
    k = kB * T / var

    if units == "A":
        r0 = r0 / 10.0      # convert Å -> nm
        k = k * 100.0       # convert kJ/mol/Å² -> kJ/mol/nm²
    elif units != "nm":
        raise ValueError("units must be 'nm' or 'A'")

    return r0, k


def _estimate_angle_params_from_hist(bins_x, hist, T=300, units="deg"):
    """
    Estimate harmonic angle parameters from a histogrammed angle distribution.

    This function derives an initial estimate of the equilibrium angle (θ₀)
    and force constant (kθ) directly from a probability distribution of angles.
    The method assumes the angle potential is approximately harmonic near its
    minimum:

        U(θ) = ½ kθ (θ − θ₀)²

    Under this assumption, statistical mechanics gives the relationship:

        σ² = k_B T / kθ

    where σ² is the variance of the angle distribution, k_B is the Boltzmann
    constant, and T is the temperature.

    The equilibrium angle is estimated as the mean of the distribution,
    and the force constant is computed from the inverse variance.

    Parameters
    ----------
    bins_x : array-like
        Histogram bin centers representing angles.
    hist : array-like
        Probability density values corresponding to `bins_x`.
    T : float, optional
        Temperature in Kelvin. Default is 300 K.
    units : {"deg", "rad"}, optional
        Units of the angles in `bins_x`.
        - "deg": angles are in degrees (converted internally)
        - "rad": angles are already in radians

    Returns
    -------
    theta0 : float
        Estimated equilibrium angle in degrees.
    ktheta : float
        Estimated harmonic force constant in kJ mol⁻¹ rad⁻² (GROMACS units).

    Notes
    -----
    The estimator assumes the angle distribution is approximately Gaussian
    around its minimum. This is usually valid for many bonded terms and makes
    the method useful for generating **initial guesses** of CG angle parameters.
    """
    
    kB = 0.008314462618  # kJ mol^-1 K^-1

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

    ktheta = kB * T / var

    return np.rad2deg(theta0), ktheta