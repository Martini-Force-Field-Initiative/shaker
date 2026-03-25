import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import warnings
from math import factorial

'''
Functions for dihedral fitting routines.
'''

###
### Helpers
def _harm_design_matrix(bins, mults):
    th = np.deg2rad(bins)
    cols = [np.ones_like(th)]
    for n in mults:
        cols.append(np.cos(n*th))
        cols.append(np.sin(n*th))
    return np.column_stack(cols)

def _aic(rss, k, N):
    return N*np.log(rss / max(N,1)) + 2*k

def _bic(rss, k, N):
    return N*np.log(rss / max(N,1)) + k*np.log(max(N,1))

def _make_weights(y, weight_mode="boltzmann", weights=None,
                  w_temp_kj=2.5):
    """
    Build a normalised weight array emphasising low-energy points.

    The returned weights have mean 1 (i.e. sum to N), so they scale the
    least-squares problem without changing its overall magnitude.

    Parameters
    ----------
    y : np.ndarray
        Energy values in kJ/mol.
    weight_mode : {"boltzmann", "none"}
        "boltzmann" — weights decay exponentially with energy above the minimum:
            w = exp(-ΔE / w_temp_kj)
        Lower w_temp_kj sharpens the emphasis on the minimum; higher flattens it.
        "none" — all points weighted equally.
    weights : array-like, optional
        Supply your own weight array directly, bypassing weight_mode entirely.
        Useful when fitting MD-derived PMFs where raw histogram bin counts
        might reflect statistical reliability of each point.
    w_temp_kj : float
        Effective temperature in kJ/mol controlling Boltzmann weight decay.
        Default 2.5 (~RT at 300 K). Only used when weight_mode="boltzmann".

    Returns
    -------
    np.ndarray
        Positive weights of the same length as y, normalised so mean = 1.
        Falls back to uniform weights if all computed weights are zero or invalid.
    """
    if weights is not None:
        w = np.asarray(weights, dtype=float)
    else:
        y0 = y - np.nanmin(y)
        if weight_mode == "boltzmann":
            w = np.exp(-y0 / float(w_temp_kj))
        elif weight_mode in (None, "none"):
            w = np.ones_like(y)
        else:
            raise ValueError("weight_mode must be 'boltzmann', 'none', or provide `weights`.")
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    total = np.sum(w)
    return w * (len(w) / total) if total > 0 else np.ones_like(y)


### InvBoltz and Savitzky
def inverted_boltzmann(data_density, kcal=False, temp=298,
                       interpolate=False, penalty=0.0,
                       interp_method="linear", zero_min=True):
    '''
    Convert a density histogram to a PMF via -kT ln P(θ).

    Parameters
    ----------
    data_density : array-like
        Raw bin counts or probability density from an MD histogram.
    kcal : bool
        If True, return energy in kcal/mol. Default False (kJ/mol).
    temp : float
        Temperature in Kelvin. Default 298.
    interpolate : bool
        If True, fill NaN bins (empty histogram bins) by interpolation.
    penalty : float
        Constant added to interpolated values to up-weight sparse regions.
    interp_method : {"linear", "cubic"}
        Interpolation method for NaN bins. Default "linear".
    zero_min : bool
        If True, shift output so the global minimum is zero. Default True.

    Returns
    -------
    np.ndarray
        PMF in kJ/mol (or kcal/mol) at each bin.
    '''
    
    data_density = data_density / np.sum(data_density, axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        data_energy = -np.log(data_density)
    data_energy -= np.nanmin(data_energy, axis=0)

    if kcal:
        data_energy *= 8.3144621 * temp / float(1000 * 4.184)
    else:
        data_energy *= 8.3144621 * temp / float(1000)
    data_energy[np.isinf(data_energy)] = np.nan

    if interpolate:
        x = np.arange(len(data_energy))
        mask = np.isnan(data_energy)
        if np.any(mask) and np.any(~mask):
            if interp_method == "cubic":
                from scipy.interpolate import CubicSpline
                data_energy[mask] = CubicSpline(x[~mask], data_energy[~mask])(x[mask]) + penalty
            else:
                data_energy[mask] = np.interp(x[mask], x[~mask], data_energy[~mask]) + penalty

    if zero_min:
        min_val = np.nanmin(data_energy)
        if np.isfinite(min_val):
            data_energy -= min_val

    return data_energy


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.array([[k**i for i in order_range]
        for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b)[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


### Periodic dihedral angle fitting.
def fit_dihedral_workflow(bins, hist, tgt=None,
                          interpolate=True, penalty=2.0, interp_method="linear",
                          sg_window=11, sg_order=2,
                          max_terms=5, max_multiplicity=5,
                          criterion="BIC", weight_mode="boltzmann", weights=None,
                          w_temp_kj=2.5, plot=False):    
    """
    Full dihedral fitting workflow: inverted Boltzmann → smooth → fit → report.

    Parameters
    ----------
    bins : array-like
        Dihedral angle bin centres in degrees.
    hist : array-like
        Raw histogram counts or probability density.
    tgt : list of str, optional
        Atom names defining the dihedral, used as the plot title.
    interpolate : bool
        Fill empty histogram bins by interpolation. Default True.
    penalty : float
        Energy penalty added to interpolated bins. Default 2.0.
    interp_method : {"linear", "cubic"}
        Interpolation method for empty bins. Default "linear".
    sg_window : int
        Savitzky-Golay window size (must be odd). Default 11.
    sg_order : int
        Savitzky-Golay polynomial order. Default 2.
    max_terms : int
        Maximum number of harmonic terms in the fit. Default 4.
    max_multiplicity : int
        Highest multiplicity to consider. Default 4.
    criterion : {"AIC", "BIC"}
        Model selection criterion. Default "BIC".
    weight_mode : {"boltzmann", "none"}
        Weighting scheme for the fit. Default "boltzmann".
    weights : array-like, optional
        Custom per-point weights. Overrides weight_mode.
    w_temp_kj : float
        Effective temperature for Boltzmann weighting. Default 2.5.
    plot : bool
        If True, plot raw potential, smoothed potential and fit. Default False.

    Returns
    -------
    dict
        Output of fit_periodic_harmonics, with two extra keys:
            "potential"        — raw PMF from inverted Boltzmann
            "smooth_potential" — Savitzky-Golay smoothed PMF
    """

    potential        = inverted_boltzmann(hist, interpolate=interpolate,
                                          penalty=penalty, interp_method=interp_method)
    smooth_potential = savitzky_golay(potential, sg_window, sg_order)
    
    model            = fit_periodic_harmonics(bins, smooth_potential,
                                              max_terms=max_terms,
                                              max_multiplicity=max_multiplicity,
                                              criterion=criterion,
                                              weight_mode=weight_mode,
                                              weights=weights,
                                              w_temp_kj=w_temp_kj)
    model["potential"]        = potential
    model["smooth_potential"] = smooth_potential
    model["report"] = report_potentials(model, bins=bins, energy_kj=smooth_potential, return_pots=True)

    if plot:

        fit_data = evaluate_model(bins, model)
        title = '-'.join(tgt) if tgt is not None else "Dihedral fit"
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.plot(bins, potential,        color='k', lw=1, alpha=0.5, label='Raw potential')
        ax.plot(bins, smooth_potential, color='k', lw=2, alpha=1.0, label='Smoothed')
        ax.plot(bins, fit_data,         color='tab:red', lw=2, alpha=1.0, label='Fit')
        ax.legend(frameon=False)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Potential (kJ/mol)', fontweight='bold')
        ax.set_xlabel('Dihedral angle (°)', fontweight='bold')
        ax.set_xlim(-180, 180)
        fig.tight_layout()
        plt.show()

        ## Also print the potential
        for line in model["report"]:
            print(line)

    return model

def fit_periodic_harmonics(
    bins, energy_kj,
    max_terms=4, max_multiplicity=4,
    criterion="BIC", zero_min=True, 
    weight_mode="boltzmann", weights=None,
    w_temp_kj=2.5):
    
    """
    Fit a dihedral potential energy surface to a sum of periodic harmonics.

    Each term takes the GROMACS 1+cos form:
        V(θ) = Σ kₙ · (1 + cos(nθ - δₙ))
    where n is the multiplicity, kₙ the barrier height, and δₙ the phase.

    The fitter performs an exhaustive search over all combinations of
    multiplicities 1..max_multiplicity, testing models with 1..max_terms
    harmonics. For each candidate model, weighted least squares is solved
    in cos/sin form and converted to amplitude/phase. The best model is
    selected by AIC or BIC.

    Low-energy regions are upweighted by default (Boltzmann weighting) since
    they dominate the populated conformational space and matter most for
    reproducing the correct dynamics.

    Parameters
    ----------
    bins : array-like
        Dihedral angle bin centres in degrees, typically -180 to 180.
    energy_kj : array-like
        Potential energy in kJ/mol at each bin. NaN and inf are ignored.
    max_terms : int
        Maximum number of harmonic terms to include. Default 4.
    max_multiplicity : int
        Highest multiplicity n to consider. Default 4.
    criterion : {"AIC", "BIC"}
        Model selection criterion used to balance fit quality against model
        complexity. AIC applies a flat penalty of 2 per parameter, favouring
        models that fit well even at the cost of extra terms. BIC penalises
        each parameter by ln(N) instead — for a typical 72-point scan that is
        ~4.3, so BIC selects simpler models and only accepts an extra harmonic
        if it substantially improves the fit. Default "AIC".
    zero_min : bool
        If True, shift the fitted potential so its global minimum is exactly
        zero. Evaluated on a dense 2001-point grid to avoid missing the true
        minimum between bins. Default True.
    weight_mode : {"boltzmann", "none"}
        Weighting scheme. "boltzmann" emphasises low-energy points using
        w = exp(-ΔE / w_temp_kj). "none" weights all points equally.
        Ignored if weights is provided. Default "boltzmann".
    weights : array-like, optional
        Custom per-point weights, e.g. raw histogram bin counts from an MD
        run. Overrides weight_mode entirely. Must be the same length as
        energy_kj.
    w_temp_kj : float
        Effective temperature controlling Boltzmann weight decay in kJ/mol.
        Default 2.5 (~RT at 300 K).

    Returns
    -------
    dict with keys:
        mults : tuple of int
            Multiplicities of the selected harmonic terms.
        amps : np.ndarray
            Barrier heights kₙ in kJ/mol for each term.
        phases_deg : np.ndarray
            Phase angles δₙ in degrees for each term.
        offset_1pluscos : float
            Constant offset so that min V(θ) = 0.
        beta : np.ndarray
            Raw least-squares coefficients in cos/sin form.
        rss_w : float
            Weighted residual sum of squares of the best model.
        score : float
            AIC or BIC score of the best model.
        criterion : str
            Which criterion was used.
        weights_summary : dict
            Records the weighting parameters used.
    """
    
    bins = np.asarray(bins)
    energy_kj = np.asarray(energy_kj)

    msk = np.isfinite(bins) & np.isfinite(energy_kj)
    x, y = bins[msk], energy_kj[msk]
    if len(y) == 0:
        raise ValueError("No valid data points.")
    N = len(y)

    # Build weights
    w_input = np.asarray(weights)[msk] if (weights is not None 
                                           and len(np.asarray(weights)) == len(energy_kj)) else weights
    w = _make_weights(y, weight_mode=weight_mode, weights=w_input, w_temp_kj=w_temp_kj)
    sqrtw = np.sqrt(w)
    all_mults = list(range(1, max_multiplicity+1))
    best = None

    for r in range(1, max_terms+1):
        for mults in combinations(all_mults, r):
            X = _harm_design_matrix(x, mults)
            # Weighted least squares: solve (W^{1/2} X) beta ≈ (W^{1/2} y)
            Xw = X * sqrtw[:, None]
            yw = y * sqrtw
            beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

            resid = y - X @ beta
            rss_w = np.sum(w * resid**2)  # weighted RSS
            k_params = X.shape[1]
            score = _aic(rss_w, k_params, N) if criterion.upper()=="AIC" else _bic(rss_w, k_params, N)

            if (best is None) or (score < best["score"]):
                c0 = beta[0]
                a_b = beta[1:]
                amps, phases = [], []
                for i in range(len(mults)):
                    a = a_b[2*i]; b = a_b[2*i+1]
                    A = np.hypot(a, b)
                    delta = np.degrees(np.arctan2(b, a))
                    amps.append(A); phases.append(delta)
                best = {"mults": mults,
                        "beta": beta,
                        "c0": c0,
                        "amps": np.array(amps),
                        "phases_deg": np.array(phases),
                        "offset_1pluscos": c0 - np.sum(amps),  # before zeroing min
                        "rss_w": rss_w,
                        "score": score,
                        "criterion": criterion.upper(),
                        "weights_summary": dict(mode=weight_mode, Tw=w_temp_kj)}

    # Enforce global min = 0 by adjusting constant
    if zero_min and best is not None:
        theta_grid = np.linspace(-180, 180, 2001)
        th = np.deg2rad(theta_grid)
        g = np.zeros_like(th, dtype=float)
        for n, k, ddeg in zip(best["mults"], best["amps"], best["phases_deg"]):
            g += k * (1.0 + np.cos(n*th - np.deg2rad(ddeg)))
        c_zero = -np.min(g)
        best["offset_1pluscos"] = c_zero

    return best


def evaluate_model(bins, model):
    """
    Evaluate the fitted dihedral potential at given bin centres.

    Parameters
    ----------
    bins : array-like
        Bin centres in degrees.
    model : dict
        Output of fit_periodic_harmonics.

    Returns
    -------
    np.ndarray
        Potential energy in kJ/mol at each bin.
    """
    th = np.deg2rad(np.asarray(bins, dtype=float))
    y = np.full_like(th, model["offset_1pluscos"], dtype=float)
    for n, k, ddeg in zip(model["mults"], model["amps"], model["phases_deg"]):
        y += k * (1.0 + np.cos(n * th - np.deg2rad(ddeg)))
    return y


def report_potentials(best, bins=None, energy_kj=None, weights=None,
                      phase_range="-180_180", sort_by="multiplicity", decimals=3, return_pots=False):
    """
    Format each fitted term as: refdegree, k, multiplicity
      - refdegree: δ (degrees) in 1+cos(nθ - δ)
      - k: amplitude in kJ/mol
      - multiplicity: integer n
    Prints to stdout, or returns as a list of strings if return_pots=True.
    If bins and energy_kj are provided, appends RMSE to the first line.
    If `weights` is provided, computes weighted RMSE instead.
    """
    mults = best["mults"]
    ks = best["amps"]               # k in kJ/mol
    phases = best["phases_deg"]     # δ in degrees

    # normalize phase to the requested range
    def norm_phase(d):
        if phase_range == "-180_180":
            return ((d + 180.0) % 360.0) - 180.0
        elif phase_range == "0_360":
            return d % 360.0
        else:
            return d

    rows = [(norm_phase(d), float(k), int(n)) for n, k, d in zip(mults, ks, phases)]

    # optional sorting
    if sort_by == "multiplicity":
        rows.sort(key=lambda r: r[2])
    elif sort_by == "k_desc":
        rows.sort(key=lambda r: r[1], reverse=True)
    elif sort_by == "refdegree":
        rows.sort(key=lambda r: r[0])
    
    # build lines
    lines = []
    for refdeg, k, n in rows:
        lines.append(f"{refdeg:.{decimals}f} {k:.{decimals}f} {n}")
    
    # RMSE (unweighted or weighted) calculations.
    if bins is not None and energy_kj is not None:
        yhat = evaluate_model(bins, best)
        y = np.asarray(energy_kj, dtype=float)
        res = y - yhat
        pot_e = float(np.sum(ks))  # Σ k
        pot_e = pot_e if np.isfinite(pot_e) and pot_e > 0 else None
    
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
            rmse = float(np.sqrt(np.sum(w * res**2) / np.sum(w))) if np.any(w) else float(np.sqrt(np.mean(res**2)))
            tag = " (weighted)" if np.any(w) else ""
        else:
            rmse = float(np.sqrt(np.mean(res**2)))
            tag = ""
            
        suffix = f"; RMSE {rmse:.3f} kJ/mol{tag}"
    
        if pot_e:
            suffix += f" ({rmse/pot_e:.3f} RMSE/POT_E)"
    
        if len(lines) == 0:
            lines.append(suffix.lstrip("; ").strip())
        else:
            lines[0] = f"{lines[0]} {suffix}"

    if return_pots:
        return lines
    else:
        for line in lines:
            print(line)
