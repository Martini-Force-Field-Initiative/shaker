### Dihedral Fitting Routines

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import warnings

def _design_matrix(theta_deg, mults):
    th = np.deg2rad(theta_deg)
    cols = [np.ones_like(th)]
    for n in mults:
        cols.append(np.cos(n*th))
        cols.append(np.sin(n*th))
    return np.column_stack(cols)

def _aic(rss, k, N):
    # Heuristic with (weighted) RSS; fine for model comparison here
    return N*np.log(rss / max(N,1)) + 2*k

def _bic(rss, k, N):
    return N*np.log(rss / max(N,1)) + k*np.log(max(N,1))

def _make_weights(y, weight_mode="boltzmann", weights=None,
                  w_temp_kj=2.5, inv_shift=0.5, inv_power=1.0):
    """
    Build positive weights emphasizing low energies.
    y: energy (kJ/mol), will be shifted so min=0 internally for weight formulas.
    """
    if weights is not None:
        w = np.asarray(weights, dtype=float)
    else:
        y0 = y - np.nanmin(y)
        if weight_mode == "boltzmann":
            # larger T_w -> flatter weights; smaller -> more emphasis on minima
            w = np.exp(-y0 / float(w_temp_kj))
        elif weight_mode == "inverse":
            # s prevents blow-up at 0; p controls steepness
            w = 1.0 / np.power(y0 + float(inv_shift), float(inv_power))
        elif weight_mode in (None, "none"):
            w = np.ones_like(y)
        else:
            raise ValueError("weight_mode must be 'boltzmann', 'inverse', 'none', or provide `weights`.")
    # Clean up
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if not np.any(w):
        w = np.ones_like(y)
    # Optional normalization so average weight = 1
    return w * (len(w) / np.sum(w))

def fit_periodic_harmonics(
    theta_deg, energy_kj,
    max_terms=4, max_multiplicity=4,
    criterion="AIC", zero_min=True, theta_grid=None,
    weight_mode="boltzmann", weights=None,
    w_temp_kj=2.5, inv_shift=0.5, inv_power=1.0
):
    """
    Weighted least squares fit of F(θ) (deg, kJ/mol) to a sum of up to 4 harmonics with integer multiplicities 1..4.
    Low-energy points get higher weight. `zero_min=True` forces min_θ F(θ)=0 after fitting.
    """
    theta_deg = np.asarray(theta_deg)
    energy_kj = np.asarray(energy_kj)

    msk = np.isfinite(theta_deg) & np.isfinite(energy_kj)
    x, y = theta_deg[msk], energy_kj[msk]
    if len(y) == 0:
        raise ValueError("No valid data points.")
    N = len(y)

    # Build weights
    w = _make_weights(y, weight_mode=weight_mode, weights=weights,
                      w_temp_kj=w_temp_kj, inv_shift=inv_shift, inv_power=inv_power)
    w = w[msk] if weights is not None and len(weights) == len(energy_kj) else w
    sqrtw = np.sqrt(w)

    all_mults = list(range(1, max_multiplicity+1))
    best = None

    for r in range(1, max_terms+1):
        for mults in combinations(all_mults, r):
            X = _design_matrix(x, mults)
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
                best = {
                    "mults": mults,
                    "beta": beta,
                    "c0": c0,
                    "amps": np.array(amps),
                    "phases_deg": np.array(phases),
                    "offset_1pluscos": c0 - np.sum(amps),  # before zeroing min
                    "rss_w": rss_w,
                    "score": score,
                    "criterion": criterion.upper(),
                    "weights_summary": dict(mode=weight_mode, Tw=w_temp_kj, inv_shift=inv_shift, inv_power=inv_power)
                }

    # Enforce global min = 0 by adjusting constant
    if zero_min and best is not None:
        if theta_grid is None:
            theta_grid = np.linspace(-180, 180, 2001)
        th = np.deg2rad(theta_grid)
        g = np.zeros_like(th, dtype=float)
        for n, k, ddeg in zip(best["mults"], best["amps"], best["phases_deg"]):
            g += k * (1.0 + np.cos(n*th - np.deg2rad(ddeg)))
        c_zero = -np.min(g)
        best["offset_1pluscos"] = c_zero
        best["theta_grid_for_min"] = theta_grid
        best["min_value_check"] = float(np.min(c_zero + g))

    return best

def evaluate_model(theta_deg, model, form="one_plus_cos"):
    th = np.deg2rad(np.asarray(theta_deg))
    mults = model["mults"]
    if form == "cos_sin":
        X = _design_matrix(np.asarray(theta_deg), mults)
        return X @ model["beta"]
    y = np.full_like(th, model["offset_1plus_cos"] if "offset_1plus_cos" in model else model["offset_1pluscos"], dtype=float)
    for n, k, ddeg in zip(mults, model["amps"], model["phases_deg"]):
        y += k * (1.0 + np.cos(n*th - np.deg2rad(ddeg)))
    return y

def report_potentials(best, theta_deg=None, energy_kj=None, weights=None,
                      phase_range="-180_180", sort_by="multiplicity", decimals=3, return_pots=False):
    """
    Print each fitted term as: refdegree, k, multiplicity
      - refdegree: δ (degrees) in 1+cos(nθ - δ)
      - k: amplitude in kJ/mol
      - multiplicity: integer n
    Then prints RMSE (kJ/mol). If `weights` is provided, prints weighted RMSE.
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

    lines = []
    
    # build lines (no printing here)
    for refdeg, k, n in rows:
        lines.append(f"{refdeg:.{decimals}f} {k:.{decimals}f} {n}")
    
    # ---- RMSE (unweighted or weighted) ----
    if theta_deg is not None and energy_kj is not None:
        yhat = evaluate_model(theta_deg, best, form="one_plus_cos")
        y = np.asarray(energy_kj, dtype=float)
        res = y - yhat
    
        pot_e = float(np.sum(ks))  # Σ k
        pot_e = pot_e if np.isfinite(pot_e) and pot_e > 0 else None
    
        if weights is None or (isinstance(weights, np.ndarray) and not np.any(np.isfinite(weights))):
            rmse = float(np.sqrt(np.mean(res**2)))
            if pot_e:
                rmse_ks = rmse / pot_e
                suffix = f"; RMSE {rmse:.3f} kJ/mol ({rmse_ks:.3f} RMSE/POT_E)"
            else:
                suffix = f"; RMSE {rmse:.3f} kJ/mol"
        else:
            w = np.asarray(weights, dtype=float)
            w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
            if np.any(w):
                rmse_w = float(np.sqrt(np.sum(w * res**2) / np.sum(w)))
                if pot_e:
                    rmse_ks = rmse_w / pot_e
                    suffix = f"; RMSE {rmse_w:.3f} kJ/mol (weighted) ({rmse_ks:.3f} RMSE/POT_E)"
                else:
                    suffix = f"; RMSE {rmse_w:.3f} kJ/mol (weighted)"
            else:
                rmse = float(np.sqrt(np.mean(res**2)))
                if pot_e:
                    rmse_ks = rmse / pot_e
                    suffix = f"; RMSE {rmse:.3f} kJ/mol ({rmse_ks:.3f} RMSE/POT_E)"
                else:
                    suffix = f"; RMSE {rmse:.3f} kJ/mol"
    
        # append RMSE info to the FIRST line
        if len(lines) == 0:
            lines.append(suffix.lstrip("; ").strip())
        else:
            lines[0] = f"{lines[0]} {suffix}"

    if return_pots:
        return lines
    else:
        # later: print from the list
        for line in lines:
            print(line)

    
### InvBoltz and Savitzky
def inverted_boltzmann(data_density, kcal=False, temp=298, interpolate=False, penalty=0.0):
    # normalise densities with respect to themselves
    data_density = data_density / np.sum(data_density, axis=0)

    # calculate energy in kT and shift them so that the min is 0    
    with warnings.catch_warnings(): #Don't worry about log(0) = -inf, it will be converted to NaN later.
        warnings.simplefilter("ignore", RuntimeWarning) 
        data_energy = -np.log(data_density)
    data_energy -= np.nanmin(data_energy, axis=0)

    # convert to energy units
    if kcal:  # kcal/mol
        data_energy *= 8.3144621 * temp / float(1000 * 4.184)
    else:  # kJ/mol
        data_energy *= 8.3144621 * temp / float(1000)

    # change infinite values to NaN
    data_energy[np.isinf(data_energy)] = np.nan

    if interpolate:
        # linear interpolation over NaNs
        x = np.arange(len(data_energy))
        mask = np.isnan(data_energy)
        if np.any(mask) and np.any(~mask):  # only interpolate if there are valid values
            interp_vals = np.interp(x[mask], x[~mask], data_energy[~mask])
            data_energy[mask] = interp_vals + penalty  # add penalty here

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
    import numpy as np
    from math import factorial
    
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
    b = np.asmatrix([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

# ---- RB helpers ----
# For the fitting of Ryckhaert Bellemena dihedrals.

def _rb_design_matrix(theta_deg, max_power=5, shift_pi=True):
    """
    Build design matrix for Ryckaert–Bellemans:
      V(θ) = sum_{n=0..max_power} c_n * cos^n(ψ)
    where ψ = θ - 180° (i.e., θ - pi) if shift_pi=True (GROMACS convention).
    """
    th = np.deg2rad(np.asarray(theta_deg, dtype=float))
    psi = th - np.pi if shift_pi else th
    x = np.cos(psi)
    return np.column_stack([x**n for n in range(max_power + 1)])  # (N, max_power+1)

def fit_rb_pmf(
    theta_deg, energy_kj,
    max_power=5,                 # 5 -> classic RB (c0..c5)
    criterion=None,              # None -> no model selection; or "AIC"/"BIC" if you want truncation
    allow_truncation=False,      # if True, test powers 1..max_power and pick best by AIC/BIC
    zero_min=True, theta_grid=None,
    shift_pi=True,               # GROMACS RB uses psi=phi-pi
    weight_mode="boltzmann", weights=None,
    w_temp_kj=2.5, inv_shift=0.5, inv_power=1.0
):
    """
    Weighted LS fit of energy_kj(theta_deg) to RB:
      V = sum_{n=0..p} c_n cos^n(psi) , psi=theta-pi (default)
    Returns dict with c0..cp, RSS, score, and a min=0 shift (if requested).
    """
    theta_deg = np.asarray(theta_deg, dtype=float)
    energy_kj = np.asarray(energy_kj, dtype=float)

    msk = np.isfinite(theta_deg) & np.isfinite(energy_kj)
    x, y = theta_deg[msk], energy_kj[msk]
    if len(y) == 0:
        raise ValueError("No valid data points.")
    N = len(y)

    # weights (optionally from counts etc.)
    w_base = _make_weights(y, weight_mode=weight_mode, weights=weights,
                           w_temp_kj=w_temp_kj, inv_shift=inv_shift, inv_power=inv_power)
    if weights is not None and len(np.asarray(weights)) == len(energy_kj):
        w = np.asarray(weights, dtype=float)[msk]
        # if user passed weights explicitly, just sanitize + normalize like your code
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if not np.any(w):
            w = w_base
        else:
            w = w * (len(w) / np.sum(w))
    else:
        w = w_base

    sqrtw = np.sqrt(w)

    def solve_for_power(p):
        X = _rb_design_matrix(x, max_power=p, shift_pi=shift_pi)
        Xw = X * sqrtw[:, None]
        yw = y * sqrtw
        c, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        resid = y - X @ c
        rss_w = float(np.sum(w * resid**2))
        return c, rss_w, X.shape[1]

    # either fit fixed max_power, or do truncation/model selection
    if allow_truncation:
        if criterion is None:
            criterion = "AIC"
        best = None
        for p in range(1, max_power + 1):
            c, rss_w, k_params = solve_for_power(p)
            score = _aic(rss_w, k_params, N) if criterion.upper() == "AIC" else _bic(rss_w, k_params, N)
            if (best is None) or (score < best["score"]):
                best = dict(p=p, c=c, rss_w=rss_w, score=score, criterion=criterion.upper())
    else:
        c, rss_w, k_params = solve_for_power(max_power)
        best = dict(p=max_power, c=c, rss_w=rss_w, score=None, criterion=None)

    # Enforce global min = 0 by shifting c0 (constant term)
    if zero_min:
        if theta_grid is None:
            theta_grid = np.linspace(-180, 180, 4001)
        Vg = evaluate_rb(theta_grid, best["c"], shift_pi=shift_pi)
        shift = float(np.min(Vg))
        best["c"] = best["c"].copy()
        best["c"][0] -= shift
        best["min_shift_applied"] = shift
        best["theta_grid_for_min"] = theta_grid
        best["min_value_check"] = float(np.min(evaluate_rb(theta_grid, best["c"], shift_pi=shift_pi)))

    # pack friendly output similar-ish to yours
    best["c0_to_cN"] = best["c"]
    best["powers"] = tuple(range(len(best["c"])))
    best["shift_pi"] = bool(shift_pi)
    best["weights_summary"] = dict(mode=weight_mode, Tw=w_temp_kj, inv_shift=inv_shift, inv_power=inv_power)
    best["N"] = int(N)

    return best


def evaluate_rb(theta_deg, c, shift_pi=True):
    """Evaluate RB at theta_deg for coefficients c0..cP."""
    theta_deg = np.asarray(theta_deg, dtype=float)
    X = _rb_design_matrix(theta_deg, max_power=len(c) - 1, shift_pi=shift_pi)
    return X @ np.asarray(c, dtype=float)


def report_rb(best, decimals=6):
    c = np.asarray(best["c"], dtype=float)
    lines = []
    for i, ci in enumerate(c):
        lines.append(f"c{i} = {ci:.{decimals}f}  # kJ/mol")
    return "\n".join(lines)
