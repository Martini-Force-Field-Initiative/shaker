import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150

'''
Collection of functions to assist in matplotlib plotting.
'''

def plot_sasa_dir(root="./SASA",
                  xvg="resarea_SASA.xvg"):
    '''
    Plot SASA values from multiple simulations stored in subdirectories.

    This function scans the specified directory for subdirectories containing
    a SASA results file (e.g. `resarea_SASA.xvg`). Each subdirectory is assumed
    to represent a different model or simulation condition. The SASA values
    and associated errors are extracted and displayed as a bar plot for
    comparison.

    Parameters
    ----------
    root : str or Path, optional
        Directory containing subdirectories with SASA output files.
        Default is "./SASA".
    xvg : str, optional
        Name of the GROMACS `.xvg` file containing SASA values within each
        subdirectory. Default is "resarea_SASA.xvg".

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    ax : matplotlib.axes.Axes
        The axes containing the bar plot.
    items : list of tuple
        List containing the parsed SASA data in the form
        `(name, value, error)` for each subdirectory.

    Notes
    -----
    The function expects each subdirectory inside `root` to contain the
    specified `.xvg` file. The label used in the bar plot corresponds to
    the name of the subdirectory.

    The resulting figure is saved as `SASABar.png`.
    '''
    root = Path(root)
    items = [(d.name, *_read_SASA_xvg(d / xvg))
             for d in sorted(root.iterdir())
             if d.is_dir() and (d / xvg).exists()]

    names, vals, errs = zip(*items)

    x = np.arange(len(names))
    fig, ax = plt.subplots(
        figsize=(max(6, 0.8*len(names)), 4),
        tight_layout=True)

    ax.bar(x, vals, yerr=errs, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("SASA (nm$^2$)", fontweight='bold')
    fig.savefig("SASABar.png", dpi=300, transparent=True, bbox_inches='tight')

    return fig, ax, items


def plot_bonded_distributions(
    *bonded_dicts,
    labels=None,
    colors=None,
    outfile="cleanbonds",
    transparent=True,
    show_peaks=True,
):
    """
    Plot bonded distributions (distances, angles, dihedrals) from one or more
    bonded dictionaries.

    Parameters
    ----------
    *bonded_dicts : dict
        Any number of bonded dictionaries with structure like:
        {
            "distances": {"targets": ..., "bins": ..., "hist": ...},
            "angles": {"targets": ..., "bins": ..., "hist": ...},
            "dihedrals": {"targets": ..., "bins": ..., "hist": ...},
        }

    labels : list[str] | None, optional
        Labels for each bonded dictionary. If None, uses Dataset 1, Dataset 2, ...

    colors : list[str] | None, optional
        Line colors for each bonded dictionary. If None, matplotlib default cycle is used.

    outfile : str | None, optional
        Output filename stem. If provided, saves <outfile>.svg/.pdf/.png.
        If None, nothing is saved.

    transparent : bool, optional
        Whether to save figures with transparent background.

    show_peaks : bool, optional
        Whether to annotate the peak position of each histogram.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.

    Raises
    ------
    ValueError
        If no dictionaries are provided, or if targets do not match across inputs.
    """
    
    if len(bonded_dicts) == 0:
        raise ValueError("At least one bonded dictionary must be provided.")

    categories = ("distances", "angles", "dihedrals")

    # Check that all dictionaries share the same targets
    ref = bonded_dicts[0]
    for cat in categories:
        ref_targets = ref[cat]["targets"]
        for idx, bonded in enumerate(bonded_dicts[1:], start=1):
            if bonded[cat]["targets"] != ref_targets:
                raise ValueError(f"Mismatch in '{cat}' targets between input 0 and input {idx}.")

    # Labels
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(bonded_dicts))]
    if len(labels) != len(bonded_dicts):
        raise ValueError("Length of 'labels' must match number of bonded dictionaries.")

    # Colors
    if colors is None:
        colors = [None] * len(bonded_dicts)
    if len(colors) != len(bonded_dicts):
        raise ValueError("Length of 'colors' must match number of bonded dictionaries.")

    # Layout helpers from shaker
    grid_dist  = _best_grid(len(ref["distances"]["targets"]))
    grid_ang   = _best_grid(len(ref["angles"]["targets"]))
    grid_dihed = _best_grid(len(ref["dihedrals"]["targets"]))

    figsize = _predict_figsize([grid_dist, grid_ang, grid_dihed])
    height_ratios = [grid_dist[0], grid_ang[0], grid_dihed[0]]

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=height_ratios, hspace=1)

    config = {
        "distances": {"grid": grid_dist,
                      "subplot": gs[0],
                      "xlabel": "Bond (Å)",},
        "angles": {"grid": grid_ang,
                   "subplot": gs[1],
                   "xlabel": "Angle (°)",},
        "dihedrals": {"grid": grid_dihed,
                      "subplot": gs[2],
                      "xlabel": "Dihedral angle (°)",},
        }

    for cat in categories:
        targets = ref[cat]["targets"]
        subgrid = gridspec.GridSpecFromSubplotSpec(
            config[cat]["grid"][0],
            config[cat]["grid"][1],
            subplot_spec=config[cat]["subplot"],
        )

        for i, distribution in enumerate(targets):
            ax = fig.add_subplot(subgrid[i])

            for bonded, label, color in zip(bonded_dicts, labels, colors):
                bins = bonded[cat]["bins"]
                hist = bonded[cat]["hist"][i]

                ax.plot(bins, hist, label=label, color=color)

                if show_peaks and len(hist) > 0:
                    peak_idx = np.argmax(hist)
                    ax.text(
                        bins[peak_idx],
                        hist[peak_idx],
                        f"{bins[peak_idx]:.2f}",
                        color=color)

            ax.set_title("-".join(distribution), fontweight="bold")
            ax.legend(loc="upper right", frameon=False, fontsize=4)
            ax.set_xlabel(config[cat]["xlabel"], fontsize=8)
            ax.set_ylabel("Prob. density")

    if outfile is not None:
        fig.savefig(f"{outfile}.svg", transparent=transparent)
        fig.savefig(f"{outfile}.pdf", transparent=transparent)
        fig.savefig(f"{outfile}.png", transparent=transparent, dpi=300)

    return fig


def _read_SASA_xvg(xvg):
    '''
    Reader for the SASA per residue .xvg file. Retrieves AVG and Std.
    '''
    rows = [l.split() for l in Path(xvg).read_text().splitlines()
            if l and l[0] not in "#@"]
    a = np.array(rows, float)
    return a[0, 1], a[0, 2]   


def _best_grid(n: int):
    """Return (nrows, ncols) for n plots using a near-square layout."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def _predict_figsize(grids, cell_w=3.2, cell_h=1.5, section_gap_h=0.6,
                    left=0.8, right=0.2, top=0.6, bottom=0.6):
    """
    grids: list of (nrows, ncols) for sections stacked vertically
    returns (fig_w, fig_h) in inches
    """
    max_cols = max(c for r, c in grids)
    total_rows = sum(r for r, c in grids)

    fig_w = left + max_cols * cell_w + right
    fig_h = bottom + total_rows * cell_h + top + section_gap_h * (len(grids) - 1)
    return fig_w, fig_h
        