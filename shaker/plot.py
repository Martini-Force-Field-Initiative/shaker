"""Publication-quality distribution and SASA plots."""

import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import warnings
from scipy.stats import wasserstein_distance

mpl.rcParams['figure.dpi'] = 150

# NumPy 2.0 renamed trapz -> trapezoid
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

def plot_sasa_dir(root="./SASA",
                  xvg="resarea_SASA.xvg"):
    '''
    Plot SASA values from multiple simulations stored in subdirectories.

    This function scans the specified directory for subdirectories containing
    a SASA results file (e.g. `resarea_SASA.xvg`). Each subdirectory is assumed
    to represent a different model or simulation condition. The SASA values
    and associated errors are extracted and displayed as a bar plot for
    comparison.

    If a subdirectory named "AA" is present, coloured background bands are
    drawn to indicate the percentage deviation from the AA reference value:
    green (0–5 %), orange (5–10 %), and red (>10 %, extending to zero).
    A dashed horizontal line marks the AA reference value. A legend is added
    explaining the colour coding.

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

    The y-axis runs from 0 to 10 % above the largest value + error across
    all datasets.

    The resulting figure is saved as `SASABar.png`.
    '''
    root = Path(root)
    items = [(d.name, *_read_SASA_xvg(d / xvg))
             for d in sorted(root.iterdir())
             if d.is_dir() and (d / xvg).exists()]
    names, vals, errs = zip(*items)
    x = np.arange(len(names))

    fig, ax = plt.subplots(
        figsize=(max(6, 0.8 * len(names)), 4),
        tight_layout=True)

    # y-axis limits
    y_max = max(v + e for v, e in zip(vals, errs))
    ax.set_ylim(0, y_max * 1.10)

    # AA reference bands
    aa_val = None
    if "AA" in names:
        aa_val = vals[names.index("AA")]

    legend_patches = []
    if aa_val is not None:
        bands = [
            (0.00, 0.05, "#2ecc71", "0–5 % from AA"),
            (0.05, 0.10, "#e67e22", "5–10 % from AA"),
            (0.10, None, "#e74c3c", ">10 % from AA"),
        ]
        for lo, hi, color, label in bands:
            if hi is None:
                ax.axhspan(0, aa_val * (1 - lo),
                           color=color, alpha=0.15, zorder=0)
                ax.axhspan(aa_val * (1 + lo), y_max * 1.10,
                           color=color, alpha=0.15, zorder=0)
            else:
                ax.axhspan(aa_val * (1 - hi), aa_val * (1 - lo),
                           color=color, alpha=0.15, zorder=0)
                ax.axhspan(aa_val * (1 + lo), aa_val * (1 + hi),
                           color=color, alpha=0.15, zorder=0)

            legend_patches.append(
                mpatches.Patch(facecolor=color, alpha=0.4,
                               edgecolor="none", label=label))

        ax.axhline(aa_val, color="dimgrey", lw=1.2,
                   ls="--", zorder=1, alpha=0.7)
        legend_patches.insert(0,
            mlines.Line2D([], [], color="dimgrey", lw=1.2,
                          ls="--", alpha=0.7, label="AA reference"))

    # bars
    ax.bar(x, vals,
           yerr=errs,
           capsize=7,
           error_kw=dict(elinewidth=1.8, ecolor="dimgrey", capthick=1.8),
           color="#888888",
           edgecolor="#bbbbbb",
           linewidth=1.2,
           zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontweight="bold")
    ax.set_ylabel("SASA (nm$^2$)", fontweight="bold")

    if legend_patches:
        ax.legend(handles=legend_patches, loc="lower left",
                  framealpha=0.8, fontsize=8)

    fig.savefig("SASABar.png", dpi=300, transparent=True, bbox_inches="tight")
    return fig, ax, items


def plot_bonded_distributions(*bonded_dicts,
                              labels=None, colors=None,
                              outfile="cleanbonds", transparent=True,
                              show_peaks=False, metrics=True):
    """
    Plot bonded distributions (distances, angles, dihedrals) from one or more
    bonded dictionaries.

    Parameters
    ----------
    *bonded_dicts : dict
        Any number of bonded dictionaries with structure like::

            {
                "distances": {"targets": ..., "bins": ..., "hist": ...},
                "angles":    {"targets": ..., "bins": ..., "hist": ...},
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

    metrics : bool, optional
        Whether to compute and display Wasserstein distance and overlap coefficient
        for each distribution against the reference (first) dataset, and shade
        the overlapping region between curves. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.

    Raises
    ------
    ValueError
        If no dictionaries are provided, if targets do not match across inputs,
        or if all categories are empty.
    """

    if len(bonded_dicts) == 0:
        raise ValueError("At least one bonded dictionary must be provided.")

    if metrics and len(bonded_dicts) == 1:
        warnings.warn(
            "metrics=True has no effect with a single dataset — nothing to compare against."
        )

    categories = ("distances", "angles", "dihedrals")

    # Check that all dictionaries share the same targets
    ref = bonded_dicts[0]
    for cat in categories:
        ref_targets = ref[cat]["targets"]
        for idx, bonded in enumerate(bonded_dicts[1:], start=1):
            if bonded[cat]["targets"] != ref_targets:
                raise ValueError(
                    f"Mismatch in '{cat}' targets between input 0 and input {idx}."
                )

    # Labels
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(bonded_dicts))]
    if len(labels) != len(bonded_dicts):
        raise ValueError("Length of 'labels' must match number of bonded dictionaries.")

    # Colors — resolve None entries against the default matplotlib color cycle
    prop_cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    resolved_colors = [
        c if c is not None else prop_cycle_colors[i % len(prop_cycle_colors)]
        for i, c in enumerate(colors or [None] * len(bonded_dicts))
    ]
    if len(resolved_colors) != len(bonded_dicts):
        raise ValueError("Length of 'colors' must match number of bonded dictionaries.")

    # Filter out empty categories, warning the user for each one skipped
    active_categories = []
    for cat in categories:
        if len(ref[cat]["targets"]) == 0:
            warnings.warn(f"No targets found for '{cat}' — skipping this block.")
        else:
            active_categories.append(cat)

    if not active_categories:
        raise ValueError(
            "All categories (distances, angles, dihedrals) are empty — nothing to plot."
        )

    xlim_map = {
        "distances": (1.5, 5.5),
        "angles": (0, 180),
        "dihedrals": (-180, 180),
    }

    # Layout helpers
    grids = {cat: _best_grid(len(ref[cat]["targets"])) for cat in active_categories}
    figsize = _predict_figsize(list(grids.values()))
    height_ratios = [grids[cat][0] for cat in active_categories]

    # Interleave spacer rows between category sections so sections are
    # visually separated. constrained_layout ignores hspace on the outer
    # GridSpec, so we insert explicit zero-height spacer rows instead.
    n_cats = len(active_categories)
    spacer_height = 0.3   # inches — tune this for more/less gap
    cell_h = 1.5          # must match _predict_figsize
    spacer_ratio = spacer_height / cell_h

    interleaved_ratios = []
    for k, r in enumerate(height_ratios):
        interleaved_ratios.append(r)
        if k < n_cats - 1:
            interleaved_ratios.append(spacer_ratio)

    n_outer_rows = len(interleaved_ratios)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(
        n_outer_rows, 1,
        figure=fig,
        height_ratios=interleaved_ratios,
    )

    # Add invisible spacer axes so constrained_layout respects the rows
    for k in range(1, n_outer_rows, 2):
        spacer_ax = fig.add_subplot(gs[k])
        spacer_ax.set_visible(False)

    xlabel_map = {
        "distances": "Bond (Å)",
        "angles": "Angle (°)",
        "dihedrals": "Dihedral angle (°)",
    }

    config = {
        cat: {
            "grid": grids[cat],
            "subplot": gs[i * 2],   # every other row; odd rows are spacers
            "xlabel": xlabel_map[cat],
        }
        for i, cat in enumerate(active_categories)
    }

    for cat in active_categories:
        targets = ref[cat]["targets"]
        subgrid = gridspec.GridSpecFromSubplotSpec(
            config[cat]["grid"][0],
            config[cat]["grid"][1],
            subplot_spec=config[cat]["subplot"],
        )

        for i, distribution in enumerate(targets):
            ax = fig.add_subplot(subgrid[i])

            # Clean up spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Keep y-axis ticks readable at small subplot sizes
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))

            ref_bins = ref[cat]["bins"]
            ref_hist = np.asarray(ref[cat]["hist"][i])

            # plot reference (dataset 0)
            ax.plot(ref_bins, ref_hist, label=labels[0], color=resolved_colors[0])

            if show_peaks and len(ref_hist) > 0:
                peak_idx = np.argmax(ref_hist)
                ax.text(
                    ref_bins[peak_idx], ref_hist[peak_idx],
                    f"{ref_bins[peak_idx]:.2f}",
                    color=resolved_colors[0],
                    fontsize=6, va="bottom", ha="center",
                )

            # plot subsequent datasets, optionally compute metrics vs reference 
            # Each entry is (text, color) so each comparison gets its own colour.
            metrics_lines = []

            for j, (bonded, label, color) in enumerate(
                zip(bonded_dicts[1:], labels[1:], resolved_colors[1:]), start=1
            ):
                bins = bonded[cat]["bins"]
                hist = np.asarray(bonded[cat]["hist"][i])

                ax.plot(bins, hist, label=label, color=color)

                if show_peaks and len(hist) > 0:
                    peak_idx = np.argmax(hist)
                    ax.text(
                        bins[peak_idx], hist[peak_idx],
                        f"{bins[peak_idx]:.2f}",
                        color=color,
                        fontsize=6, va="bottom", ha="center",
                    )

                if metrics:
                    overlap_y = np.minimum(ref_hist, hist)

                    # Shade the overlapping region (kept out of the legend —
                    # it's visually self-evident, and an entry per comparison
                    # dataset would make the legend grow unboundedly).
                    ax.fill_between(
                        ref_bins, overlap_y,
                        alpha=0.15,
                        color=color,
                    )

                    # Wasserstein distance
                    w_dist = wasserstein_distance(ref_bins, bins, ref_hist, hist)

                    # Overlap coefficient: ∫ min(p, q) dx
                    # For normalized densities this equals the histogram intersection.
                    oc = _trapz(overlap_y, ref_bins)

                    if oc >= 0.8:
                        line_color, status = "green", "[GOOD]"
                    elif oc >= 0.65:
                        line_color, status = "orange", "[WARN]"
                    else:
                        line_color, status = "red", "[POOR]"

                    metrics_lines.append(
                        (f"{label}  W={w_dist:.3f}  OC={oc:.3f}  {status}", line_color)
                    )

            # Annotate metrics in the upper-right corner, one text call per line
            # so each comparison carries its own colour.
            if metrics and metrics_lines:
                line_height = 0.10
                for k, (line, line_color) in enumerate(metrics_lines):
                    ax.text(
                        0.98, 0.95 - k * line_height,
                        line,
                        transform=ax.transAxes,
                        fontsize=5, fontweight="bold",
                        va="top", ha="right",
                        family="monospace",
                        color=line_color,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, lw=0),
                    )

            # --- axis limits ---
            all_hists = [np.asarray(bonded[cat]["hist"][i]) for bonded in bonded_dicts]
            valid_maxes = [h.max() for h in all_hists if len(h) > 0 and h.max() > 0]
            y_max = max(valid_maxes) if valid_maxes else 1.0
            ax.set_ylim(0, y_max * 1.2)
            ax.set_xlim(xlim_map[cat])

            ax.set_title("-".join(distribution), fontweight="bold")
            ax.legend(loc="upper left", fontsize=5, ncols=1, labelspacing=0.3,
                     frameon=True, fancybox=True, edgecolor="none",
                     facecolor="white", framealpha=0.8)
            ax.set_xlabel(config[cat]["xlabel"], fontsize=9)
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