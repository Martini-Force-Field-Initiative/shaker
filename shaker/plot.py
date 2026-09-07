"""Publication-quality distribution and SASA plots."""

import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.patches as mpatches
import warnings
from scipy.stats import wasserstein_distance

mpl.rcParams['figure.dpi'] = 150

# NumPy 2.0 renamed trapz -> trapezoid
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

def plot_sasa_dir(root="./SASA",
                  xvg="SASA.xvg",
                  kind="violin"):
    '''
    Plot SASA values from multiple simulations stored in subdirectories.

    This function scans the specified directory for subdirectories
    containing a per-frame SASA time series (`xvg`, written by `run_SASA`
    via `gmx sasa -o`). Each subdirectory is assumed to represent a
    different model or simulation condition. Mean and std (used for "bar"
    and the returned summary values) are computed from this same per-frame
    data for every `kind`, so all three chart styles — and the numbers
    returned alongside them — are derived the same way.

    Three chart styles are available via `kind`:

    - "bar" — mean ± std as a bar per subdirectory.
    - "violin" (default) — same layout as "bar", but each column is a
      violin showing the full per-frame spread instead of just mean ± std.
    - "overlay" — per-frame SASA distributions overlaid as density curves
      (same visual style as `plot_bonded_distributions`).

    If a subdirectory named "AA" is present, coloured background bands are
    drawn to indicate the percentage deviation from the AA reference value:
    green (0-5 %), orange (5-10 %), and red (>10 %). A dashed line marks the
    AA reference value. For "bar"/"violin" these are horizontal (SASA is the
    y-axis); for "overlay" they're vertical (SASA is the x-axis there).

    Parameters
    ----------
    root : str or Path, optional
        Directory containing subdirectories with SASA output files.
        Default is "./SASA".
    xvg : str, optional
        Name of the per-frame GROMACS `.xvg` file (`gmx sasa -o`) within
        each subdirectory. Default is "SASA.xvg".
    kind : {"bar", "violin", "overlay"}, optional
        Chart style, see above. Default is "violin".

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    items : list of tuple
        `(name, value, error)` per subdirectory — mean and std SASA over
        all frames, sorted by subdirectory name.

    Notes
    -----
    The y-axis for "bar"/"violin" runs from 0 to 10 % above the largest
    value (+ error, for "bar") across all datasets.

    The resulting figure is saved as `SASABar.png`.
    '''
    if kind not in ("bar", "violin", "overlay"):
        raise ValueError("kind must be 'bar', 'violin', or 'overlay'")

    root = Path(root)

    if kind == "overlay":
        return _plot_sasa_overlay(root, xvg)

    entries = [(d.name, _read_SASA_timeseries(d / xvg))
              for d in sorted(root.iterdir())
              if d.is_dir() and (d / xvg).exists()]
    if not entries:
        raise ValueError(f"No '{xvg}' files found under {root}")

    names = [name for name, _ in entries]
    distributions = [dist for _, dist in entries]
    vals = [dist.mean() for dist in distributions]
    errs = [dist.std() for dist in distributions]
    items = list(zip(names, vals, errs))
    x = np.arange(len(names))

    fig, ax = plt.subplots(
        figsize=(max(6, 0.8 * len(names)), 4),
        tight_layout=True)

    # AA reference value, used both for the deviation bands below and (for
    # "violin") as the center of the y-axis window.
    aa_val = None
    if "AA" in names:
        aa_val = vals[names.index("AA")]

    # y-axis limits: bars start from zero so the eye can compare magnitude.
    # A violin has no such baseline — default to mean ± 12.5 % around the
    # AA reference (a little past the outermost ">10 %" deviation band),
    # but widen to the actual data range if any distribution extends
    # beyond that window, so nothing gets clipped.
    if kind == "violin":
        ref_mean = aa_val if aa_val is not None else float(np.mean(vals))
        window = ref_mean * 0.125
        dist_min = min(dist.min() for dist in distributions)
        dist_max = max(dist.max() for dist in distributions)
        y_min = min(ref_mean - window, dist_min)
        y_top = max(ref_mean + window, dist_max)
    else:
        y_min = 0
        y_top = max(v + e for v, e in zip(vals, errs)) * 1.10
    ax.set_ylim(y_min, y_top)

    legend_patches = []
    if aa_val is not None:
        bands = [
            (0.00, 0.05, "#2ecc71", "0–5 % from AA"),
            (0.05, 0.10, "#e67e22", "5–10 % from AA"),
            (0.10, None, "#e74c3c", ">10 % from AA"),
        ]
        for lo, hi, color, label in bands:
            if hi is None:
                ax.axhspan(y_min, aa_val * (1 - lo),
                           color=color, alpha=0.15, zorder=0)
                ax.axhspan(aa_val * (1 + lo), y_top,
                           color=color, alpha=0.15, zorder=0)
            else:
                ax.axhspan(aa_val * (1 - hi), aa_val * (1 - lo),
                           color=color, alpha=0.15, zorder=0)
                ax.axhspan(aa_val * (1 + lo), aa_val * (1 + hi),
                           color=color, alpha=0.15, zorder=0)

            legend_patches.append(
                mpatches.Patch(facecolor=color, alpha=0.4,
                               edgecolor="none", label=label))

        # The dashed line's meaning is self-evident (it sits on the AA
        # bar/violin), so it doesn't get its own legend entry.
        ax.axhline(aa_val, color="dimgrey", lw=1.2,
                   ls="--", zorder=1, alpha=0.7)

    if kind == "bar":
        ax.bar(x, vals,
               yerr=errs,
               capsize=7,
               error_kw=dict(elinewidth=1.8, ecolor="dimgrey", capthick=1.8),
               color="#888888",
               edgecolor="#bbbbbb",
               linewidth=1.2,
               zorder=2)
    else:  # violin
        parts = ax.violinplot(distributions, positions=x,
                              showmeans=True, showextrema=True, widths=0.7)
        for body in parts["bodies"]:
            body.set_facecolor("#888888")
            body.set_edgecolor("#555555")
            body.set_alpha(0.7)
            body.set_zorder(2)
        for key in ("cbars", "cmins", "cmaxes", "cmeans"):
            parts[key].set_color("dimgrey")
            parts[key].set_zorder(2)
        for key in ("cmins", "cmaxes", "cmeans"):
            # Shrink the mean/min/max horizontal markers to half their
            # default width (which otherwise spans the full violin) so
            # they read as tick marks rather than bars.
            segments = []
            for (x0, y0), (x1, y1) in parts[key].get_segments():
                xc = (x0 + x1) / 2
                half = (x1 - x0) / 4
                segments.append([(xc - half, y0), (xc + half, y1)])
            parts[key].set_segments(segments)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontweight="bold")
    ax.set_ylabel("SASA (nm$^2$)", fontweight="bold")

    if legend_patches:
        ax.legend(handles=legend_patches, loc="lower left", fontsize=8,
                  labelspacing=0.3, frameon=True, fancybox=True,
                  edgecolor="none", facecolor="white", framealpha=0.8)

    fig.savefig("SASABar.png", dpi=300, transparent=True, bbox_inches="tight")
    return fig, ax, items


_SASA_OVERLAY_COLORS = ["tab:blue", "tab:red", "tab:green", "tab:orange",
                        "tab:purple", "tab:brown", "tab:pink", "tab:gray",
                        "tab:olive", "tab:cyan"]


def _plot_sasa_overlay(root, xvg, bins=60):
    '''
    Plot per-frame SASA distributions for every subdirectory in `root` as
    overlaid density curves, in the same visual style as
    `plot_bonded_distributions`. See `plot_sasa_dir` (kind="overlay").
    '''
    entries = [(d.name, _read_SASA_timeseries(d / xvg))
               for d in sorted(root.iterdir())
               if d.is_dir() and (d / xvg).exists()]
    if not entries:
        raise ValueError(f"No '{xvg}' files found under {root}")

    names = [name for name, _ in entries]
    means = {name: vals.mean() for name, vals in entries}
    all_vals = np.concatenate([vals for _, vals in entries])
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins + 1)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    hists = {name: np.histogram(vals, bins=bin_edges, density=True)[0]
             for name, vals in entries}

    ref_name = "AA" if "AA" in hists else names[0]
    ref_hist = hists[ref_name]

    # Reference (AA, if present) always gets tab:blue and the first
    # comparison dataset tab:red, matching plot_bonded_distributions'
    # convention, rather than whatever order subdirectories sort in.
    plot_order = [ref_name] + [n for n in names if n != ref_name]
    colors = {name: _SASA_OVERLAY_COLORS[i % len(_SASA_OVERLAY_COLORS)]
             for i, name in enumerate(plot_order)}

    # x-axis limits: default to mean ± 12.5 % around the reference (AA, or
    # the first dataset if there's no AA), widened to the actual data range
    # if any distribution extends beyond that window, so nothing is clipped.
    ref_mean = means[ref_name]
    window = ref_mean * 0.125
    x_min = min(ref_mean - window, all_vals.min())
    x_max = max(ref_mean + window, all_vals.max())

    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

    # AA reference bands, same deviation thresholds as plot_sasa_dir's
    # bar/violin charts, just as vertical spans since SASA is the x-axis
    # here instead of the y-axis.
    aa_val = means.get("AA")
    legend_patches = []
    if aa_val is not None:
        bands = [
            (0.00, 0.05, "#2ecc71", "0–5 % from AA"),
            (0.05, 0.10, "#e67e22", "5–10 % from AA"),
            (0.10, None, "#e74c3c", ">10 % from AA"),
        ]
        for lo, hi, color, label in bands:
            if hi is None:
                ax.axvspan(x_min, aa_val * (1 - lo),
                          color=color, alpha=0.15, zorder=0)
                ax.axvspan(aa_val * (1 + lo), x_max,
                          color=color, alpha=0.15, zorder=0)
            else:
                ax.axvspan(aa_val * (1 - hi), aa_val * (1 - lo),
                          color=color, alpha=0.15, zorder=0)
                ax.axvspan(aa_val * (1 + lo), aa_val * (1 + hi),
                          color=color, alpha=0.15, zorder=0)

            legend_patches.append(
                mpatches.Patch(facecolor=color, alpha=0.4,
                               edgecolor="none", label=label))

        # The dashed line's meaning is self-evident (it sits on the AA
        # curve), so it doesn't get its own legend entry.
        ax.axvline(aa_val, color="dimgrey", lw=1.2,
                  ls="--", zorder=1, alpha=0.7)

    curve_handles = ax.plot(bin_centers, ref_hist, label=ref_name,
                            color=colors[ref_name], zorder=2)

    for name in names:
        if name == ref_name:
            continue
        hist, color = hists[name], colors[name]
        curve_handles += ax.plot(bin_centers, hist, label=name, color=color, zorder=2)

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("SASA (nm$^2$)", fontweight="bold")
    ax.set_ylabel("Prob. density", fontweight="bold")

    ax.legend(handles=curve_handles + legend_patches,
             loc="upper left", fontsize=8, ncols=2, labelspacing=0.3,
             columnspacing=1.0, frameon=True, fancybox=True, edgecolor="none",
             facecolor="white", framealpha=0.8)

    fig.savefig("SASABar.png", dpi=300, transparent=True, bbox_inches="tight")

    items = [(name, float(means[name]), float(vals.std())) for name, vals in entries]
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


def _read_SASA_timeseries(xvg):
    '''
    Reader for the SASA-vs-time .xvg file (`gmx sasa -o`). Retrieves the
    per-frame total SASA values, ignoring the time column.
    '''
    rows = [l.split() for l in Path(xvg).read_text().splitlines()
            if l and l[0] not in "#@"]
    a = np.array(rows, float)
    return a[:, 1]


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