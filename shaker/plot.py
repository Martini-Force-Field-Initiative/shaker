"""Publication-quality distribution and SASA plots."""

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.backend_bases import FigureCanvasBase
from scipy.signal import find_peaks
from scipy.stats import wasserstein_distance

from .options import _resolve

mpl.rcParams["figure.dpi"] = 150

# NumPy 2.0 renamed trapz -> trapezoid
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz  # type: ignore[attr-defined]

# SASA deviation from AA: ✓ / green band up to _SASA_OK, ⚠ / orange up to
# _SASA_WARN, ✗ / red beyond (fractions of the AA value). The SASA axis spans
# at least ±_SASA_WARN around the reference.
_SASA_OK, _SASA_WARN = 0.05, 0.10

# Overlap-coefficient thresholds for GOOD (✓) / WARN (⚠); below is POOR (✗).
_OC_GOOD, _OC_WARN = 0.80, 0.65
# How each status is shown on the bonded plot: (colour, label).
_STATUS_STYLE = {
    "✓": ("green", "[GOOD]"),
    "⚠": ("orange", "[WARN]"),
    "✗": ("red", "[POOR]"),
}

# Default dataset colours, in order: reference, first comparison, second...
# Shared by plot_bonded_distributions and plot_sasa_dir(kind="overlay").
_COLORS = [
    "tab:blue",
    "tab:red",
    "tab:grey",
    "tab:green",
    "tab:orange",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
]

# Height of one subplot row in plot_bonded_distributions, in inches.
_CELL_H = 1.5

# Extensions matplotlib can write, e.g. {"svg", "png", "pdf", ...}.
_IMAGE_EXTS = FigureCanvasBase.get_supported_filetypes()


def plot_sasa_dir(
    root="./SASA",
    xvg="SASA.xvg",
    kind="violin",
    outfile="SASABar",
    transparent=True,
    show=None,
    verbose=None,
):
    """
    Plot SASA values from multiple simulations stored in subdirectories.

    This function scans the specified directory for subdirectories
    containing a per-frame SASA time series (`xvg`, written by `run_SASA`
    via `gmx sasa -o`). Each subdirectory is assumed to represent a
    different model or simulation condition. Mean and std (used for the
    returned summary values) are computed from this same per-frame data
    for every `kind`, so both chart styles — and the numbers returned
    alongside them — are derived the same way.

    Two chart styles are available via `kind`:

    - "violin" (default) — a violin per subdirectory, showing the full
      per-frame spread rather than just mean ± std.
    - "overlay" — per-frame SASA distributions overlaid as density curves
      (same visual style as `plot_bonded_distributions`).

    If a subdirectory named "AA" is present, coloured background bands are
    drawn to indicate the percentage deviation from the AA reference value:
    green (0-5 %), orange (5-10 %), and red (>10 %). A dashed line marks the
    AA reference value. For "violin" this is horizontal (SASA is the
    y-axis); for "overlay" it's vertical (SASA is the x-axis there).

    Parameters
    ----------
    root : str or Path, optional
        Directory containing subdirectories with SASA output files.
        Default is "./SASA".
    xvg : str, optional
        Name of the per-frame GROMACS `.xvg` file (`gmx sasa -o`) within
        each subdirectory. Default is "SASA.xvg".
    kind : {"violin", "overlay"}, optional
        Chart style, see above. Default is "violin".
    outfile : str or Path or None, optional
        Where to save the figure: as SVG (`<outfile>.svg`) unless it ends in
        another image extension, e.g. "sasa.png". Overwritten if it exists;
        None saves nothing. Default "SASABar".
    transparent : bool, optional
        Save with a transparent background. Default True.
    show : bool or None, optional
        Whether to display the figure in the notebook. None (default) uses
        the SHAKER-wide setting (see `set_options`), which is True unless
        changed. The figure is saved either way.
    verbose : bool or None, optional
        Whether to print a short report: mean ± std per subdirectory and, if
        "AA" is present, the difference from AA in nm² and %, marked ✓
        (within 5 %), ⚠ (5–10 %) or ✗ (>10 %), the same thresholds as the
        plot's bands. Printed regardless of `show`. None (default) uses the
        SHAKER-wide setting (see `set_options`), which is True unless
        changed.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    items : list of tuple
        `(name, value, error)` per subdirectory — mean and std SASA over
        all frames, sorted by subdirectory name.
    """
    if kind not in ("violin", "overlay"):
        raise ValueError("kind must be 'violin' or 'overlay'")

    entries = _load_sasa(Path(root), xvg)
    items = [(name, float(vals.mean()), float(vals.std())) for name, vals in entries]
    if kind == "overlay":
        fig, ax = _plot_sasa_overlay(entries)
    else:
        fig, ax = _plot_sasa_violin(entries)
    _finish_figure(fig, outfile, transparent, show)
    if _resolve(verbose, "verbose"):
        print(_sasa_report(items))
    return fig, ax, items


def _load_sasa(root, xvg):
    """(name, per-frame SASA) for every subdirectory of `root` holding `xvg`."""
    entries = [
        (d.name, _read_SASA_timeseries(d / xvg))
        for d in sorted(root.iterdir())
        if d.is_dir() and (d / xvg).exists()
    ]
    if not entries:
        raise ValueError(f"No '{xvg}' files found under {root}")
    return entries


def _sasa_report(items, reference="AA"):
    """
    Text report for `plot_sasa_dir`: one line per entry, with the difference
    from `reference` (if present) in nm² and %, e.g.::

        SASA (nm²)       mean ± std   vs AA
          AA           5.12 ± 0.08
          CG_Mapped    4.98 ± 0.10   -0.14 nm²  (-2.7%)  ✓
    """
    title = "SASA (nm²)"
    width = max(len(title), *(len(name) for name, _, _ in items))
    ref = {name: mean for name, mean, _ in items}.get(reference)
    lines = [
        f"{title:<{width + 2}}  {'mean ± std':>13}"
        + (f"   vs {reference}" if ref is not None else "")
    ]
    for name, mean, std in items:
        line = f"  {name:<{width}}  {mean:6.2f} ± {std:4.2f}"
        if ref is not None and name != reference:
            rel = (mean - ref) / ref
            mark = (
                "✓" if abs(rel) <= _SASA_OK else "⚠" if abs(rel) <= _SASA_WARN else "✗"
            )
            line += f"   {mean - ref:+.2f} nm²  ({rel:+.1%})  {mark}"
        lines.append(line)
    return "\n".join(lines)


def _plot_sasa_violin(entries):
    """
    Plot SASA as a violin chart, one column per `_load_sasa` entry.
    See `plot_sasa_dir` (kind="violin").
    """
    names = [name for name, _ in entries]
    distributions = [dist for _, dist in entries]
    vals = [dist.mean() for dist in distributions]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(names)), 4), tight_layout=True)

    # AA reference value, used both for the deviation bands below and as
    # the center of the y-axis window.
    aa_val = None
    if "AA" in names:
        aa_val = vals[names.index("AA")]

    # y-axis limits: the AA reference ± _SASA_WARN (10 %), widened to the
    # actual data range if any distribution extends beyond it, so nothing
    # gets clipped.
    ref_mean = aa_val if aa_val is not None else float(np.mean(vals))
    window = ref_mean * _SASA_WARN
    dist_min = min(dist.min() for dist in distributions)
    dist_max = max(dist.max() for dist in distributions)
    y_min = min(ref_mean - window, dist_min)
    y_top = max(ref_mean + window, dist_max)
    ax.set_ylim(y_min, y_top)

    legend_patches = []
    if aa_val is not None:
        bands = [
            (0.00, _SASA_OK, "#2ecc71", f"0–{_SASA_OK:.0%} from AA"),
            (
                _SASA_OK,
                _SASA_WARN,
                "#e67e22",
                f"{_SASA_OK:.0%}–{_SASA_WARN:.0%} from AA",
            ),
            (_SASA_WARN, None, "#e74c3c", f">{_SASA_WARN:.0%} from AA"),
        ]
        for lo, hi, color, label in bands:
            if hi is None:
                ax.axhspan(y_min, aa_val * (1 - lo), color=color, alpha=0.15, zorder=0)
                ax.axhspan(aa_val * (1 + lo), y_top, color=color, alpha=0.15, zorder=0)
            else:
                ax.axhspan(
                    aa_val * (1 - hi),
                    aa_val * (1 - lo),
                    color=color,
                    alpha=0.15,
                    zorder=0,
                )
                ax.axhspan(
                    aa_val * (1 + lo),
                    aa_val * (1 + hi),
                    color=color,
                    alpha=0.15,
                    zorder=0,
                )

            legend_patches.append(
                mpatches.Patch(
                    facecolor=color, alpha=0.4, edgecolor="none", label=label
                )
            )

        # The dashed line's meaning is self-evident (it sits on the AA
        # violin), so it doesn't get its own legend entry.
        ax.axhline(aa_val, color="dimgrey", lw=1.2, ls="--", zorder=1, alpha=0.7)

    parts = ax.violinplot(
        distributions, positions=x, showmeans=True, showextrema=True, widths=0.7
    )
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
        ax.legend(
            handles=legend_patches,
            loc="lower left",
            fontsize=8,
            labelspacing=0.3,
            frameon=True,
            fancybox=True,
            edgecolor="none",
            facecolor="white",
            framealpha=0.8,
        )

    return fig, ax


def _plot_sasa_overlay(entries, bins=60):
    """
    Plot per-frame SASA distributions for every `_load_sasa` entry as
    overlaid density curves, in the same visual style as
    `plot_bonded_distributions`. See `plot_sasa_dir` (kind="overlay").
    """
    names = [name for name, _ in entries]
    means = {name: vals.mean() for name, vals in entries}
    all_vals = np.concatenate([vals for _, vals in entries])
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins + 1)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    hists = {
        name: np.histogram(vals, bins=bin_edges, density=True)[0]
        for name, vals in entries
    }

    ref_name = "AA" if "AA" in hists else names[0]
    ref_hist = hists[ref_name]

    # Reference (AA, if present) always gets tab:blue and the first
    # comparison dataset tab:red, matching plot_bonded_distributions'
    # convention, rather than whatever order subdirectories sort in.
    plot_order = [ref_name] + [n for n in names if n != ref_name]
    colors = {name: _COLORS[i % len(_COLORS)] for i, name in enumerate(plot_order)}

    # x-axis limits: the reference (AA, or the first dataset if there's no
    # AA) ± _SASA_WARN (10 %), widened to the actual data range if any
    # distribution extends beyond it, so nothing is clipped.
    ref_mean = means[ref_name]
    window = ref_mean * _SASA_WARN
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
            (0.00, _SASA_OK, "#2ecc71", f"0–{_SASA_OK:.0%} from AA"),
            (
                _SASA_OK,
                _SASA_WARN,
                "#e67e22",
                f"{_SASA_OK:.0%}–{_SASA_WARN:.0%} from AA",
            ),
            (_SASA_WARN, None, "#e74c3c", f">{_SASA_WARN:.0%} from AA"),
        ]
        for lo, hi, color, label in bands:
            if hi is None:
                ax.axvspan(x_min, aa_val * (1 - lo), color=color, alpha=0.15, zorder=0)
                ax.axvspan(aa_val * (1 + lo), x_max, color=color, alpha=0.15, zorder=0)
            else:
                ax.axvspan(
                    aa_val * (1 - hi),
                    aa_val * (1 - lo),
                    color=color,
                    alpha=0.15,
                    zorder=0,
                )
                ax.axvspan(
                    aa_val * (1 + lo),
                    aa_val * (1 + hi),
                    color=color,
                    alpha=0.15,
                    zorder=0,
                )

            legend_patches.append(
                mpatches.Patch(
                    facecolor=color, alpha=0.4, edgecolor="none", label=label
                )
            )

        # The dashed line's meaning is self-evident (it sits on the AA
        # curve), so it doesn't get its own legend entry.
        ax.axvline(aa_val, color="dimgrey", lw=1.2, ls="--", zorder=1, alpha=0.7)

    curve_handles = ax.plot(
        bin_centers, ref_hist, label=ref_name, color=colors[ref_name], zorder=2
    )

    for name in names:
        if name == ref_name:
            continue
        hist, color = hists[name], colors[name]
        curve_handles += ax.plot(bin_centers, hist, label=name, color=color, zorder=2)

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("SASA (nm$^2$)", fontweight="bold")
    ax.set_ylabel("Prob. density", fontweight="bold")

    ax.legend(
        handles=curve_handles + legend_patches,
        loc="upper left",
        fontsize=8,
        ncols=2,
        labelspacing=0.3,
        columnspacing=1.0,
        frameon=True,
        fancybox=True,
        edgecolor="none",
        facecolor="white",
        framealpha=0.8,
    )

    return fig, ax


def plot_bonded_distributions(
    *bonded_dicts,
    labels=None,
    colors=None,
    outfile="cleanbonds",
    transparent=True,
    show_peaks=False,
    metrics=True,
    show=None,
    verbose=None,
    only_flagged=False,
):
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

    labels : list of str or None, optional
        Labels for each bonded dictionary. If None, uses Dataset 1, Dataset 2, ...

    colors : list of str or None, optional
        Line colors for each bonded dictionary. If None (or for None entries),
        the default palette is used: tab:blue, tab:red, tab:grey, then the
        other tab: colours.

    outfile : str or Path or None, optional
        Where to save the figure: as SVG (`<outfile>.svg`) unless it ends in
        another image extension, e.g. "bonds.png". Overwritten if it exists;
        None saves nothing. Default "cleanbonds".

    transparent : bool, optional
        Save with a transparent background. Default True.

    show_peaks : bool, optional
        Whether to annotate the peak position of each histogram.

    metrics : bool, optional
        Whether to compute and display Wasserstein distance and overlap coefficient
        for each distribution against the reference (first) dataset, and shade
        the overlapping region between curves. Default is True. For
        dihedrals, W is measured on the circle (-179° and 179° are 2° apart).

    show : bool or None, optional
        Whether to display the figure in the notebook. None (default) uses
        the SHAKER-wide setting (see `set_options`), which is True unless
        changed. The figure is saved either way.

    verbose : bool or None, optional
        Whether to print (with at least two datasets) a text report against
        the reference (first) dataset. The second dataset (e.g. CG) gets a
        table, one line per term under category headings: mean ± sd of
        both, Δ mean, OC and W (the same numbers as on the plot), marked ✓
        (OC ≥ 0.80), ⚠ (≥ 0.65) or ✗, plus "mean-off" when |Δ| is larger
        than the reference's sd, and hints about the reference distribution
        (multimodal, flat, planar on terms that are not ✓; near-linear angles
        always); then a count line. Each further dataset
        (e.g. Prev. CG) gets only its count line and the terms whose mark
        differs from the second dataset's. Units are Å for distances and
        degrees for angles and dihedrals. Printed regardless of `show`.
        None (default) uses the SHAKER-wide setting (see `set_options`),
        which is True unless changed.

    only_flagged : bool, optional
        With `verbose`, list only the terms that are not ✓ or are mean-off in
        the table (count lines and further datasets are always printed).
        Default False.

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

    categories = ("distances", "angles", "dihedrals")

    # Check that all dictionaries share the same targets and bins (the
    # metrics compare histograms bin by bin).
    ref = bonded_dicts[0]
    for cat in categories:
        ref_targets = ref[cat]["targets"]
        for idx, bonded in enumerate(bonded_dicts[1:], start=1):
            if bonded[cat]["targets"] != ref_targets:
                raise ValueError(
                    f"Mismatch in '{cat}' targets between input 0 and input {idx}."
                )
            if ref_targets and not np.array_equal(
                bonded[cat]["bins"], ref[cat]["bins"]
            ):
                raise ValueError(
                    f"Mismatch in '{cat}' bins between input 0 and input {idx}; "
                    "measure both with the same bins_* arguments."
                )

    # Labels
    if labels is None:
        labels = [f"Dataset {i + 1}" for i in range(len(bonded_dicts))]
    if len(labels) != len(bonded_dicts):
        raise ValueError("Length of 'labels' must match number of bonded dictionaries.")

    # Colors — None entries fall back to the default palette
    resolved_colors = [
        c if c is not None else _COLORS[i % len(_COLORS)]
        for i, c in enumerate(colors or [None] * len(bonded_dicts))
    ]
    if len(resolved_colors) != len(bonded_dicts):
        raise ValueError("Length of 'colors' must match number of bonded dictionaries.")

    # Categories without targets (e.g. no dihedrals) are simply left out.
    active_categories = [cat for cat in categories if len(ref[cat]["targets"])]

    if not active_categories:
        raise ValueError(
            "All categories (distances, angles, dihedrals) are empty — nothing to plot."
        )

    # Metrics for every comparison dataset vs the reference, computed once and
    # used for both the panel annotations and the text report. Keyed by
    # (category, target index).
    summaries = [
        {
            (cat, i): row
            for cat in active_categories
            for i, row in enumerate(_bonded_summary(ref, other, [cat]))
        }
        for other in bonded_dicts[1:]
    ]

    xlim_map = {"angles": (0, 180), "dihedrals": (-180, 180)}  # distances: from data

    # Layout helpers
    grids = {cat: _best_grid(len(ref[cat]["targets"])) for cat in active_categories}
    figsize = _predict_figsize(list(grids.values()))
    height_ratios = [grids[cat][0] for cat in active_categories]

    # Interleave spacer rows between category sections so sections are
    # visually separated. constrained_layout ignores hspace on the outer
    # GridSpec, so we insert explicit zero-height spacer rows instead.
    n_cats = len(active_categories)
    spacer_height = 0.3  # inches — tune this for more/less gap
    spacer_ratio = spacer_height / _CELL_H

    interleaved_ratios = []
    for k, r in enumerate(height_ratios):
        interleaved_ratios.append(r)
        if k < n_cats - 1:
            interleaved_ratios.append(spacer_ratio)

    n_outer_rows = len(interleaved_ratios)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(
        n_outer_rows,
        1,
        figure=fig,
        height_ratios=interleaved_ratios,
    )

    # Add invisible spacer axes so constrained_layout respects the rows
    for k in range(1, n_outer_rows, 2):
        spacer_ax = fig.add_subplot(gs[k])
        spacer_ax.set_visible(False)

    xlabel_map = {
        "distances": "Distance (Å)",
        "angles": "Angle (°)",
        "dihedrals": "Dihedral angle (°)",
    }

    config = {
        cat: {
            "grid": grids[cat],
            "subplot": gs[i * 2],  # every other row; odd rows are spacers
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

            # Plot every dataset; compare each one after the first against the
            # reference (dataset 0). Each metrics entry is (text, color) so
            # each comparison gets its own colour.
            metrics_lines = []
            for j, (bonded, label, color) in enumerate(
                zip(bonded_dicts, labels, resolved_colors)
            ):
                bins = bonded[cat]["bins"]
                hist = np.asarray(bonded[cat]["hist"][i])

                ax.plot(bins, hist, label=label, color=color)

                if show_peaks and len(hist) > 0:
                    peak_idx = np.argmax(hist)
                    ax.text(
                        bins[peak_idx],
                        hist[peak_idx],
                        f"{bins[peak_idx]:.2f}",
                        color=color,
                        fontsize=6,
                        va="bottom",
                        ha="center",
                    )

                if metrics and j > 0:
                    # Shade the overlapping region (kept out of the legend —
                    # it's visually self-evident, and an entry per comparison
                    # dataset would make the legend grow unboundedly).
                    ax.fill_between(
                        ref_bins, np.minimum(ref_hist, hist), alpha=0.15, color=color
                    )

                    row = summaries[j - 1][cat, i]
                    line_color, status = _STATUS_STYLE[row["status"]]
                    d = 2 if cat == "distances" else 1  # decimals: Å vs degrees
                    metrics_lines.append(
                        (
                            f"{label}  W={row['w']:.{d}f}  OC={row['oc']:.2f}  {status}",
                            line_color,
                        )
                    )

            # Annotate metrics in the upper-right corner, one text call per line
            # so each comparison carries its own colour.
            if metrics and metrics_lines:
                line_height = 0.10
                for k, (line, line_color) in enumerate(metrics_lines):
                    ax.text(
                        0.98,
                        0.95 - k * line_height,
                        line,
                        transform=ax.transAxes,
                        fontsize=5,
                        fontweight="bold",
                        va="top",
                        ha="right",
                        family="monospace",
                        color=line_color,
                        bbox={
                            "boxstyle": "round,pad=0.3",
                            "fc": "white",
                            "alpha": 0.8,
                            "lw": 0,
                        },
                    )

            # --- axis limits ---
            all_hists = [np.asarray(bonded[cat]["hist"][i]) for bonded in bonded_dicts]
            valid_maxes = [h.max() for h in all_hists if len(h) > 0 and h.max() > 0]
            y_max = max(valid_maxes) if valid_maxes else 1.0
            ax.set_ylim(0, y_max * 1.2)
            if cat == "distances":
                ax.set_xlim(_distance_xlim(ref_bins, all_hists))
            else:
                ax.set_xlim(xlim_map[cat])

            ax.set_title("-".join(map(str, distribution)), fontweight="bold")
            ax.legend(
                loc="upper left",
                fontsize=5,
                ncols=1,
                labelspacing=0.3,
                frameon=True,
                fancybox=True,
                edgecolor="none",
                facecolor="white",
                framealpha=0.8,
            )
            ax.set_xlabel(config[cat]["xlabel"], fontsize=9)
            ax.set_ylabel("Prob. density")

    _finish_figure(fig, outfile, transparent, show)
    if _resolve(verbose, "verbose") and summaries:
        print(_bonded_report(summaries, labels, only_flagged))
    return fig


def _oc_status(oc):
    """
    ✓ / ⚠ / ✗ for an overlap coefficient, judged on the 2-decimal value that
    is displayed (so "0.80" is never shown with ⚠).
    """
    oc = round(float(oc), 2)
    return "✓" if oc >= _OC_GOOD else "⚠" if oc >= _OC_WARN else "✗"


def _distance_xlim(bins, hists, default=(1.5, 5.5), width=4.0, frac=0.01):
    """
    x-limits for a distance panel: `default` unless some dataset has
    significant density (> `frac` of its peak) outside it; then a window of
    the same `width` (Å) centred between the datasets' peaks, so panels stay
    comparable at a glance (tails of very broad distributions may be cut).
    """
    bins = np.asarray(bins, float)
    present = np.zeros(bins.size, bool)
    peaks = []
    for h in hists:
        h = np.nan_to_num(np.asarray(h, float))
        if h.size and h.max() > 0:
            present |= h > frac * h.max()
            peaks.append(bins[np.argmax(h)])
    if not peaks:
        return default
    lo, hi = bins[present].min(), bins[present].max()
    if default[0] <= lo and hi <= default[1]:
        return default
    center = max((min(peaks) + max(peaks)) / 2, width / 2)  # never below 0 Å
    return float(center - width / 2), float(center + width / 2)


def _overlap_coefficient(bins, p, q):
    """∫ min(p, q) dx for two densities at the same bin centres (NaN → 0)."""
    p, q = np.nan_to_num(np.asarray(p, float)), np.nan_to_num(np.asarray(q, float))
    return float(_trapz(np.minimum(p, q), bins))


def _distribution_metrics(ref_bins, ref_hist, bins, hist, periodic=False):
    """
    Overlap coefficient ∫ min(p, q) dx and Wasserstein distance between two
    densities given at bin centres. With `periodic` (dihedrals, same bins),
    W is measured on the circle: -179° and 179° are 2° apart, not 358°.
    """
    ref_hist = np.nan_to_num(np.asarray(ref_hist, float))
    hist = np.nan_to_num(np.asarray(hist, float))
    oc = _overlap_coefficient(ref_bins, ref_hist, hist)
    if ref_hist.sum() == 0 or hist.sum() == 0:
        return oc, float("nan")
    if not periodic:
        return oc, wasserstein_distance(ref_bins, bins, ref_hist, hist)
    # Circular W1: ∫ |F - G - c| dθ with c the median of F - G (F, G = CDFs).
    diff = np.cumsum(ref_hist / ref_hist.sum()) - np.cumsum(hist / hist.sum())
    return oc, np.sum(np.abs(diff - np.median(diff))) * np.mean(np.diff(ref_bins))


def _hist_stats(bins, hist, periodic=False):
    """Mean and sd of a density at bin centres; circular for `periodic`."""
    w = np.nan_to_num(np.asarray(hist, float))
    if w.sum() == 0:
        return float("nan"), float("nan")
    w = w / w.sum()
    x = np.asarray(bins, float)
    if periodic:
        mean = np.degrees(np.angle(np.sum(w * np.exp(1j * np.radians(x)))))
        dev = (x - mean + 180) % 360 - 180
    else:
        mean = np.sum(w * x)
        dev = x - mean
    return float(mean), float(np.sqrt(np.sum(w * dev**2)))


def _term_hints(cat, bins, hist):
    """
    Hints about a reference distribution for the bonded report: multimodal
    (any term), flat / planar (dihedrals), near-linear (angles).
    """
    x, h = np.asarray(bins, float), np.nan_to_num(np.asarray(hist, float))
    if h.sum() == 0:
        return []
    periodic = cat == "dihedrals"
    mean, sd = _hist_stats(x, h, periodic)
    hints = ["multimodal"] if _n_peaks(h, periodic) > 1 else []
    if periodic and sd > 60 and not hints:  # broad single peak, not several
        hints.append("flat")
    elif periodic and sd < 20 and min(abs(mean), 180 - abs(mean)) < 20:
        hints.append("planar")
    if cat == "angles":
        p99 = x[np.searchsorted(np.cumsum(h) / h.sum(), 0.99)]
        if mean > 150 or p99 > 170:
            hints.append("near-linear")
    return hints


def _n_peaks(hist, periodic=False, width=5, prominence=0.2):
    """
    Number of clear peaks in a histogram: after a `width`-bin moving average,
    peaks standing out by at least `prominence` × the highest point. For
    `periodic` data the ends wrap around, so a peak across ±180° counts once.
    """
    kernel = np.ones(width) / width
    if periodic:
        pad = width // 2
        h = np.convolve(
            np.concatenate([hist[-pad:], hist, hist[:pad]]), kernel, "valid"
        )
        h = np.roll(h, -np.argmin(h))  # lowest point at the ends: no split peaks
    else:
        h = np.convolve(hist, kernel, "same")
    peaks, _ = find_peaks(
        np.concatenate([[0], h, [0]]), prominence=prominence * h.max()
    )
    return len(peaks)


def _bonded_summary(ref, other, categories=("distances", "angles", "dihedrals")):
    """
    Compare two `measure_bonded_terms` dicts term by term, `ref` being the
    reference. Returns one dict per term with keys: type, term, ref_mean,
    ref_sd, mean, sd, delta (wrapped to ±180 for dihedrals), oc, w, status
    ("✓"/"⚠"/"✗" from OC) and mean_off (|delta| > ref_sd).
    """
    summary = []
    for cat in categories:
        periodic = cat == "dihedrals"
        for i, tgt in enumerate(ref[cat]["targets"]):
            ref_bins, ref_hist = ref[cat]["bins"], ref[cat]["hist"][i]
            bins, hist = other[cat]["bins"], other[cat]["hist"][i]
            oc, w = _distribution_metrics(ref_bins, ref_hist, bins, hist, periodic)
            ref_mean, ref_sd = _hist_stats(ref_bins, ref_hist, periodic)
            mean, sd = _hist_stats(bins, hist, periodic)
            delta = mean - ref_mean
            if periodic:
                delta = (delta + 180) % 360 - 180
            summary.append(
                {
                    "type": cat[:-1],
                    "term": "-".join(map(str, tgt)),
                    "ref_mean": ref_mean,
                    "ref_sd": ref_sd,
                    "mean": mean,
                    "sd": sd,
                    "delta": delta,
                    "oc": float(oc),
                    "w": float(w),
                    "status": _oc_status(oc),
                    "mean_off": abs(delta) > ref_sd,
                    "hints": _term_hints(cat, ref_bins, ref_hist),
                }
            )
    return summary


_CATEGORY_HEADINGS = {
    "distance": "distances (Å)",
    "angle": "angles (°)",
    "dihedral": "dihedrals (°)",
}


def _bonded_report(summaries, labels, only_flagged=False):
    """
    Text report for `plot_bonded_distributions`. `summaries` holds one
    `_bonded_summary` result (a dict in term order) per comparison dataset.

    The first comparison (the current model, e.g. CG) gets a full table:
    one line per term under category headings, then a count line. Each
    further comparison (e.g. Prev. CG) gets only its counts and the terms
    whose status differs from the first comparison, to keep output short.
    """
    ref, main = labels[0], labels[1]
    summary = list(summaries[0].values())

    header = ("", f"{ref} mean ± sd", f"{main} mean ± sd", "Δ", "", "", "", "")
    rows, categories, any_hints = [header], [None], False
    for t in summary:
        # near-linear is a stability warning, so always shown; the other
        # hints only explain terms that don't match.
        hints = [h for h in t["hints"] if h == "near-linear" or t["status"] != "✓"]
        any_hints |= bool(hints)
        if only_flagged and t["status"] == "✓" and not t["mean_off"] and not hints:
            continue
        d = 2 if t["type"] == "distance" else 1  # decimals: Å vs degrees
        rows.append(
            (
                f"  {t['term']}",
                f"{t['ref_mean']:.{d}f} ± {t['ref_sd']:.{d}f}",
                f"{t['mean']:.{d}f} ± {t['sd']:.{d}f}",
                f"{t['delta']:+.{d}f}",
                f"OC {t['oc']:.2f}",
                f"W {t['w']:.{d}f}",
                t["status"] + (" mean-off" if t["mean_off"] else ""),
                ", ".join(hints),
            )
        )
        categories.append(t["type"])

    lines = [f"Bonded: {main} vs {ref}"]
    # Align all rows as one table, then put a heading before each category.
    for line, cat, prev in zip(
        _table(rows, align="<>>>>><<"), categories, [None, *categories]
    ):
        if cat is not None and cat != prev:
            lines.append(_CATEGORY_HEADINGS[cat])
        lines.append(line)
    lines.append(_status_counts(summary))

    # Further comparisons: counts, plus what changed relative to `main`.
    rank = {"✓": 0, "⚠": 1, "✗": 2}
    for other, label in zip(summaries[1:], labels[2:]):
        lines += ["", f"{label} vs {ref}: {_status_counts(other.values())}"]
        changes = {"worse": [], "better": []}
        for key, t in summaries[0].items():
            o = other[key]
            if t["status"] != o["status"]:
                kind = "worse" if rank[t["status"]] > rank[o["status"]] else "better"
                changes[kind].append(
                    f"{t['term']} {o['status']}→{t['status']} "
                    f"(OC {o['oc']:.2f}→{t['oc']:.2f})"
                )
        for kind, items in changes.items():
            lines.append(f"  {main} {kind + ':':7} {', '.join(items) or 'none'}")

    if any_hints:
        lines.append(
            f"(hints, from {ref}: multimodal = several clear peaks, one harmonic "
            "can't fit; flat = one broad dihedral peak (sd > 60°), likely needs no "
            "potential; planar = "
            "dihedral near 0/180° with sd < 20°, improper candidate; near-linear = "
            "angle mean > 150° or reaching 170°, unstable in dihedrals)"
        )
    lines.append(
        f"(OC ✓ ≥ {_OC_GOOD:.2f}, ⚠ ≥ {_OC_WARN:.2f}; mean-off: |Δ| > {ref} sd; "
        "distances Å, angles/dihedrals °)"
    )
    return "\n".join(lines)


def _status_counts(summary):
    """'5 terms: 2 ✓  1 ⚠  2 ✗, 2 mean-off' for a `_bonded_summary`."""
    summary = list(summary)
    n = {mark: sum(t["status"] == mark for t in summary) for mark in "✓⚠✗"}
    off = sum(t["mean_off"] for t in summary)
    return f"{len(summary)} terms: {n['✓']} ✓  {n['⚠']} ⚠  {n['✗']} ✗, {off} mean-off"


def _table(rows, align):
    """Rows of strings as aligned text lines; `align` has one '<'/'>' per column."""
    widths = [max(len(row[k]) for row in rows) for k in range(len(align))]
    return [
        "  ".join(f"{c:{a}{w}}" for c, a, w in zip(row, align, widths)).rstrip()
        for row in rows
    ]


def _figure_path(outfile, tag=None):
    """
    Resolve an `outfile` argument to the file to write: SVG unless it already
    ends in an image extension (e.g. ".png"). `tag` is inserted before the
    extension, e.g. one file per dihedral: "fit" → "fit_B1-B2-B3-B4.svg".
    """
    path = Path(outfile)
    ext = path.suffix if path.suffix[1:].lower() in _IMAGE_EXTS else ""
    stem = path.name[: len(path.name) - len(ext)]
    if tag:
        stem = f"{stem}_{tag}"
    return path.with_name(stem + (ext or ".svg"))


def _finish_figure(fig, outfile, transparent, show, tag=None):
    """
    Save `fig` (see `_figure_path`; None saves nothing), then close it unless
    it should be shown. Returns the resolved `show`.

    Closing is what stops Jupyter's inline backend from rendering the
    figure at the end of the cell; the returned Figure object still works.
    """
    if outfile is not None:
        path = _figure_path(outfile, tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        # dpi only matters for raster formats (png, jpg, ...).
        fig.savefig(path, transparent=transparent, dpi=300, bbox_inches="tight")

    show = _resolve(show, "show")
    if not show:
        plt.close(fig)
    return show


def _read_SASA_timeseries(xvg):
    """
    Reader for the SASA-vs-time .xvg file (`gmx sasa -o`). Retrieves the
    per-frame total SASA values, ignoring the time column.
    """
    return np.loadtxt(xvg, comments=("#", "@"), usecols=1, ndmin=1)


def _best_grid(n: int):
    """Return (nrows, ncols) for n plots using a near-square layout."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def _predict_figsize(
    grids,
    cell_w=3.2,
    cell_h=_CELL_H,
    section_gap_h=0.6,
    left=0.8,
    right=0.2,
    top=0.6,
    bottom=0.6,
):
    """
    grids: list of (nrows, ncols) for sections stacked vertically
    returns (fig_w, fig_h) in inches
    """
    max_cols = max(c for r, c in grids)
    total_rows = sum(r for r, c in grids)

    fig_w = left + max_cols * cell_w + right
    fig_h = bottom + total_rows * cell_h + top + section_gap_h * (len(grids) - 1)
    return fig_w, fig_h
