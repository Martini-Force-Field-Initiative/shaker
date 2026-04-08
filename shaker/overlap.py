import numpy as np
from tqdm.autonotebook import tqdm
from MDAnalysis.analysis.distances import self_distance_array
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors


def _make_overlap_cmap():
    rdylgn = plt.get_cmap('RdYlGn')
    cmap   = mcolors.LinearSegmentedColormap.from_list(
        'overlap',
        [(0.00, rdylgn(0.00)),
         (0.60, rdylgn(0.45)),
         (0.80, rdylgn(0.65)),
         (1.00, rdylgn(1.00))],
        N=512
    )
    cmap.set_bad(color='black')
    return cmap


def _measure_intra_bead_distances(u, resname,
                                   bead_names=None,
                                   start=0, stop=None, stride=1,
                                   bins_dist=None,
                                   save_npy=False, outname=''):
    '''
    Measure all pairwise intra-residue bead distance distributions.

    For each unique bead pair (i < j), collects distances across all residues
    and frames, then histograms them as normalised probability densities.
    PBC-correct via self_distance_array.

    If bead_names is None, all beads in the first residue of resname are used.
    Returns dict with keys: bins, hist (n_pairs × n_bins), pairs, bead_names.
    '''
    if bins_dist is None:
        bins_dist = np.arange(0, 15, 0.2)

    if save_npy and not outname:
        raise ValueError("outname must be provided when save_npy=True")

    if bead_names is None:
        ag = u.select_atoms(f'resname {resname}')
        if len(ag) == 0:
            raise ValueError(f"No atoms found for resname '{resname}'")
        first_resid = ag.resids[0]
        bead_names  = list(u.select_atoms(
            f'resname {resname} and resid {first_resid}').names)

    bins_x  = (bins_dist[1:] + bins_dist[:-1]) / 2
    pairs   = [(bead_names[i], bead_names[j])
               for i in range(len(bead_names))
               for j in range(i + 1, len(bead_names))]
    raw_out = [[] for _ in pairs]
    resids  = np.unique(u.select_atoms(f'resname {resname}').resids)

    sel = 'resname {} and resid {} and name {}'
    bead_selections = {
        (resid, bead): u.select_atoms(sel.format(resname, resid, bead))
        for resid in resids for bead in bead_names
    }

    for ts in tqdm(u.trajectory[start:stop:stride], desc="Measuring intra-bead distances"):
        for resid in resids:
            positions = np.array([bead_selections[(resid, bead)].positions[0]
                                  for bead in bead_names])
            dists = self_distance_array(positions, box=ts.dimensions)
            for k, d in enumerate(dists):
                raw_out[k].append(d)

    hists = np.array([np.histogram(v, bins=bins_dist, density=True)[0]
                      for v in raw_out])

    results = {"bins": bins_x, "hist": hists,
               "pairs": pairs, "bead_names": bead_names}

    if save_npy:
        np.save(f'{outname}_intra_bead_distances_raw.npy', raw_out)
        np.save(f'{outname}_intra_bead_distances_hists.npy', results)

    return results


def _compute_overlap_matrix(results_aa, results_cg):
    '''
    Compute a symmetric N×N overlap coefficient matrix from AA and CG
    intra-bead distance distributions.

    OC = ∫ min(p_AA, p_CG) dx ≈ sum(min(P, Q)) * bin_width

    Valid only when histograms are density=True. Diagonal set to 1.
    Returns dict with keys: matrix, overlaps, pairs, bead_names.
    '''
    if results_aa['pairs'] != results_cg['pairs']:
        raise ValueError("AA and CG results must have the same pairs — "
                         "were they computed with the same resname/bead_names?")

    bead_names = results_aa['bead_names']
    pairs      = results_aa['pairs']
    n          = len(bead_names)
    bead_idx   = {b: i for i, b in enumerate(bead_names)}
    bin_width  = results_aa['bins'][1] - results_aa['bins'][0]

    overlaps = np.array([
        np.sum(np.minimum(h_aa, h_cg)) * bin_width
        for h_aa, h_cg in zip(results_aa['hist'], results_cg['hist'])
    ])

    matrix = np.eye(n)
    for k, (bi, bj) in enumerate(pairs):
        i, j         = bead_idx[bi], bead_idx[bj]
        matrix[i, j] = overlaps[k]
        matrix[j, i] = overlaps[k]

    return {"matrix": matrix, "overlaps": overlaps,
            "pairs": pairs, "bead_names": bead_names}


def _plot_overlap_matrix(overlap_results,
                         vmin=0, vmax=1,
                         title='Intra-bead Distance Overlap',
                         cell_fontsize=11):
    '''
    Plot the overlap coefficient matrix as an annotated heatmap with a
    per-bead mean column. Diagonal cells are black (self excluded from means).
    Returns fig, (ax_matrix, ax_mean, ax_cbar).
    '''
    matrix     = overlap_results['matrix']
    bead_names = overlap_results['bead_names']
    n          = len(bead_names)

    mask_diag      = np.eye(n, dtype=bool)
    matrix_no_diag = np.where(mask_diag, np.nan, matrix)
    row_means      = np.nanmean(matrix_no_diag, axis=1)
    cmap           = _make_overlap_cmap()

    fig = plt.figure(figsize=(n * 1.2 + 3.5, n * 1.2 + 1))
    gs  = gridspec.GridSpec(1, 4, width_ratios=[n, 1, 0.15, 0.5], wspace=0.02)
    ax_matrix = fig.add_subplot(gs[0])
    ax_mean   = fig.add_subplot(gs[1])
    ax_cbar   = fig.add_subplot(gs[3])

    im = ax_matrix.imshow(matrix_no_diag, cmap=cmap, vmin=vmin, vmax=vmax,
                          aspect='equal')

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = matrix[i, j]
            ax_matrix.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=cell_fontsize,
                           color='#2b2b2b' if 0.25 < val < 0.92 else 'white')

    ax_matrix.set_xticks(range(n))
    ax_matrix.set_yticks(range(n))
    ax_matrix.set_xticklabels(bead_names, rotation=45, ha='right',
                               fontsize=12, fontweight='bold')
    ax_matrix.set_yticklabels(bead_names, fontsize=12, fontweight='bold')
    ax_matrix.set_title(title, fontsize=11, pad=10, fontweight='bold')
    ax_matrix.tick_params(length=0)
    for spine in ax_matrix.spines.values():
        spine.set_visible(False)

    ax_mean.imshow(row_means.reshape(n, 1), cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect='equal')

    for i in range(n):
        val = row_means[i]
        ax_mean.text(0, i, f'{val:.2f}', ha='center', va='center',
                     fontsize=cell_fontsize,
                     color='#2b2b2b' if 0.25 < val < 0.92 else 'white')

    ax_mean.set_xticks([0])
    ax_mean.set_xticklabels(['mean'], fontsize=10, fontweight='bold')
    ax_mean.set_yticks([])
    ax_mean.tick_params(length=0)
    for spine in ax_mean.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label('Overlap coefficient', fontsize=10, fontweight='bold')
    cbar.locator   = ticker.MultipleLocator(0.2)
    cbar.formatter = ticker.FormatStrFormatter('%.1f')
    cbar.update_ticks()
    cbar.outline.set_visible(False)

    fig.tight_layout()
    return fig, (ax_matrix, ax_mean, ax_cbar)


def assess_overlap_matrix(u_aa, u_cg, resname,
                           bead_names=None,
                           start=0, stop=None, stride=1,
                           bins_dist=None,
                           vmin=0, vmax=1,
                           cell_fontsize=11,
                           title='Intra-bead Distance Overlap',
                           save_npy=False, outname=''):
    '''
    Assess CG parameterisation quality by comparing intra-bead distance
    distributions between AA and CG trajectories.

    For a given residue, measures the pairwise distance distribution for every
    bead pair in both trajectories, computes the overlap coefficient for each
    pair, and visualises the result as an annotated N×N heatmap. A per-bead
    mean overlap column is appended to the right of the matrix, providing an
    at-a-glance summary of which beads are well or poorly reproduced by the
    CG model.

    Parameters
    ----------
    u_aa : MDAnalysis.Universe
        AA reference trajectory.
    u_cg : MDAnalysis.Universe
        CG trajectory to evaluate.
    resname : str
        Residue name to analyse. Bead names are inferred from the topology
        unless explicitly provided via bead_names.
    bead_names : list of str or None, optional
        Bead names to include in the analysis. If None, all beads in the
        residue are used. Provide a subset to focus on specific beads,
        e.g. ['BB', 'SC1', 'SC2'].
    start : int, optional
        First frame to analyse (default: 0).
    stop : int or None, optional
        Last frame to analyse (default: None, i.e. end of trajectory).
    stride : int, optional
        Frame stride (default: 1).
    bins_dist : array-like or None, optional
        Bin edges for distance histograms (default: 0–15 Å in 0.2 Å steps).
    vmin : float, optional
        Lower bound for colormap normalisation (default: 0).
    vmax : float, optional
        Upper bound for colormap normalisation (default: 1).
    cell_fontsize : int, optional
        Font size for overlap values inside each cell (default: 11).
    title : str, optional
        Title for the overlap matrix plot.
    save_npy : bool, optional
        If True, saves raw and histogram arrays to .npy files.
    outname : str, optional
        Prefix for saved files. If provided, the figure is saved as
        ``{outname}_overlap_matrix.png`` and histogram arrays as
        ``{outname}_aa/cg_intra_bead_distances_*.npy``.

    Returns
    -------
    overlap_results : dict
        Output of _compute_overlap_matrix — contains the N×N overlap matrix,
        per-pair overlap coefficients, pair labels, and bead names.
    fig : matplotlib.figure.Figure
    axes : tuple of (ax_matrix, ax_mean, ax_cbar)

    Examples
    --------
    >>> oc, fig, axes = assess_overlap_matrix(u_aa, u_cg, resname='MOL',
    ...                                       outname='MOL')

    >>> oc, fig, axes = assess_overlap_matrix(u_aa, u_cg, resname='MOL',
    ...                                       bead_names=['BB', 'SC1', 'SC2'])

    Notes
    -----
    The overlap coefficient for each bead pair is defined as the intersection
    of the AA and CG probability density functions:

        OC = ∫ min(p_AA(x), p_CG(x)) dx  ≈  sum(min(P, Q)) * bin_width

    where P and Q are the normalised probability densities (density=True) of
    the AA and CG distance distributions respectively. Values range from 0
    (no overlap) to 1 (identical distributions). The diagonal (self-pairs)
    is excluded from the mean overlap calculation.
    '''
    
    if bins_dist is None:
        bins_dist = np.arange(0, 15, 0.2)

    results_aa = _measure_intra_bead_distances(u_aa, resname,
                                               bead_names=bead_names,
                                               start=start, stop=stop,
                                               stride=stride,
                                               bins_dist=bins_dist,
                                               save_npy=save_npy,
                                               outname=outname + '_aa' if outname else '')
    results_cg = _measure_intra_bead_distances(u_cg, resname,
                                               bead_names=bead_names,
                                               start=start, stop=stop,
                                               stride=stride,
                                               bins_dist=bins_dist,
                                               save_npy=save_npy,
                                               outname=outname + '_cg' if outname else '')
    oc        = _compute_overlap_matrix(results_aa, results_cg)
    fig, axes = _plot_overlap_matrix(oc, vmin=vmin, vmax=vmax, title=title,
                                     cell_fontsize=cell_fontsize)

    if outname:
        fig.savefig(f'{outname}_overlap_matrix.png', dpi=150, bbox_inches='tight')

    return oc, fig, axes