import numpy as np
from tqdm.autonotebook import tqdm 

def measure_bonded_terms (u, resname,
                          dist_tgts, ang_tgts, dihed_tgts, 
                          start=0, stop=None, stride=1, 
                          bins_dist=np.arange(0,15,0.2),
                          bins_angl=np.arange(-1,181,2),
                          bins_dihed=np.arange(-181,181,2), 
                          save_npy=False, outname='',
                         ):
    '''
    Measure bonded distributions (distances, angles, and dihedrals) from a trajectory.
    
    This function evaluates geometric properties for a specified residue type
    across a trajectory and computes normalized probability distributions for
    distances, angles, and dihedral angles. The geometric terms are defined by
    user-provided bead/atom name iterables.
    
    Parameters
    ----------
    u : MDAnalysis.Universe
        Universe containing the structure and trajectory to analyze.
    resname : str
        Residue name for which bonded terms should be measured.
    dist_tgts : sequence of sequence of str
        Distance targets. Each entry contains two atom/bead names defining a bond.
        Example: [('BB','SC1'), ('SC1','SC2')] or [['BB','SC1'], ['SC1','SC2']].
    ang_tgts : sequence of sequence of str
        Angle targets. Each entry contains three atom/bead names defining an angle.
        Example: [('BB','SC1','SC2')].
    dihed_tgts : sequence of sequence of str
        Dihedral targets. Each entry contains four atom/bead names defining a dihedral.
    start : int, optional
        First trajectory frame to analyze (default is 0).
    stop : int or None, optional
        Last frame to analyze (default is None, meaning the trajectory end).
    stride : int, optional
        Frame stride used when iterating through the trajectory (default is 1).
    bins_dist : array-like, optional
        Bin edges used for distance histograms.
    bins_angl : array-like, optional
        Bin edges used for angle histograms.
    bins_dihed : array-like, optional
        Bin edges used for dihedral histograms.
    save_npy : bool, optional
        If True, raw measurements and histogram data are saved to `.npy` files.
    outname : str, optional
        Prefix used for saved output files. Required if `save_npy=True`.
    
    Returns
    -------
    dict
        Dictionary containing histogram data and metadata for each bonded term:
    
        - ``results["distances"]`` : distance distributions
        - ``results["angles"]`` : angle distributions
        - ``results["dihedrals"]`` : dihedral distributions
    
        Each entry contains:
            - ``bins`` : bin centers
            - ``hist`` : normalized histogram values
            - ``targets`` : bead/atom definitions used to compute the term
    
    Notes
    -----
    - Distances, angles, and dihedrals are computed for every residue with the
      specified `resname`.
    - Histogram values are normalized probability densities (`density=True`).
    - Raw measurement arrays can optionally be saved for further analysis.
    - The trajectory is iterated using the provided frame slicing
      ``u.trajectory[start:stop:stride]``.
    '''
    
    if save_npy and not outname:
        raise ValueError("outname must be provided when save_npy=True")
    
    bins_dist_x  = (bins_dist[1:] + bins_dist[:-1]) / 2
    bins_angl_x  = (bins_angl[1:] + bins_angl[:-1]) / 2
    bins_dihed_x = (bins_dihed[1:] + bins_dihed[:-1]) / 2
    
    dist_out  = [[] for _ in dist_tgts]
    ang_out   = [[] for _ in ang_tgts]
    dihed_out = [[] for _ in dihed_tgts]

    # print('Calculating distributions...')
    resids = np.unique(u.select_atoms(f'resname {resname}').resids)
    
    for resid in tqdm(resids, desc="Measuring Bonded parameters"):
        dists = [_beadstodistance(u, resid, resname, a, 
                                    resid, resname, b)
                 for (a, b) in dist_tgts]
        
        angs = [_beadstoangle(u, resid, resname, a, 
                                resid, resname, b, 
                                resid, resname, c)
                for (a, b, c) in ang_tgts]
        
        diheds = [_beadstodihedral(u, resid, resname, a, 
                                     resid, resname, b,
                                     resid, resname, c, 
                                     resid, resname, d)
                  for (a, b, c, d) in dihed_tgts]
        
        for ts in u.trajectory[start:stop:stride]:
            for i, dist in enumerate(dists):
                dist_out[i].append(dist.value())
            for i, ang in enumerate(angs):
                ang_out[i].append(ang.value())
            for i, dihed in enumerate(diheds):
                dihed_out[i].append(dihed.value())

    # print('Histogramming...')
    dist_hist  = [np.histogram(vals, bins=bins_dist,  density=True)[0] for vals in dist_out]
    ang_hist   = [np.histogram(vals, bins=bins_angl,  density=True)[0] for vals in ang_out]
    dihed_hist = [np.histogram(vals, bins=bins_dihed, density=True)[0] for vals in dihed_out]

    results = {"distances": {"bins": bins_dist_x,
                             "hist": np.array(dist_hist),
                             "targets": dist_tgts,},
               
               "angles": {"bins": bins_angl_x,
                          "hist": np.array(ang_hist),
                          "targets": ang_tgts,},
               
               "dihedrals": {"bins": bins_dihed_x,
                             "hist": np.array(dihed_hist),
                             "targets": dihed_tgts,},
              }

    if save_npy:
        np.save(f'{outname}_angles.npy', ang_out)
        np.save(f'{outname}_distances.npy', dist_out)
        np.save(f'{outname}_dihedrals.npy', dihed_out)
        np.save(f'{outname}_hists.npy', results)
    
    return results


###Define function to return a dihedral object for a specified set of 4 beads
def _beadstodihedral(u, resid1, resname1, name1, 
                       resid2, resname2, name2, 
                       resid3, resname3, name3, 
                       resid4, resname4, name4):
    selection = 'resname {} and resid {} and name {}'    
    A = selection.format(resname1, resid1, name1)
    B = selection.format(resname2, resid2, name2)
    C = selection.format(resname3, resid3, name3)
    D = selection.format(resname4, resid4, name4)
    ag = (u.select_atoms(A) + u.select_atoms(B) +
          u.select_atoms(C) + u.select_atoms(D))
    return ag.dihedral
    
###Define function to get the angle object for a specified set of 3 beads
def _beadstoangle(u, resid1, resname1, name1, 
                    resid2, resname2, name2, 
                    resid3, resname3, name3):
    selection = 'resname {} and resid {} and name {}'
    A = selection.format(resname1, resid1, name1)
    B = selection.format(resname2, resid2, name2)
    C = selection.format(resname3, resid3, name3)
    ag = u.select_atoms(A) + u.select_atoms(B) + u.select_atoms(C)
    return ag.angle

###Define function to get the distances between 2 specified beads
class _Distance:
    def __init__(self, A, B):
        self.A = A
        self.B = B
    def value(self):
        return np.linalg.norm(self.A.positions[0] - self.B.positions[0])
        
def _beadstodistance(u, resid1, resname1, name1, resid2, resname2, name2):
    selection = 'resname {} and resid {} and name {}'
    A = u.select_atoms(selection.format(resname1, resid1, name1))
    B = u.select_atoms(selection.format(resname2, resid2, name2))
    if len(A) != 1 or len(B) != 1:
        raise ValueError(f"Expected 1 atom each, got len(A)={len(A)}, len(B)={len(B)} "
                         f"for {resname1}:{resid1}:{name1} and {resname2}:{resid2}:{name2}")
    return _Distance(A, B)
    