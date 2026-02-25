import MDAnalysis as md
import numpy as np
from tqdm import tqdm 

def measure_bonded_terms (u, resname,
                          dist_tgts, ang_tgts, dihed_tgts, 
                          start=0, stop=None, stride=100, 
                          bins_dist=np.arange(0,15,0.2),
                          bins_angl=np.arange(-1,181,2),
                          bins_dihed=np.arange(-181,181,2), 
                          save_npy=False, outname='',
                         ):
    '''
    Measure bonded parameters, bonds, angles and dihedrals. Run this for both AA and CG sims.
    Returns the normalized distributions and saves both the raw date and the distributions in the working directory under the outname provided.
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
    
    for resid in tqdm(resids):
        dists = [beadstodistance(u, resid, resname, a, 
                                    resid, resname, b)
                 for (a, b) in dist_tgts]
        
        angs = [beadstoangle(u, resid, resname, a, 
                                resid, resname, b, 
                                resid, resname, c)
                for (a, b, c) in ang_tgts]
        
        diheds = [beadstodihedral(u, resid, resname, a, 
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
def beadstodihedral(u, resid1, resname1, name1, 
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
def beadstoangle(u, resid1, resname1, name1, 
                    resid2, resname2, name2, 
                    resid3, resname3, name3):
    selection = 'resname {} and resid {} and name {}'
    A = selection.format(resname1, resid1, name1)
    B = selection.format(resname2, resid2, name2)
    C = selection.format(resname3, resid3, name3)
    ag = u.select_atoms(A) + u.select_atoms(B) + u.select_atoms(C)
    return ag.angle

###Define function to get the distances between 2 specified beads
class Distance:
    def __init__(self, A, B):
        self.A = A
        self.B = B
    def value(self):
        return np.linalg.norm(self.A.positions[0] - self.B.positions[0])
        
def beadstodistance(u, resid1, resname1, name1, resid2, resname2, name2):
    selection = 'resname {} and resid {} and name {}'
    A = u.select_atoms(selection.format(resname1, resid1, name1))
    B = u.select_atoms(selection.format(resname2, resid2, name2))
    if len(A) != 1 or len(B) != 1:
        raise ValueError(f"Expected 1 atom each, got len(A)={len(A)}, len(B)={len(B)} "
                         f"for {resname1}:{resid1}:{name1} and {resname2}:{resid2}:{name2}")
    return Distance(A, B)
    