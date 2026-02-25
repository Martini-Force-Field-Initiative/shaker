import numpy as np
import MDAnalysis as md
import itertools


def align_everything(GRO, XTC, TPR, frame_atoms, selection='all'):
    frame_str = ' '.join(frame_atoms)
    frame = f'name {frame_str}'
    ref = md.Merge(md.Universe(GRO).select_atoms(selection).residues[0].atoms)
    ref_sel = ref.atoms.select_atoms(frame)    ### Frame for the virtual site stuff
    ref_ats = [i for i, item in enumerate(ref.atoms.names.tolist()) if item in frame_atoms]
    print(ref_ats)
    u = md.Universe(TPR, XTC)
    chols = u.select_atoms(selection).residues

    ### Write aligned trajectory
    aligned_traj = md.Writer('aligned_aa.xtc', n_atoms=len(ref.atoms))
    for frame in tqdm(u.trajectory):
        for chol in chols.atoms.fragments:
            mobile = md.Merge(chol)
            align.alignto(mobile.atoms[ref_ats], ref_sel)
            aligned_traj.write(mobile.atoms)
    aligned_traj.close()
    chol.atoms.write('chol_aa.gro')

    ### Write averaged structure
    aligned = md.Universe('chol_aa.gro', 'aligned_aa.xtc')
    allpos = np.empty((len(aligned.trajectory), len(aligned.atoms), 3), dtype=np.float32)
    for i, frame in tqdm(enumerate(aligned.trajectory), total=len(aligned.trajectory)):
        allpos[i] = frame.positions
    averaged = allpos.mean(axis=0)
    mobile.atoms.positions = averaged
    mobile.atoms.write('averaged_aa.gro')


def run_VS(syst, masses, frame, mapping, pos, bead_names):
    aaats = syst.atoms ## if ignoring atoms we need to apply a selection here
    # u_cg = md.Merge(aaats[:len(mapping)])
    u_cg = md.Universe('cg_good.gro')
    u_cg.add_TopologyAttr('masses')
    cgats = u_cg.atoms
    cgats.masses = masses
    # pos = [ag.center_of_geometry() for ag in mapping]
    # cgats.positions = pos
    # cgats.names = bead_names

    vs = cgats ## if ignoring atoms we need to apply a selection here
    #cgats.write('vs0.gro')

    # # frame = np.array([8, 2, 1])
    # aaats.positions -= cgats.positions[frame[0]] 
    # cgats.positions -= cgats.positions[frame[0]]
    # #cgats.write('vs1.gro')

    # #long_ax = vs.principal_axes()[-1]
    # long_ax = (vs.positions[frame[1]] + vs.positions[frame[2]])/2
    # long_ax /= np.linalg.norm(long_ax)
    # angle = np.degrees(mdamath.angle(long_ax, [1,0,0]))
    # ax = transformations.rotaxis(long_ax, [1,0,0])
    # cgats.rotateby(angle, ax, point=[0,0,0])
    # aaats.rotateby(angle, ax, point=[0,0,0])
    # #cgats.write('vs2.gro')
    # mid_ax = cgats.positions[frame[1]] - cgats.positions[frame[2]]
    # print(f'mid_ax; {mid_ax}')
    # mid_ax[0] = 0
    # mid_ax /= np.linalg.norm(mid_ax)
    # angle = np.degrees(mdamath.angle(mid_ax, [0,1,0]))
    # ax = transformations.rotaxis(mid_ax, [0,1,0])
    # cgats.rotateby(angle, ax, point=[0,0,0])
    # aaats.rotateby(angle, ax, point=[0,0,0])
    # # force frame flatness in z
    # cgats[frame].positions *= [1,1,0]
    # #cgats.write('cg.gro')
    # #aaats.write('aa.gro')
    # print(cgats.positions)

    # # ring_ats = sum(mapping)
    # ring_ats = aaats
    # aapos = ring_ats.positions
    # ring_ats.positions *= [1,1,0]
    # #vs.write('vs5.gro')
    # #print(vs.positions)
    # aa_flat_com = ring_ats.center_of_mass()
    # ring_ats.positions -= aa_flat_com
    # cgats.positions -= aa_flat_com
    # MoIt = ring_ats.moment_of_inertia()
    # ring_ats.positions = aapos - aa_flat_com
    #cgats.write('cg.gro')
    #aaats.write('aa.gro')
    
    #cg_pos = cgats.positions
    #cg_flat_pos = cg_pos * [1,1,0]
    #
    #cgats.positions = cg_flat_pos
    #cg_flat_com = cgats.center_of_mass() 
    #cgats.positions -= cg_flat_com
    #cg_pos -= cg_flat_com
    #ring_ats.positions -= cg_flat_com
    #MoIt = cgats.moment_of_inertia()
    #cgats.positions = cg_pos

    # t = totmass = vs.total_mass()

    # print(f'Total mass: {totmass}')
    # pos = cgats.positions[frame,:2]
    # m1 = ((t*pos[2,0]/(pos[1,0]-pos[2,0])) - (t*pos[2,1]/(pos[1,1]-pos[2,1]))) / ((pos[2,0]-pos[0,0])/(pos[1,0]-pos[2,0]) - (pos[2,1]-pos[0,1])/(pos[1,1]-pos[2,1]))
    # m2 = ((t*pos[2,0]/(pos[0,0]-pos[2,0])) - (t*pos[2,1]/(pos[0,1]-pos[2,1]))) / ((pos[2,0]-pos[1,0])/(pos[0,0]-pos[2,0]) - (pos[2,1]-pos[1,1])/(pos[0,1]-pos[2,1]))
    # m3 = totmass - m1 - m2

    # masses = np.zeros(len(vs)) 
    # masses[frame] = [m1, m2, m3]
    # vs.masses = masses
    # print("target MoI")
    # print(MoIt)
    # print("Masses:")
    # print(vs.masses)
    # #print "final MoI myway"
    # print("final MoI MDAnalysis way")
    # print(vs.moment_of_inertia())
    # print("final frame COM:")
    # print(vs.center_of_mass())

    # Done with MoI

    # back to centering on frame[0]
    vs.positions -= vs.positions[frame[0]]
    vecs = vs.positions[frame[1:]]
    #cross = [0, 0, -1]
    cross = np.cross(*vecs)/10
    #cross = cross/np.linalg.norm(cross) # 1nm unit vector
    vecs = np.row_stack((vecs, cross)) 
    print(f'Basis vecs: {vecs}')
    inv_vecs = np.linalg.inv(vecs)
    factors = np.dot(vs.positions, inv_vecs)

    atnums = vs.indices + 1
    framenums = atnums[frame]
    frame_txt = "  ".join(framenums.astype('str'))
    itp_out=[]
    
    print("\n")
    print("[ constraints ]")
    itp_out.append("[ constraints ]\n")
    for i, j in itertools.combinations(framenums, 2):
        dist = np.linalg.norm(vs.positions[i-1] - vs.positions[j-1])*0.1
        print(" {}  {}   1  {:.5f}".format(i, j, dist))
        itp_out.append(" {}  {}   1  {:.5f} \n".format(i, j, dist))

    print()

    print("[ virtual_sites3 ]")
    itp_out.append("\n")
    itp_out.append("[ virtual_sites3 ] \n")
    for atnum, factor in zip(atnums, factors):
        if atnum not in framenums:
            if False: #factor[2] < 0.3:
                print(" {}   {}  1  {:.5f}  {:.5f}".format(atnum, frame_txt, *factor[:2]))
                itp_out.append(" {}   {}  1  {:.5f}  {:.5f} \n".format(atnum, frame_txt, *factor[:2]))
            else:
                print(" {}   {}  4  {:.5f}  {:.5f}  {:.5f}".format(atnum, frame_txt, *factor))
                itp_out.append(" {}   {}  4  {:.5f}  {:.5f}  {:.5f} \n".format(atnum, frame_txt, *factor))

    return itp_out, vs.masses
    
def names_to_ag(u, names):
    return u.atoms[(np.array(names)[:,None] == u.atoms.names[None,:]).nonzero()[1]]