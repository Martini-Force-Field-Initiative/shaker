from pathlib import Path
import nglview as nv
import MDAnalysis as md

'''
Collection of functions to assist in nglview rendering.
'''

def render_mapping(cg_mapped_gro, cg_mapped_xtc, 
                   aa_gro, aa_xtc,
                  size='900px'):
    """
    Render an overlay of an atomistic trajectory and its coarse-grained mapping.

    This function loads two trajectories into an NGL viewer:
    an atomistic (AA) system and a mapped coarse-grained (CG) system.
    The AA system is displayed as ball-and-stick, while the CG system
    is displayed as spheres representing beads. 

    Parameters
    ----------
    cg_mapped_gro : str or Path
        GRO file containing the topology/coordinates of the mapped CG system.
    cg_mapped_xtc : str or Path
        XTC trajectory corresponding to the mapped CG system.
    aa_gro : str or Path
        GRO file containing the topology/coordinates of the atomistic system.
    aa_xtc : str or Path
        XTC trajectory corresponding to the atomistic system.
    size : str, optional
        Size of the NGL viewer widget (CSS format), e.g. "900px".
        The viewer is rendered as a square of this size.

    Returns
    -------
    nglview.NGLWidget
        Interactive viewer containing the AA and CG trajectories overlayed.

    """
    
    cg_mapped_gro = Path(cg_mapped_gro).resolve()
    cg_mapped_xtc = Path(cg_mapped_xtc).resolve()
    aa_gro = Path(aa_gro).resolve()
    aa_xtc = Path(aa_xtc).resolve()
    
    view = nv.NGLWidget()
    u_aa = md.Universe(str(aa_gro), str(aa_xtc))
    u_cg = md.Universe(str(cg_mapped_gro), str(cg_mapped_xtc))

    view = nv.NGLWidget()
    view.add_trajectory(u_aa)
    view.add_trajectory(u_cg)

    # clear defaults
    view[0].clear_representations()
    view[1].clear_representations()
    
    # atomistic molecule
    view[0].add_representation("ball+stick", selection="NOT DOT")

    # CG molecule
    view[1].add_representation("spacefill", selection="NOT DOT", opacity=1.0, radius=.9)

    view.center()
    view._set_size(size, size)
    view.camera = "orthographic"
    view.render_image(frame=True, trim=True, transparent=True, factor=6)

    return view

def render_connely_surface(SASA_folder='./SASA', size='900px'):
    """
    Render overlapping Connolly (SASA) surfaces for atomistic and coarse-grained models.

    This function loads two structures containing precomputed SASA dot surfaces
    (typically generated with `gmx sasa`). The atomistic (AA) and coarse-grained (CG)
    systems are displayed together to visually compare their connely surfaces.

    Parameters
    ----------
    SASA_folder : str or Path, optional
        Directory containing the SASA surface files. The function expects
        the following structure:

            SASA_folder/
                AA/surface.pdb
                CG/surface.pdb

        Each `surface.pdb` must contain the molecular coordinates and
        pseudo-atoms with residue name `DOT` representing the sampled
        Connolly surface.
    size : str, optional
        Size of the NGL viewer widget (CSS format), e.g. "900px".
        The viewer is rendered as a square of this size.

    Returns
    -------
    nglview.NGLWidget
        Interactive viewer displaying the AA and CG structures with their
        corresponding SASA dot surfaces.

    Notes
    -----
    - AA atoms are shown as ball-and-stick.
    - CG beads are shown as spheres.
    - AA surface dots are colored blue.
    - CG surface dots are colored red.
    """

    SASA_folder = Path(SASA_folder).resolve()

    view = nv.NGLWidget()
    
    # component 0: atomistic structure + its DOT surface
    view.add_component(f"{SASA_folder}/AA/surface.pdb")
    # component 1: CG structure + its DOT surface
    view.add_component(f"{SASA_folder}/CG_Mapped/surface.pdb")
    
    # clear defaults
    view[0].clear_representations()
    view[1].clear_representations()
    
    # atomistic molecule
    view[0].add_representation("ball+stick", selection="NOT DOT")
    view[0].add_representation(
        "spacefill",
        selection="DOT",
        radius=0.11,
        opacity=0.35,
        color="blue"
    )
    # CG molecule
    view[1].add_representation("spacefill", selection="NOT DOT", opacity=1.0, radius=.9)
    view[1].add_representation(
        "spacefill",
        selection="DOT",
        radius=0.11,
        opacity=0.45,
        color="red"
    )
    view.center()

    view._set_size(size, size)
    view.camera = "orthographic"
    view.render_image(frame=True,trim=True,transparent=True,factor=6)
    return view