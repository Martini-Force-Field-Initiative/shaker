"""3D visualization of atomistic and coarse-grained trajectories via nglview."""

import warnings
from pathlib import Path

import MDAnalysis as md
import nglview as nv
import numpy as np
from MDAnalysis.analysis import align


def render_mapping(cg_mapped_gro, cg_mapped_xtc, aa_gro, aa_xtc, size="900px"):
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

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Element information is missing, elements attribute will not be populated.*",
            category=UserWarning,
            module=r"MDAnalysis\.topology\.PDBParser",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Reload offsets from trajectory",
            category=UserWarning,
        )
        u_aa = md.Universe(str(aa_gro), str(aa_xtc))
        u_cg = md.Universe(str(cg_mapped_gro), str(cg_mapped_xtc))

    view = nv.NGLWidget()
    view.add_trajectory(u_aa)
    view.add_trajectory(u_cg)

    view[0].clear_representations()
    view[1].clear_representations()

    view[0].add_representation("ball+stick", selection="NOT DOT")
    view[1].add_representation(
        "spacefill", selection="NOT DOT", opacity=1.0, radius=0.9
    )

    view.center()
    view._set_size(size, size)
    view.camera = "orthographic"
    view.render_image(frame=True, trim=True, transparent=True, factor=6)

    return view


def render_connely_surface(
    SASA_folder="./SASA", aa_subfolder="AA", cg_subfolder="CG_Mapped", size="900px"
):
    """
    Render overlapping Connolly (SASA) surfaces for atomistic and coarse-grained models.

    This function loads two structures containing precomputed SASA dot surfaces
    (typically generated with `gmx sasa`). The atomistic (AA) and coarse-grained (CG)
    systems are displayed together to visually compare their Connolly surfaces.

    Parameters
    ----------
    SASA_folder : str or Path, optional
        Directory containing the SASA surface files. The function expects
        the following structure::

            SASA_folder/
                <aa_subfolder>/surface.pdb
                <cg_subfolder>/surface.pdb

        Each ``surface.pdb`` must contain the molecular coordinates and
        pseudo-atoms with residue name ``DOT`` representing the sampled
        Connolly surface.
    aa_subfolder : str, optional
        Name of the subdirectory containing the AA surface file.
        Default is ``'AA'``.
    cg_subfolder : str, optional
        Name of the subdirectory containing the CG surface file.
        Default is ``'CG_Mapped'``.
    size : str, optional
        Size of the NGL viewer widget (CSS format), e.g. ``"900px"``.
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
    aa_surface = str(SASA_folder / aa_subfolder / "surface.pdb")
    cg_surface = str(SASA_folder / cg_subfolder / "surface.pdb")

    view = nv.NGLWidget()

    view.add_component(aa_surface)
    view.add_component(cg_surface)

    view[0].clear_representations()
    view[1].clear_representations()

    view[0].add_representation("ball+stick", selection="NOT DOT")
    view[0].add_representation(
        "spacefill", selection="DOT", radius=0.11, opacity=0.35, color="blue"
    )

    view[1].add_representation(
        "spacefill", selection="NOT DOT", opacity=1.0, radius=0.9
    )
    view[1].add_representation(
        "spacefill", selection="DOT", radius=0.11, opacity=0.45, color="red"
    )

    view.center()
    view._set_size(size, size)
    view.camera = "orthographic"
    view.render_image(trim=True, transparent=True, factor=6)
    return view


def render_ensemble(
    gro, xtc, sel="all", render_sel="all", step=None, n_frames=None, size="400px"
):
    """
    Visualize an aligned ensemble of trajectory frames simultaneously.

    Parameters
    ----------
    gro : str or Path
        Topology file (e.g. GRO or PDB).
    xtc : str or Path
        Trajectory file (e.g. XTC).
    sel : str, optional
        Atom selection used for alignment and centering.
        Default is ``"all"``.
    render_sel : str, optional
        Atom selection of what to actually display (e.g. to exclude
        solvent/ions or show only a fragment), independent of ``sel``.
        Centering still uses ``sel``'s center of mass, so the rendered
        subset stays aligned to the same reference point across frames.
        Default is ``"all"``.
    step : int, optional
        Show every ``step`` frames.
    n_frames : int, optional
        Total number of frames to display, sampled evenly.
    size : str, optional
        Viewer size (CSS units), default ``"400px"``.

    Returns
    -------
    nglview.NGLWidget
        Interactive NGL viewer showing multiple aligned frames.
    """

    gro = Path(gro).resolve()
    xtc = Path(xtc).resolve()

    u = md.Universe(str(gro), str(xtc))
    ref = md.Universe(str(gro))

    align.AlignTraj(u, ref, select=sel, in_memory=True).run()

    ag = u.select_atoms(sel)
    render_ag = u.select_atoms(render_sel)

    if step is None and n_frames is None:
        step = max(len(u.trajectory) // 20, 1)

    if n_frames is not None:
        frame_idx = np.linspace(0, len(u.trajectory) - 1, n_frames).astype(int)
    else:
        frame_idx = range(0, len(u.trajectory), step)

    view = nv.NGLWidget()

    for i in frame_idx:
        u.trajectory[i]

        coords = render_ag.positions.copy()
        coords -= ag.center_of_mass()

        snap = md.Merge(render_ag)
        snap.atoms.positions = coords

        comp = view.add_trajectory(snap)
        comp.clear_representations()
        comp.add_representation("licorice", selection="all", opacity=0.5)

    view.center()
    view._set_size(size, size)
    view.camera = "orthographic"

    return view
