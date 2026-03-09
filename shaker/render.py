from collections import Counter
from pathlib import Path

import nglview as nv
import numpy as np

import MDAnalysis as md
from MDAnalysis.analysis import align

from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Chem.Draw import rdMolDraw2D

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
    
    view.add_component(f"{SASA_folder}/AA/surface.pdb")
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


def render_ensemble(gro, xtc, sel="all", step=None, n_frames=None, size="400px"):
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
        Default is "all".
    step : int, optional
        Show every `step` frames.
    n_frames : int, optional
        Total number of frames to display, sampled evenly.
    size : str, optional
        Viewer size (CSS units), default "400px".

    Returns
    -------
    nglview.NGLWidget
        Interactive NGL viewer showing multiple aligned frames.
    """

    gro = Path(gro).resolve()
    xtc = Path(xtc).resolve()

    u = md.Universe(gro, xtc)
    ref = md.Universe(gro, xtc)
    ref.trajectory[0]

    align.AlignTraj(u, ref, select=sel, in_memory=True).run()

    ag = u.select_atoms(sel)

    if step is None and n_frames is None:
        step = max(len(u.trajectory) // 20, 1)

    if n_frames is not None:
        idx = np.linspace(0, len(u.trajectory) - 1, n_frames).astype(int)
    else:
        idx = range(0, len(u.trajectory), step)

    view = nv.NGLWidget()

    for i in idx:
        u.trajectory[i]

        coords = u.atoms.positions.copy()
        coords -= ag.center_of_mass()

        snap = md.Merge(u.atoms)
        snap.atoms.positions = coords

        comp = view.add_trajectory(snap)
        comp.clear_representations()
        comp.add_representation("licorice", selection="all", opacity=0.5)

    view.center()
    view._set_size(size, size)
    view.camera = "orthographic"

    return view
    
def render_2dMapping(pdb_file, resname, bead_assignments, bead_names,
                     out_svg="cg_overlay.svg", size=(950, 480), mode="connected",         
                     bead_r=20.0, conn_r=12.0, alpha=0.3, line_w=2.0,
                     font_size=14, label_dy=20.0,):
    """
    Render a 2D atomistic structure with an overlaid coarse-grained (CG) mapping.
    
    The atomistic molecule is read from a PDB file and depicted in 2D using RDKit.
    CG beads defined in `bead_assignments` are then visualized on top of the
    structure using one of several styles (centroid circles, atom blobs, or
    connectivity-following shading).
    
    Hydrogen atoms listed in bead assignments are automatically mapped onto their
    bonded heavy atoms so that bead centroids and connectivity are computed
    correctly.
    
    Parameters
    ----------
    pdb_file : str
        Input PDB file containing the molecule.
    resname : str
        Residue name identifying the molecule in the PDB.
    bead_assignments : list[list[str]]
        Atom names assigned to each CG bead (repeated names act as weights).
    bead_names : list[str]
        Labels for each bead.
    out_svg : str, optional
        Output SVG filename.
    mode : {"circle","atomblobs","connected","both"}, optional
        CG rendering style.
    size, bead_r, conn_r, alpha, line_w, font_size, label_dy : optional
        Visualization parameters controlling canvas size, bead radius,
        transparency, outlines, and label placement.
    
    Writes
    ------
    SVG file with the atomistic structure and CG mapping overlay.
    """

    if mode not in {"circle", "atomblobs", "connected", "both"}:
        raise ValueError("mode must be one of: 'circle', 'atomblobs', 'connected', 'both'")

    molH, mol = _load_mols(pdb_file)
    idxH = _atom_name_map(molH, resname)
    idx = _atom_name_map(mol, resname)
    hmap = _h_to_heavy_map(molH, resname)
    colors = _palette(len(bead_assignments))

    ## Find if any atoms assigned in the mapping are missing in the pdb supplied.
    missing = sorted({a for bead in bead_assignments for a in bead} - set(idxH))
    if missing:
        preview = ", ".join(missing[:20])
        tail = " ..." if len(missing) > 20 else ""
        raise ValueError(f"Missing atom names in residue {resname}: {preview}{tail}")

    # Base AA drawing
    w, h = size
    drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
    opts = drawer.drawOptions()
    opts.explicitMethyl = False
    opts.addAtomIndices = False
    opts.addStereoAnnotation = False

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    overlay = ['<g id="cg_overlay">']

    for bead, label, (r, g, b) in zip(bead_assignments, bead_names, colors):
        weights = Counter(hmap.get(a, a) if a.startswith("H") else a for a in bead)
        weights = Counter({name: wt for name, wt in weights.items() if name in idx})

        if not weights:
            continue

        # Weighted centroid in draw coordinates
        sx = sy = sw = 0.0
        for name, wt in weights.items():
            pt = drawer.GetDrawCoords(idx[name])
            sx += wt * pt.x
            sy += wt * pt.y
            sw += wt
        cx, cy = sx / sw, sy / sw

        heavy_names = _bead_heavy_names(bead, hmap)
        atom_pts = {
            name: (drawer.GetDrawCoords(idx[name]).x, drawer.GetDrawCoords(idx[name]).y)
            for name in heavy_names
            if name in idx }
        edges = _bead_edges(mol, heavy_names, idx)

        fill = _rgba(r, g, b, alpha)
        edge = _rgb(r, g, b)

        # Connectivity-following shading
        if mode in ("connected", "both"):
            if edges:
                path = " ".join(
                    f'M {atom_pts[a][0]:.2f},{atom_pts[a][1]:.2f} L {atom_pts[b2][0]:.2f},{atom_pts[b2][1]:.2f}'
                    for a, b2 in edges )
                overlay.append(
                    f'<path d="{path}" fill="none" stroke="{fill}" '
                    f'stroke-width="{2*conn_r:.2f}" stroke-linecap="round" stroke-linejoin="round" />')
            elif len(atom_pts) == 1:
                x, y = next(iter(atom_pts.values()))
                overlay.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{conn_r:.2f}" '
                    f'fill="{fill}" stroke="none" />')

        # One circle per heavy atom
        if mode == "atomblobs":
            for x, y in atom_pts.values():
                overlay.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{conn_r:.2f}" '
                    f'fill="{fill}" stroke="{edge}" stroke-width="{line_w:.2f}" />')

        # Centroid circle in the COG of the mapped bead.
        if mode in ("circle", "both"):
            overlay.append(
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{bead_r:.2f}" '
                f'fill="{fill}" stroke="{edge}" stroke-width="{line_w:.2f}" />')

        tx = cx + max(bead_r, conn_r) + 2
        ty = cy + label_dy

        overlay.append(
            f'<text x="{tx:.2f}" y="{ty:.2f}" font-family="sans-serif" '
            f'font-size="{font_size}" font-weight="bold" dominant-baseline="middle" '
            f'text-anchor="start" stroke="white" stroke-width="2" fill="white">{label}</text>')
        overlay.append(
            f'<text x="{tx:.2f}" y="{ty:.2f}" font-family="sans-serif" '
            f'font-size="{font_size}" font-weight="bold" dominant-baseline="middle" '
            f'text-anchor="start" fill="{edge}">{label}</text>')

    overlay.append("</g>")
    svg = svg.replace("</svg>", "\n".join(overlay) + "\n</svg>")
    Path(out_svg).write_text(svg, encoding="utf-8")


def _load_mols(pdb_file):
    """Return (molH, mol): sanitized molecule with Hs, and no-H molecule with 2D coords."""
    molH = Chem.MolFromPDBFile(pdb_file, sanitize=False, removeHs=False)
    if molH is None:
        raise ValueError(f"Could not read PDB: {pdb_file}")

    rdDetermineBonds.DetermineBonds(molH)
    rdDetermineBonds.DetermineBondOrders(molH)
    Chem.SanitizeMol(molH)

    mol = Chem.RemoveHs(molH)
    AllChem.Compute2DCoords(mol)
    return molH, mol


def _atom_name_map(mol, resname):
    """Map PDB atom name -> RDKit atom index for one residue name."""
    out = {}
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info and info.GetResidueName().strip() == resname:
            out[info.GetName().strip()] = atom.GetIdx()
    return out


def _h_to_heavy_map(molH, resname):
    """Map each H atom name to its bonded heavy atom name."""
    out = {}
    for atom in molH.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if not info or info.GetResidueName().strip() != resname:
            continue
        if atom.GetAtomicNum() != 1 or not atom.GetNeighbors():
            continue

        nb = atom.GetNeighbors()[0]
        nbinfo = nb.GetPDBResidueInfo()
        if nbinfo and nbinfo.GetResidueName().strip() == resname:
            out[info.GetName().strip()] = nbinfo.GetName().strip()
    return out


def _palette(n):
    colors = [(0.12, 0.47, 0.71),  # blue
              (1.00, 0.50, 0.05),  # orange
              (0.17, 0.63, 0.17),  # green
              (0.84, 0.15, 0.16),  # red
              (0.58, 0.40, 0.74),  # purple
              (0.55, 0.34, 0.29),  # brown
              (0.89, 0.47, 0.76),  # pink
              (0.50, 0.50, 0.50),  # grey
              (0.74, 0.74, 0.13),  # olive
              (0.09, 0.75, 0.81)]  # cyan
    return [colors[i % len(colors)] for i in range(n)]


def _bead_heavy_names(bead, hmap):
    """Unique heavy atoms contributing to a bead, with Hs folded onto bonded heavy atoms."""
    out = []
    seen = set()
    for name in bead:
        heavy = hmap.get(name, name) if name.startswith("H") else name
        if heavy.startswith("H") or heavy in seen:
            continue
        seen.add(heavy)
        out.append(heavy)
    return out


def _bead_edges(mol, names, idx_map):
    """Bonded heavy-atom pairs within one bead."""
    names = {n for n in names if n in idx_map}
    edges = set()

    for name in names:
        atom = mol.GetAtomWithIdx(idx_map[name])
        for nb in atom.GetNeighbors():
            info = nb.GetPDBResidueInfo()
            if not info:
                continue
            nb_name = info.GetName().strip()
            if nb_name in names:
                edges.add(tuple(sorted((name, nb_name))))
                
    return sorted(edges)


def _rgb(r, g, b):
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def _rgba(r, g, b, a):
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a})"