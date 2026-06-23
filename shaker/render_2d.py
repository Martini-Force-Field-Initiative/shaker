"""2D chemical structure drawing with coarse-grained mapping overlay."""

from collections import Counter
import math
from pathlib import Path
import warnings

from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Chem.Draw import rdMolDraw2D

from .helper import _category_from_type


def render_2dMapping(pdb_file, resname, mapping,
                     out_svg="cg_overlay.svg", size=(950, 480), mode="connected",
                     bead_r=30.0, conn_r=30.0, alpha=0.2, line_w=3.0,
                     font_size=35, label_dy=20.0, net_charge=None):
    """
    Render a 2D atomistic structure with an overlaid coarse-grained (CG) mapping.

    The atomistic molecule is read from a PDB file and depicted in 2D using RDKit.
    CG beads defined in ``mapping`` are then visualized on top of the structure
    using one of several styles (centroid circles, atom blobs, or
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
    mapping : dict
        Mapping dictionary in SHAKER format::

            mapping = {
                "RESNAME": {
                    "BEAD1": {"type": "...", "charge": 0, "atoms": [...]},
                    "BEAD2": {"type": "...", "charge": 0, "atoms": [...]},
                }
            }
    out_svg : str, optional
        Output SVG filename.
    mode : {"circle", "atomblobs", "connected", "both"}, optional
        CG rendering style.
    size : tuple of int, optional
        Canvas size in pixels as ``(width, height)``.
    bead_r : float, optional
        Radius of the centroid circle in SVG units, used in "circle"/"both"
        mode and for single-atom beads (no internal bonds) in "connected"
        mode.
    conn_r : float, optional
        Half-width of the connected-mode capsule shading / atom blob radius.
    alpha : float, optional
        Fill opacity for bead overlays.
    line_w : float, optional
        Stroke width for bead outlines.
    font_size : int, optional
        Font size for bead labels.
    label_dy : float, optional
        Vertical offset of labels relative to the bead centroid.
    net_charge : int, optional
        Net formal charge of the molecule, passed to RDKit's bond-order
        determination (required for charged molecules — without it,
        bond-order assignment is attempted assuming a neutral molecule
        and fails or produces incorrect bonds for anything else). If not
        given (default), it is inferred as the sum of the per-bead
        ``"charge"`` entries in ``mapping[resname]``; this requires every
        bead in the mapping to define a ``"charge"``, and the sum must be
        a whole number.

    Returns
    -------
    str
        The SVG source string. The same content is also written to ``out_svg``.
    """

    if mode not in {"circle", "atomblobs", "connected", "both"}:
        raise ValueError("mode must be one of: 'circle', 'atomblobs', 'connected', 'both'")

    if resname not in mapping:
        raise ValueError(f"No mapping found for resname '{resname}'")

    bead_map = mapping[resname]
    _require_bead_key(bead_map, "atoms", resname)

    bead_names       = list(bead_map.keys())
    bead_assignments = [b["atoms"] for b in bead_map.values()]

    if net_charge is None:
        _require_bead_key(bead_map, "charge", resname,
                          hint=" Either add 'charge' to every bead in the "
                               "mapping, or pass net_charge explicitly.")
        charge_sum = sum(b["charge"] for b in bead_map.values())
        if charge_sum != int(charge_sum):
            raise ValueError(
                f"Bead charges for '{resname}' sum to a non-integer net "
                f"charge ({charge_sum}); pass net_charge explicitly.")
        net_charge = int(charge_sum)

    molH, mol = _load_mols(pdb_file, resname, net_charge)
    idxH = _atom_name_map(molH, resname)
    idx  = _atom_name_map(mol,  resname, warn_duplicates=False)

    hmap   = _h_to_heavy_map(molH, resname)
    colors = _bead_colors(bead_map)

    missing = sorted({a for bead in bead_assignments for a in bead} - set(idxH))
    if missing:
        preview = ", ".join(missing[:20])
        tail = " ..." if len(missing) > 20 else ""
        raise ValueError(f"Missing atom names in residue {resname}: {preview}{tail}")

    w, h = size
    drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
    opts = drawer.drawOptions()
    opts.explicitMethyl    = False
    opts.addAtomIndices    = False
    opts.addStereoAnnotation = False
    opts.bondLineWidth = 3.5
    opts.scaleBondWidth = True

    drawer.DrawMolecule(mol)

    draw_coords = {
        name: (drawer.GetDrawCoords(i).x, drawer.GetDrawCoords(i).y)
        for name, i in idx.items()
    }

    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    shading_layer = []
    label_layer   = []
    label_specs   = []

    for bead_idx, (bead, label, (r, g, b)) in enumerate(zip(bead_assignments, bead_names, colors)):
        weights = Counter(hmap.get(a, a) for a in bead)
        weights = Counter({name: wt for name, wt in weights.items() if name in idx})

        dropped = {hmap.get(a, a) for a in bead} & set(idxH) - set(idx)
        if dropped:
            warnings.warn(
                f"Bead '{label}': atoms {sorted(dropped)} exist in the H-mol "
                f"but are absent from the heavy-atom mol; centroid may be shifted.",
                UserWarning, stacklevel=2)

        if not weights:
            continue

        sx = sy = sw = 0.0
        for name, wt in weights.items():
            x, y = draw_coords[name]
            sx += wt * x
            sy += wt * y
            sw += wt
        cx, cy = sx / sw, sy / sw

        heavy_names = _bead_heavy_names(bead, hmap, mol, idx)
        atom_pts    = {name: draw_coords[name] for name in heavy_names if name in draw_coords}
        edges       = _bead_edges(mol, heavy_names, idx)

        fill        = _rgba(r, g, b, alpha)
        label_color = _rgb(r, g, b)
        outline     = _rgb(*_darken(r, g, b))

        if mode in ("connected", "both"):
            if edges:
                # Real closed capsule shapes (not stroked lines), so the
                # fill/stroke are non-overlapping like a circle's and the
                # bond underneath shows through. A multi-branch bead has
                # several capsules in one path; stroking that directly
                # would draw each capsule's own boundary independently,
                # crossing at shared joints. Instead, mask a wider, plain
                # "outline" shape down to a clean ring — wide-minus-narrow,
                # computed per pixel rather than via separate strokes — so
                # the outline has no seam regardless of how many branches
                # meet at a point.
                narrow = " ".join(
                    _capsule_path(atom_pts[a], atom_pts[b2], conn_r)
                    for a, b2 in edges
                )
                wide = " ".join(
                    _capsule_path(atom_pts[a], atom_pts[b2], conn_r + line_w)
                    for a, b2 in edges
                )
                mask_id = f"capsule_mask_{bead_idx}"
                shading_layer.append(
                    f'<mask id="{mask_id}">'
                    f'<rect x="0" y="0" width="{w}" height="{h}" fill="white" />'
                    f'<path d="{narrow}" fill="black" /></mask>')
                shading_layer.append(
                    f'<path d="{wide}" fill="{outline}" mask="url(#{mask_id})" />')
                shading_layer.append(f'<path d="{narrow}" fill="{fill}" />')
            else:
                shading_layer.append(
                    f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{bead_r:.2f}" '
                    f'fill="{fill}" stroke="{outline}" stroke-width="{line_w:.2f}" />')

        if mode == "atomblobs":
            for x, y in atom_pts.values():
                shading_layer.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{conn_r:.2f}" '
                    f'fill="{fill}" stroke="{outline}" stroke-width="{line_w:.2f}" />')

        if mode in ("circle", "both"):
            shading_layer.append(
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{bead_r:.2f}" '
                f'fill="{fill}" stroke="{outline}" stroke-width="{line_w:.2f}" />')

        label_offset = max(bead_r, conn_r) + 2
        label_width  = len(label) * font_size * 0.6
        if cx + label_offset + label_width > w:
            tx     = cx - label_offset
            anchor = "end"
        else:
            tx     = cx + label_offset
            anchor = "start"
        ty = cy + label_dy

        label_specs.append({"label": label, "tx": tx, "ty": ty,
                            "anchor": anchor, "edge": label_color, "width": label_width})

    _avoid_label_collisions(label_specs, font_size)

    for spec in label_specs:
        label_layer.append(
            f'<text x="{spec["tx"]:.2f}" y="{spec["ty"]:.2f}" font-family="sans-serif" '
            f'font-size="{font_size}" font-weight="bold" dominant-baseline="middle" '
            f'text-anchor="{spec["anchor"]}" stroke="white" stroke-width="2" fill="white">'
            f'{spec["label"]}</text>')
        label_layer.append(
            f'<text x="{spec["tx"]:.2f}" y="{spec["ty"]:.2f}" font-family="sans-serif" '
            f'font-size="{font_size}" font-weight="bold" dominant-baseline="middle" '
            f'text-anchor="{spec["anchor"]}" fill="{spec["edge"]}">{spec["label"]}</text>')

    overlay = ['<g id="cg_overlay">'] + shading_layer + label_layer + ['</g>']

    parts = svg.rsplit("</svg>", 1)
    svg   = parts[0] + "\n".join(overlay) + "\n</svg>" + parts[1]

    Path(out_svg).write_text(svg, encoding="utf-8")
    return svg


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_mols(pdb_file, resname, net_charge):
    """
    Return (molH, mol): sanitized molecule with Hs, and no-H molecule with
    2D coords, containing only the atoms belonging to ``resname``.

    The target residue is isolated into its own molecule before bond
    determination. Without this, any other residue present in the PDB
    (solvent, ions, a second copy of the molecule) would be folded into
    the same bond-order/charge search as the target molecule, which can
    throw off both the charge balance and the inferred connectivity.

    If the target residue is covalently bonded to another residue (e.g.
    an amino acid mid-chain), each such bond is capped with a generic
    placeholder atom rather than dropped or followed — see
    ``_isolate_residue`` and ``_label_caps_as_r``.
    """
    molH_full = Chem.MolFromPDBFile(pdb_file, sanitize=False, removeHs=False)
    if molH_full is None:
        raise ValueError(f"Could not read PDB: {pdb_file}")

    target_idx = [
        atom.GetIdx() for atom in molH_full.GetAtoms()
        if atom.GetPDBResidueInfo() and _norm_resname(atom.GetPDBResidueInfo()) == resname
    ]
    res_ids = {
        (info.GetResidueNumber(), info.GetChainId())
        for info in (molH_full.GetAtomWithIdx(i).GetPDBResidueInfo() for i in target_idx)
    }
    if len(res_ids) == 0:
        present = sorted({
            _norm_resname(atom.GetPDBResidueInfo())
            for atom in molH_full.GetAtoms()
            if atom.GetPDBResidueInfo()
        })
        raise ValueError(
            f"No residue with resname '{resname}' found in {pdb_file}. "
            f"Residue names present: {present}")
    if len(res_ids) > 1:
        raise ValueError(
            f"render_2dMapping expects exactly 1 residue with resname '{resname}' "
            f"in {pdb_file}, found {len(res_ids)}")

    molH, cap_indices = _isolate_residue(molH_full, target_idx)

    rdDetermineBonds.DetermineBonds(molH, charge=net_charge)
    Chem.SanitizeMol(molH)

    molH = _label_caps_as_r(molH, cap_indices)

    mol = Chem.RemoveHs(molH)
    AllChem.Compute2DCoords(mol)
    return molH, mol


def _isolate_residue(molH_full, target_idx):
    """
    Build a fresh, bond-free molecule containing ``target_idx`` plus one
    capping atom for every bond that crosses into a different residue.

    Each cap is added as a generic monovalent placeholder (atomic number
    1) at the real neighbor atom's position, so DetermineBonds sees a
    valence-complete fragment without needing that neighbor's own
    bonding context (which would otherwise cascade into needing its
    other neighbors too). ``_label_caps_as_r`` converts these caps into
    display-only "R" atoms after bond determination succeeds.

    Returns (mol, cap_indices): the isolated molecule (bonds not yet
    determined) and the indices of its capping atoms.
    """
    conf_full = molH_full.GetConformer()
    target_set = set(target_idx)
    cap_positions = [
        conf_full.GetAtomPosition(nb.GetIdx())
        for old_i in target_idx
        for nb in molH_full.GetAtomWithIdx(old_i).GetNeighbors()
        if nb.GetIdx() not in target_set
    ]

    rw = Chem.RWMol()
    conf = Chem.Conformer(len(target_idx) + len(cap_positions))

    for new_i, old_i in enumerate(target_idx):
        src = molH_full.GetAtomWithIdx(old_i)
        new_atom = Chem.Atom(src.GetAtomicNum())
        info = src.GetPDBResidueInfo()
        if info is not None:
            new_atom.SetMonomerInfo(info)
        rw.AddAtom(new_atom)
        conf.SetAtomPosition(new_i, conf_full.GetAtomPosition(old_i))

    cap_indices = []
    for j, pos in enumerate(cap_positions):
        rw.AddAtom(Chem.Atom(1))
        cap_indices.append(len(target_idx) + j)
        conf.SetAtomPosition(len(target_idx) + j, pos)

    rw.AddConformer(conf, assignId=True)
    return rw.GetMol(), cap_indices


def _label_caps_as_r(molH, cap_indices):
    """
    Convert capping placeholder atoms (added by ``_isolate_residue``) from
    hydrogen into dummy "R" atoms for display, after bond determination
    and sanitization have already used their real (monovalent) valence.
    """
    if not cap_indices:
        return molH
    rw = Chem.RWMol(molH)
    for ci in cap_indices:
        atom = rw.GetAtomWithIdx(ci)
        atom.SetAtomicNum(0)
        atom.SetNoImplicit(True)
        atom.SetProp("atomLabel", "R")
    mol = rw.GetMol()
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
    return mol


def _norm_resname(info):
    """
    Normalise a residue name from an RDKit PDBResidueInfo object.

    RDKit stores residue names in a fixed-width field; ``strip()`` handles the
    common trailing-space padding. Returns an empty string when ``info`` is None.
    """
    return info.GetResidueName().strip() if info else ""


def _require_bead_key(bead_map, key, resname, hint=""):
    """
    Raise a clear ValueError listing any beads in ``bead_map`` missing
    ``key``, instead of letting a later ``bead[key]`` raise a bare KeyError.
    """
    missing = [name for name, b in bead_map.items() if key not in b]
    if missing:
        raise ValueError(
            f"Beads missing a {key!r} entry for resname '{resname}': {missing}.{hint}")


def _atom_name_map(mol, resname, warn_duplicates=True):
    """
    Map PDB atom name -> RDKit atom index for one residue name.

    ``warn_duplicates`` can be set to False when this is called on a
    molecule whose atoms are a known subset of one already checked (e.g.
    the no-H molecule after the H-inclusive one), to avoid warning about
    the same underlying duplicate names twice.
    """
    out = {}
    duplicates = set()
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info and _norm_resname(info) == resname:
            name = info.GetName().strip()
            if name in out:
                duplicates.add(name)
            out[name] = atom.GetIdx()
    if duplicates and warn_duplicates:
        warnings.warn(
            f"Residue '{resname}' has duplicate atom names {sorted(duplicates)} "
            "(e.g. alternate locations); only the last atom with each name is "
            "used, which may shift bead centroids and connectivity.",
            UserWarning, stacklevel=2)
    return out


def _h_to_heavy_map(molH, resname):
    """Map each H atom name to its bonded heavy atom name."""
    out = {}
    for atom in molH.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if not info or _norm_resname(info) != resname:
            continue
        if atom.GetAtomicNum() != 1 or not atom.GetNeighbors():
            continue

        nb     = atom.GetNeighbors()[0]
        nbinfo = nb.GetPDBResidueInfo()
        if nbinfo and _norm_resname(nbinfo) == resname:
            out[info.GetName().strip()] = nbinfo.GetName().strip()
    return out


def _avoid_label_collisions(label_specs, font_size, min_gap=2.0):
    """
    Nudge label y-positions downward to reduce overlap between bead labels.

    Processes labels in their original (bead) order and pushes each one
    down past any already-placed label whose approximate bounding box it
    would otherwise overlap. Bounding boxes reuse each spec's ``"width"``
    (the same estimate already used for the initial canvas-edge
    placement), so this is a cheap heuristic, not an exact layout solver
    — it breaks up the common case of directly overlapping labels on
    adjacent beads, not every possible collision.

    ``label_specs`` is modified in place: each dict's ``"ty"`` may be
    increased.
    """
    height = font_size * 1.2
    placed = []  # (x_min, x_max, y_min, y_max) of already-placed labels

    for spec in label_specs:
        width = spec["width"]
        if spec["anchor"] == "start":
            x_min, x_max = spec["tx"], spec["tx"] + width
        else:
            x_min, x_max = spec["tx"] - width, spec["tx"]

        ty = spec["ty"]
        moved = True
        while moved:
            moved = False
            y_min, y_max = ty - height / 2, ty + height / 2
            for px_min, px_max, py_min, py_max in placed:
                if x_min < px_max and x_max > px_min and y_min < py_max and y_max > py_min:
                    ty += height + min_gap
                    moved = True
                    break

        spec["ty"] = ty
        placed.append((x_min, x_max, ty - height / 2, ty + height / 2))


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


_MARTINI_CATEGORY_COLORS = {
    "C": (0.50, 0.50, 0.50),  # grey   - apolar
    "P": (0.84, 0.15, 0.16),  # red    - polar
    "N": (0.12, 0.47, 0.71),  # blue   - intermediate (semi-polar/apolar)
    "X": (0.17, 0.63, 0.17),  # green  - halocarbon
    "Q": (1.00, 0.50, 0.05),  # orange - charged ("Q" or "D")
    "U": (0.80, 0.80, 0.80),  # light grey - virtual
}


def _martini_category(bead_type):
    """
    Classify a Martini 3 bead type string into a broad chemical category
    for coloring: C (apolar), P (polar), N (intermediate), X (halocarbon),
    Q (charged — includes "D", e.g. "SD"/"TD"), or U (virtual). Size
    prefix/suffix stripping is delegated to ``helper._category_from_type``.
    Returns None if the category isn't one rendered with a fixed color
    (e.g. water beads "W"/"SW").
    """
    category = _category_from_type(bead_type)
    if category == "D":
        return "Q"
    return category if category in _MARTINI_CATEGORY_COLORS else None


def _bead_colors(bead_map):
    """
    Assign one color per bead, in ``bead_map`` iteration order.

    If every bead defines a "type", beads are colored by Martini chemical
    category (see ``_martini_category``), so e.g. "SC3" and "TC5" share
    the apolar color regardless of size/number/suffix. Types that don't
    map to a known category fall back to a distinct color per unique
    type. If types aren't all present, falls back to one distinct color
    per bead by position, as before.
    """
    bead_defs = list(bead_map.values())
    if bead_defs and all("type" in b for b in bead_defs):
        categories = [_martini_category(b["type"]) for b in bead_defs]
        unknown_types = []
        for b, cat in zip(bead_defs, categories):
            if cat is None and b["type"] not in unknown_types:
                unknown_types.append(b["type"])
        fallback_colors = dict(zip(unknown_types, _palette(len(unknown_types))))
        return [_MARTINI_CATEGORY_COLORS[cat] if cat is not None else fallback_colors[b["type"]]
                for b, cat in zip(bead_defs, categories)]
    return _palette(len(bead_defs))


def _bead_heavy_names(bead, hmap, mol, idx_map):
    """
    Unique heavy atoms contributing to a bead, with Hs folded onto their
    bonded heavy atoms.
    """
    out  = []
    seen = set()
    for name in bead:
        heavy = hmap.get(name, name)
        if heavy in seen:
            continue
        if heavy in idx_map:
            if mol.GetAtomWithIdx(idx_map[heavy]).GetAtomicNum() == 1:
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


def _darken(r, g, b, factor=0.6):
    """
    Darken an (r, g, b) color for use as a bead outline. Pale category
    colors (e.g. "U") would otherwise get an equally pale stroke, making
    the bead look faded; darkening keeps the outline color-coordinated
    with the bead while staying clearly visible against a white background.
    """
    return (r * factor, g * factor, b * factor)


def _capsule_path(p1, p2, r):
    """
    Build a closed SVG sub-path string for a rounded "capsule" (stadium)
    shape: a thick line from ``p1`` to ``p2`` with half-width ``r``,
    capped by semicircles at both ends.

    Used instead of a plain stroked line for connected-mode bead shading:
    a stroked line has no region separate from its stroke, so a
    translucent fill would blend against whatever is drawn underneath it
    (e.g. an outline pass) rather than the true background. A closed
    shape has a proper non-overlapping fill (interior) and stroke
    (boundary), exactly like a circle, so the underlying AA structure
    shows through the translucent fill and the outline stays crisp.
    """
    (x1, y1), (x2, y2) = p1, p2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return (f'M {x1 - r:.2f},{y1:.2f} '
                f'A {r:.2f},{r:.2f} 0 1,0 {x1 + r:.2f},{y1:.2f} '
                f'A {r:.2f},{r:.2f} 0 1,0 {x1 - r:.2f},{y1:.2f} Z')

    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux  # unit normal, perpendicular to the line

    a1 = (x1 + nx * r, y1 + ny * r)
    a2 = (x2 + nx * r, y2 + ny * r)
    b2 = (x2 - nx * r, y2 - ny * r)
    b1 = (x1 - nx * r, y1 - ny * r)

    # sweep-flag=0: arcs bulge outward (away from the line), giving a
    # convex rounded cap. The flipped (=1) value curves inward instead,
    # producing a concave notch at each end.
    return (
        f'M {a1[0]:.2f},{a1[1]:.2f} '
        f'L {a2[0]:.2f},{a2[1]:.2f} '
        f'A {r:.2f},{r:.2f} 0 1,0 {b2[0]:.2f},{b2[1]:.2f} '
        f'L {b1[0]:.2f},{b1[1]:.2f} '
        f'A {r:.2f},{r:.2f} 0 1,0 {a1[0]:.2f},{a1[1]:.2f} Z'
    )
