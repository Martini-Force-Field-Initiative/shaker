"""2D chemical structure drawing with coarse-grained mapping overlay."""

from collections import Counter
import math
from pathlib import Path
import warnings
from xml.sax.saxutils import escape as _xml_escape

from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit.Chem.Draw import rdMolDraw2D

from .helper import _category_from_type


def render_2dMapping(pdb_file, resname, mapping,
                     out_svg="cg_overlay.svg", size=(950, 480), mode="connected",
                     bead_r=15.0, conn_r=15.0, alpha=0.2, line_w=3.0,
                     font_size=24, label_dy=20.0, net_charge=None,
                     show_bead_type=None):
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
        Net formal charge of everything in ``pdb_file``, passed to RDKit's
        bond-order determination (required for charged molecules — without
        it, bond-order assignment is attempted assuming a neutral molecule
        and fails or produces incorrect bonds for anything else). If not
        given (default), it is inferred as the sum of the per-bead
        ``"charge"`` entries in ``mapping[resname]``; this requires every
        bead in the mapping to define a ``"charge"``, and the sum must be
        a whole number. That inferred value is correct when the file holds
        only the mapped molecule, or when everything else in it is
        neutral; pass it explicitly when the surrounding structure is
        itself charged.
    show_bead_type : bool or None, optional
        Whether to show each bead's ``"type"`` in italics on a second line
        under its name label. The default (``None``) shows it for every
        bead that defines one, and beads without a ``"type"`` simply get
        no second line. Pass ``False`` to suppress the types even when the
        mapping provides them.

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
        net_charge = _infer_net_charge(bead_map, resname)

    molH, mol = _load_mols(pdb_file, resname, net_charge)
    idxH = _atom_name_map(molH, resname)
    idx  = _atom_name_map(mol,  resname, warn_duplicates=False)

    hmap   = _h_to_heavy_map(molH, resname)
    colors = _bead_colors(bead_map)

    idxH_set = set(idxH)
    idx_set  = set(idx)

    missing = sorted({a for bead in bead_assignments for a in bead} - idxH_set)
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

        dropped = {hmap.get(a, a) for a in bead} & idxH_set - idx_set
        if dropped:
            warnings.warn(
                f"Bead '{label}': atoms {sorted(dropped)} exist in the H-mol "
                f"but are absent from the heavy-atom mol; centroid may be shifted.",
                UserWarning, stacklevel=2)

        if not weights:
            warnings.warn(
                f"Bead '{label}': none of its atoms resolved to a heavy atom, "
                f"so it will not be drawn.",
                UserWarning, stacklevel=2)
            continue

        sx = sy = sw = 0.0
        for name, wt in weights.items():
            x, y = draw_coords[name]
            sx += wt * x
            sy += wt * y
            sw += wt
        cx, cy = sx / sw, sy / sw

        heavy_names = _bead_heavy_names(bead, hmap)
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
                    _circle(cx, cy, bead_r, fill, outline, line_w))

        if mode == "atomblobs":
            for x, y in atom_pts.values():
                shading_layer.append(
                    _circle(x, y, conn_r, fill, outline, line_w))

        if mode in ("circle", "both"):
            shading_layer.append(
                _circle(cx, cy, bead_r, fill, outline, line_w))

        bead_type = bead_map[label].get("type") if show_bead_type is not False else None
        type_font_size = font_size * 0.7

        label_offset = max(bead_r, conn_r) + 2
        label_width  = _text_width(label, font_size)
        type_width   = _text_width(bead_type, type_font_size) if bead_type else 0.0
        width        = max(label_width, type_width)
        # Grow the label to the right by default, flipping to the left
        # only when that would overrun the right edge *and* the flipped
        # label actually fits — otherwise flipping just moves the overflow
        # to the other side.
        overflows_right = cx + label_offset + width > w
        fits_flipped    = cx - label_offset - width >= 0
        if overflows_right and fits_flipped:
            tx     = cx - label_offset
            anchor = "end"
        else:
            tx     = cx + label_offset
            anchor = "start"
        ty = cy + label_dy

        height = font_size * 1.2
        if bead_type:
            height += type_font_size * 1.2

        label_specs.append({"label": label, "type": bead_type, "tx": tx, "ty": ty,
                            "anchor": anchor, "edge": label_color, "outline": outline,
                            "width": width, "height": height,
                            "type_font_size": type_font_size})

    _avoid_label_collisions(label_specs, font_size)

    for spec in label_specs:
        label_layer.extend(_halo_text(
            spec["tx"], spec["ty"], spec["label"], font_size, spec["anchor"],
            spec["edge"], spec["outline"], stroke_width=0.8, bold=True))

        if spec["type"]:
            type_font_size = spec["type_font_size"]
            type_ty = spec["ty"] + font_size * 0.5 + type_font_size * 0.5 + 2
            label_layer.extend(_halo_text(
                spec["tx"], type_ty, spec["type"], type_font_size, spec["anchor"],
                spec["edge"], spec["outline"], stroke_width=0.6, italic=True))

    overlay = ['<g id="cg_overlay">'] + shading_layer + label_layer + ['</g>']

    parts = svg.rsplit("</svg>", 1)
    svg   = parts[0] + "\n".join(overlay) + "\n</svg>" + parts[1]

    Path(out_svg).write_text(svg, encoding="utf-8")
    return svg


def _load_mols(pdb_file, resname, net_charge):
    """
    Return (molH, mol): sanitized molecule with Hs, and no-H molecule with
    2D coords, containing only the atoms belonging to ``resname``.

    Bonds are determined once on the full structure, then the target
    residue is sliced out of it, carrying the resulting bond orders and
    formal charges with it. Determining bonds on the isolated residue
    instead would discard the context its neighbors provide, which can
    fail or produce wrong bond orders for groups that are ambiguous on
    their own — see ``_isolate_residue``.

    If the target residue is covalently bonded to another residue (e.g.
    an amino acid mid-chain), each such bond is capped with a placeholder
    atom rather than dropped or followed — see ``_isolate_residue`` and
    ``_label_caps_as_r``.
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

    # Determine bonds on the whole structure, before isolating anything:
    # a capped fragment lacks the chemical context that constrains the
    # search, which can fail outright or silently produce wrong bond
    # orders for groups that are ambiguous in isolation (see
    # `_isolate_residue`).
    try:
        rdDetermineBonds.DetermineBonds(molH_full, charge=net_charge)
    except ValueError as exc:
        raise ValueError(
            f"Could not determine bonds for {pdb_file} at net charge "
            f"{net_charge}. That charge applies to everything in the file, "
            f"not only residue '{resname}'. If the file contains more than "
            f"the mapped molecule (other chains, charged termini, ions, "
            f"solvent), pass net_charge explicitly as the total formal "
            f"charge of the whole file.") from exc

    molH, cap_indices = _isolate_residue(molH_full, target_idx)
    Chem.SanitizeMol(molH)

    molH = _label_caps_as_r(molH, cap_indices)

    mol = Chem.RemoveHs(molH)
    AllChem.Compute2DCoords(mol)
    return molH, mol


def _isolate_residue(molH_full, target_idx):
    """
    Build a molecule containing ``target_idx`` plus one capping atom for
    every bond that crosses into a different residue.

    Bond orders and formal charges are *copied* from ``molH_full``, which
    must already have had its bonds determined. Perception is
    deliberately not repeated on the isolated fragment: cutting a residue
    out removes the surrounding context that constrains the bond-order
    search, which for groups that are ambiguous in isolation (nitro,
    carboxylate and similar resonance cases) can fail outright or settle
    on a valid-looking but chemically wrong assignment.

    Each cap is a single-bonded hydrogen at the real neighbor atom's
    position, giving a valence-complete fragment.
    ``_label_caps_as_r`` converts these caps into display-only "R" atoms.

    Returns (mol, cap_indices): the isolated molecule and the indices of
    its capping atoms.
    """
    conf_full = molH_full.GetConformer()
    target_set = set(target_idx)
    old2new = {old_i: new_i for new_i, old_i in enumerate(target_idx)}

    rw = Chem.RWMol()
    positions = []

    for old_i in target_idx:
        src = molH_full.GetAtomWithIdx(old_i)
        new_atom = Chem.Atom(src.GetAtomicNum())
        new_atom.SetFormalCharge(src.GetFormalCharge())
        new_atom.SetNumExplicitHs(src.GetNumExplicitHs())
        new_atom.SetNoImplicit(True)
        info = src.GetPDBResidueInfo()
        if info is not None:
            new_atom.SetMonomerInfo(info)
        rw.AddAtom(new_atom)
        positions.append(conf_full.GetAtomPosition(old_i))

    for bond in molH_full.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in target_set and j in target_set:
            rw.AddBond(old2new[i], old2new[j], bond.GetBondType())

    cap_indices = []
    for old_i in target_idx:
        for nb in molH_full.GetAtomWithIdx(old_i).GetNeighbors():
            if nb.GetIdx() not in target_set:
                cap_i = rw.AddAtom(Chem.Atom(1))
                rw.AddBond(old2new[old_i], cap_i, Chem.BondType.SINGLE)
                cap_indices.append(cap_i)
                positions.append(conf_full.GetAtomPosition(nb.GetIdx()))

    conf = Chem.Conformer(len(positions))
    for i, pos in enumerate(positions):
        conf.SetAtomPosition(i, pos)
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


def _infer_net_charge(bead_map, resname):
    """
    Infer a molecule's net formal charge as the sum of its beads' charges.

    Requires every bead in ``bead_map`` to define a "charge", and the
    sum to be a whole number; otherwise raises a ValueError pointing to
    the explicit ``net_charge`` override as the alternative.
    """
    _require_bead_key(bead_map, "charge", resname,
                      hint=" Either add 'charge' to every bead in the "
                           "mapping, or pass net_charge explicitly.")
    charge_sum = sum(b["charge"] for b in bead_map.values())
    if charge_sum != int(charge_sum):
        raise ValueError(
            f"Bead charges for '{resname}' sum to a non-integer net "
            f"charge ({charge_sum}); pass net_charge explicitly.")
    return int(charge_sum)


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


def _text_width(s, font_size):
    """
    Rough text width estimate (characters x font size x a fixed average
    glyph-width factor), used both for canvas-edge label flipping and
    for the collision-avoidance bounding boxes below.
    """
    return len(s) * font_size * 0.6


def _circle(cx, cy, r, fill, outline, line_w):
    """One filled, outlined SVG circle — a whole bead, or a single atom blob."""
    return (f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" '
            f'fill="{fill}" stroke="{outline}" stroke-width="{line_w:.2f}" />')


def _halo_text(x, y, text, font_size, anchor, fill_color, outline_color,
               stroke_width, bold=False, italic=False):
    """
    Build the two stacked SVG <text> elements used for every bead label
    and type subtitle: a white halo pass (so the text stays legible over
    the structure regardless of what's behind it) followed by the
    colored, outlined fill pass.

    Bead names and types come from the user's mapping, so the text is
    XML-escaped — an unescaped "&" or "<" would otherwise produce an SVG
    that no parser can read, without anything here failing.
    """
    style = ""
    if bold:
        style += ' font-weight="bold"'
    if italic:
        style += ' font-style="italic"'
    text = _xml_escape(str(text))
    common = (f'x="{x:.2f}" y="{y:.2f}" font-family="sans-serif" '
              f'font-size="{font_size:.2f}"{style} dominant-baseline="middle" '
              f'text-anchor="{anchor}"')
    return [
        f'<text {common} stroke="white" stroke-width="2" fill="white">{text}</text>',
        f'<text {common} stroke="{outline_color}" stroke-width="{stroke_width}" '
        f'fill="{fill_color}">{text}</text>',
    ]


def _avoid_label_collisions(label_specs, font_size, min_gap=2.0):
    """
    Nudge label y-positions to reduce overlap between bead labels.

    Processes labels in their original (bead) order and places each one at
    the closest non-overlapping y-position to its preferred baseline,
    trying offsets in the order 0, +step, -step, +2*step, -2*step, ... .
    This keeps labels from drifting too far from their beads when labels
    are taller (e.g. when ``show_bead_type=True`` adds a second line).

    Bounding boxes reuse each spec's ``"width"`` and ``"height"`` (the same
    estimates already used for the initial canvas-edge placement;
    ``"height"`` covers both lines when a bead type subtitle is shown), so
    this is a cheap heuristic, not an exact layout solver.

    ``label_specs`` is modified in place: each dict's ``"ty"`` may be
    increased.
    """
    placed = []  # (x_min, x_max, y_min, y_max) of already-placed labels

    def _box_for(spec, ty):
        width = spec["width"]
        height = spec.get("height", font_size * 1.2)
        if spec["anchor"] == "start":
            x_min, x_max = spec["tx"], spec["tx"] + width
        else:
            x_min, x_max = spec["tx"] - width, spec["tx"]
        y_min = ty - font_size * 0.6
        y_max = y_min + height
        return x_min, x_max, y_min, y_max

    def _overlaps_any(box):
        x_min, x_max, y_min, y_max = box
        for px_min, px_max, py_min, py_max in placed:
            if x_min < px_max and x_max > px_min and y_min < py_max and y_max > py_min:
                return True
        return False

    crowded = []
    for spec in label_specs:
        pref_ty = spec["ty"]
        height = spec.get("height", font_size * 1.2)
        step = max(height + min_gap, font_size * 0.8)

        # Try nearest offsets first to keep each label close to its bead.
        chosen_ty = pref_ty
        max_tries = 80
        for i in range(max_tries):
            if i == 0:
                cand = pref_ty
            else:
                k = (i + 1) // 2
                sign = 1 if i % 2 == 1 else -1
                cand = pref_ty + sign * k * step

            box = _box_for(spec, cand)
            if not _overlaps_any(box):
                chosen_ty = cand
                break
        else:
            crowded.append(spec["label"])

        spec["ty"] = chosen_ty
        placed.append(_box_for(spec, chosen_ty))

    if crowded:
        warnings.warn(
            f"Could not place {len(crowded)} bead label(s) without overlap "
            f"({', '.join(crowded)}); they are drawn at their preferred "
            f"position. A larger canvas or smaller font_size may help.",
            UserWarning, stacklevel=3)


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


def _bead_heavy_names(bead, hmap):
    """
    Unique heavy atoms contributing to a bead, with Hs folded onto their
    bonded heavy atoms, in first-seen order.

    Names that don't correspond to an atom in the drawn molecule are kept
    here and filtered by the caller, which has the draw coordinates.
    """
    out  = []
    seen = set()
    for name in bead:
        heavy = hmap.get(name, name)
        if heavy in seen:
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
