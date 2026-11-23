# Martini 3 bonded-parameter rules (small molecules)

These are working rules for choosing and refining bonded terms. Sources are in brackets; see the list
at the bottom. When a rule conflicts with the AA data, the data wins, but write the deviation down.

## Topology basics
- Martini uses `nrexcl = 1`: **only directly bonded pairs are excluded**. 1-3 and 1-4 pairs still feel
  full LJ and Coulomb. A 1-3 pair sitting near its LJ minimum (~0.45–0.55 nm for S/T beads) is pulled
  together unless a bond, constraint, or `[ exclusions ]` entry covers it. This is the most common
  cause of CG geometry that comes out too compact.
- Shaker writes `nrexcl = 1`. Explicit `[ exclusions ]` are a last resort: none in the first
  iterations, and later only sparingly, when a bond/constraint can't fix the problem. Justify each one.
  Exception: the exclusions within a rigid unit (see "Rigid and fused systems") belong to its
  construction and go in from iteration 1.
- Bead order in `[ atoms ]` follows the mapping dict. All itp indices refer to it.
- It is often beneficial to introduce bonded terms gradually and systematically, instead of all at
  once. This is particularly important for angles/dihedrals, to avoid introducing redundant or
  unnecessary terms that are already well captured by the non-bonded interactions. Iteration 1 uses
  bonds/constraints only. After it, measure the angles and dihedrals in AA and CG. Iteration 2 adds the
  obvious angles; from iteration 3 on, add more complex angles and dihedrals, only where the CG run
  shows they are needed.

## Bonds and constraints
- Derive r0 and k from the mapped AA distribution (Boltzmann inversion, k = kT/σ²). Shaker does this
  from 0.2 Å histograms, so r0 can be off by ~0.03 Å for sharp distributions. Check the CG result
  rather than trusting the estimate.
- Use constraints for very stiff distances (k above ~25 000 kJ mol⁻¹ nm⁻²), typically aromatic ring
  triangles. Constraint triangles are fine with the default LINCS settings. Avoid very obtuse triangles.
- Don't over-constrain. Prefer harmonic bonds for non-ring connections unless the distribution is
  extremely sharp, and never build a constraint network that fixes the same geometry twice
  (it becomes rigid or singular).
- Maximum number of constraints coming out of a bead is 4. The 5th will lead to a crash.
- Avoid scaling bond lengths to improve SASA. Goal is to match bonded distribution.

## Angles
- Harmonic angles (GROMACS type 1, as written by Shaker) or cosine-harmonic are both fine for θ0 up
  to ~140°.
- **Near-linear angles** (reaching ~180°) are fine on their own: a stiff type-1 harmonic angle, up to
  the k cap, holds them. Prefer that to a 1-3 bond: near 180° a 1-3 distance barely changes with the
  angle. The problem is a dihedral running through such an angle, which becomes singular at 180°. For
  that, use **restricted bending, type 10**, on the angle, which diverges at 180° and so protects the
  dihedral [Bulacu 2013], or choose a different angle or quadruplet that stays further from 180°.
- Cap k_θ at ~250 kJ mol⁻¹ rad⁻² for stability at a 20 fs time step (Shaker's default cap).
- Angles inside a constraint triangle are redundant. Don't add them.
- **Non-obvious angles (iteration 3 onward).** Skip-bead angles (see "Terms off the bonded frame"), or
  a second angle aimed at a region an existing angle already targets, when the first alone doesn't pin
  the shape. Neither belongs in iterations 1–2.
- Use a bond/constraint for a close non-neighbour pair only when it closes a ring triangle or no angle
  can hold it. Otherwise list the pair as a comparison target and handle it with an angle in iteration 2.
  An angle doesn't remove the pair from LJ; it has to push against it with a capped k. "No angle can
  hold it" means: the angle was added and the pair is still ✗ or `mean-off` in the next run, or the
  estimator capped that angle's k (`capped from k=…` comment).

## Dihedrals
- **No dihedral without its angles.** A dihedral i-j-k-l may only be added once both i-j-k and j-k-l
  are defined. This applies to proper and improper dihedrals alike. The only exception: an angle may
  be replaced by a bond/constraint between its outer beads (i-k for i-j-k, j-l for j-k-l). Together
  with the two existing bonds, that 1-3 bond forms a triangle that fixes the angle. Nothing else counts.
- **Impropers** (type 2, harmonic) keep substituents in the plane of a ring and hold rings planar.
  For |ξ0| ≈ 180°, use Shaker's estimator, which handles the wrap-around.
- **Proper dihedrals** (type 9/1) only when the AA torsion has clear minima (circular sd below ~40–60°).
  Flat or nearly flat torsions need no term, and adding one only risks instability.
- Never put a proper dihedral across an angle that can approach 180°. Use type 10 on those angles, or
  combined bending-torsion (type 11), or pick another quadruplet. [Bulacu 2013]
- An out-of-plane preference often comes out of two angles alone. Check before adding a dihedral.
- For a rotation between two rigid parts, try quadruplets from both ends (para beads, far side of a
  ring). Pick the sharpest one whose two angles stay < 140°. Don't reject a degree of freedom because
  its obvious quadruplet crosses a near-linear angle; look for another.

## Terms off the bonded frame
Angles and dihedrals don't have to follow the chain of bonded neighbours; they can skip beads. These
are non-obvious terms: never in iterations 1–2, only from iteration 3 onward.
- **Angles: common, but use them strategically.** A skip-bead angle is worth it when the chain angles
  for the same geometry are broad, multimodal or near-linear while the skip-bead one is sharp and
  unimodal. Use it instead of those chain angles, not on top of them.
- **Dihedrals: less common, because of stability.** Use a skip-bead dihedral mainly to set the relative
  plane of two rigid parts, when no chain quadruplet does it cleanly. "No dihedral without its angles"
  applies to its own angles, and both should stay < 140°.
- **Redundancy:** a skip-bead term and a chain term on the same degree of freedom fight each other and
  hurt stability. Pick one.
- Candidates are measured like any other term with `measure_bonded_terms`, on the mapped AA and the CG
  trajectory, and judged by the same tests (sd, unimodal, near-linear, flat).

## Rigid and fused systems
- Fused rings or larger rigid groups (more than 3 beads in one rigid unit): use constraints on a
  3–4-bead frame plus **virtual sites** for the rest (`virtual_sitesn` or `virtual_sites3`, via
  `shaker.generate_virtual_sitesN/3`). Move the mass onto the frame beads (`mass_split="equal"`) and
  exclude within the rigid unit. A 4-bead "hinge" (two triangles sharing an edge, with an improper) is
  the standard for fused 6-6 rings. [Alessandri 2022]
- A fused unit is two or more 3-bead rings from `bead_neighbours` that share an **edge or a vertex**,
  where every pair distance within the unit is sharp in AA (sd ≲ 0.1 Å, roughly k ≳ 20 000). Rings
  that share beads but aren't rigid (any pair with sd > 0.15 Å) don't count. Build a fused unit in
  iteration 1 as a constrained frame plus virtual sites, starting with `align_mol_to_single_traj`:
  - a 4-bead hinge frame (two triangles sharing an edge) with the rest as `virtual_sitesn`
    (`generate_virtual_sitesN`). Hold the hinge flat with an improper and check the fold in the
    comparison; or
  - a constrained 3-bead triangle with all other beads as `virtual_sites3` (`generate_virtual_sites3`).
    The triangle is rigid by itself, so no hinge improper is needed.

  Never constrain all triangles plus cross pairs.
- **Large rigid cores** (steroid-like, e.g. cholesterol). Only for cores that are rigid in AA: every pair
  distance within the core is sharp (sd ≲ 0.1 Å, none > 0.15 Å). If any pair flexes (puckering,
  hinge motion), this doesn't apply; parameterize those beads with ordinary terms. In a rigid core,
  everything except the flexible parts (e.g. the alkyl tail) is one rigid unit. Build it as one constrained 3-bead triangle with every other core bead
  as `virtual_sites3`; the flexible parts stay real beads with ordinary bonds and angles. Pick the
  triangle so that:
  - most other core beads fall within its borders (projected onto its plane), so their positions are
    interpolated from the frame rather than extrapolated;
  - it is as close to equilateral as possible;
  - its sides are sharp in AA;
  - the worst-case `generate_virtual_sites3` reconstruction error is small.

  If it doesn't cost the criteria above, include the bead the flexible part attaches to, so it hangs
  off a frame bead.
- `generate_virtual_sitesN` places each site as a weighted average of the frame beads, so it only works
  for sites inside the frame. If it fails the reconstruction check for a site, use
  `generate_virtual_sites3` with a 3-bead frame, which can also place sites outside or out of the plane.

## Multimodal distributions
A single harmonic term can't reproduce a distribution with several peaks. For a multimodal angle or
distance, go through these in order (never before iteration 3):
1. **Leave it free** if the CG run already shows the peaks roughly right from bonds and non-bonded
   interactions.
2. **Look for a unimodal coordinate for the same degree of freedom.** A multimodal term is often the
   projection of a unimodal one. Measure the other triplets on the same end bead and an out-of-plane
   improper with `measure_bonded_terms`, and use the sharpest.
3. **Peaks close together:** one harmonic term matched on mean and sd (what the estimator does; it
   gives a broad, weak well).
4. **Peaks far apart, no unimodal alternative:** still match mean and sd, keep k weak, and report that
   its overlap stays low by construction. Don't chase its overlap.

Do not use tabulated potentials.

## Refinement heuristics
- **Mean offset**: shift the reference value by the observed offset, in the opposite direction. Keep k.
- **Width mismatch**: k_new = k_old · (σ_CG/σ_AA)².
- **Everything too compact**: look for close 1-3 pairs before tuning anything else. Cover them with an
  angle first, then a bond/constraint if the angle can't hold them. Exclusions only as a last resort.
- Change one thing at a time where possible. Always check emergent observables (end-to-end distances,
  unparameterized dihedrals, SASA), not just the terms you fitted.

## Validation targets
- SASA within ±5 % of AA (computed with Martini bead radii). [Alessandri 2022] Within 10 % is acceptable
  only if 5 % can't be reached without hurting the bonded match.
- A stable 200 ns run at dt = 20 fs with no LINCS warnings.
- Overlap coefficient ≥ 0.65 for unimodal terms (Shaker's ⚠ or better; ✓ is ≥ 0.80). Mean
  intra-molecular distance overlap as high as the model allows (≈0.75+ is good for a flexible
  molecule).

## Sources
- Souza P.C.T. et al., "Martini 3: a general purpose force field for coarse-grained molecular
  dynamics", *Nat. Methods* 18, 382–388 (2021). doi:10.1038/s41592-021-01098-3
- Alessandri R. et al., "Martini 3 coarse-grained force field: small molecules",
  *Adv. Theory Simul.* 5, 2100391 (2022). doi:10.1002/adts.202100391
- Bulacu M. et al., "Improved angle potentials for coarse-grained molecular dynamics simulations",
  *J. Chem. Theory Comput.* 9, 3282–3292 (2013). doi:10.1021/ct400219n
