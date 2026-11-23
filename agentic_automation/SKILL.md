---
name: martini-param-with-shaker
description: Parameterize Martini 3 coarse-grained bonded terms for a small molecule with the Shaker package - choose bonds/constraints/angles/dihedrals from a mapped atomistic (AA) trajectory, run CG simulations in GROMACS, and iteratively refine against the AA reference. Use when the user asks to parameterize, refine, or validate Martini 3 bonded parameters, or to run/continue a Shaker notebook routine.
---

# Martini 3 bonded parameterization with Shaker

The input is a Shaker notebook with a finished mapping (usually one notebook per molecule).
The output is a validated `CG_WAT/initial_CG.itp`, a documented notebook, and a short report.
The user owns the mapping and bead types. **Trust them as given and do not change them.**

Supporting files in this skill (read them when the step says so):
- `martini_rules.md`: the Martini 3 bonded rules to apply. **Read before choosing terms.**
- `shaker_reference.md`: Shaker API, return shapes, report formats, gotchas. **Read before the first
  Shaker call.**
- `references/README.md`: index of papers, if any. Open a PDF only for the pages the index points to.

Use Shaker's own functions and reports. Avoid writing helper modules; a few lines in a scratch cell
are fine for one-off checks.

## Operating rules

1. **Edit only the notebook (through the Jupyter MCP tools) and the CG `.itp` file** unless the user says
   otherwise. Check the local `CLAUDE.md` or the user's instructions for permissions and for how notebook
   paths are resolved (often relative to the Jupyter root `/`, not the working directory). Reading any
   file is fine.
2. **Keep cell output small.** Right after the imports, run
   `shaker.set_options(show=False, progress=False)`. Figures are then saved (SVG) but not displayed,
   and progress bars are off; Shaker's text reports still print. Read those reports instead of figures.
3. **Figures: SVG for the user, PNG only for your own look.** Keep the default SVG outputs. When the
   *shape* of a distribution matters, re-plot to a scratch PNG (`outfile="_agent_<name>.png"`), `Read`
   it, then delete it, so only SVGs remain. Whenever you point the user at a figure, say which iteration
   it's from.
4. **Long cells** (`map_aa2cg`, `run_SASA`, `runSim`) outlast the ~10 s run_cell timeout. Execution
   continues in the kernel and later cells queue behind it. For simulations:
   - GROMACS output goes to `CG_WAT/gmx_run.log`; the cell prints one line per stage when it's done.
   - Watch `CG_WAT/runSim.status` (see step 4) with a background watcher if one is available.
   - A 200 ns single-molecule run typically takes 5–10 min on a GPU. A failed or hung run raises an
     error with the last lines of GROMACS output; read it before retrying.
5. **Order matters.** The notebook reuses global names between steps. Re-run the cell that defines the
   comparison targets before re-measuring, or give step 3 and step 5 different variable names.
6. Keep scratch cells to a minimum, end them with `_ = ...` so nothing renders into the output, and
   delete them at the end. Put reusable analysis in proper cells.


## Workflow

### 0. Orient
Read the notebook (`read_notebook`), the mapping, and the AA paths. Run the import and setup cells,
then `shaker.set_options(show=False, progress=False)`. Note the bead order, which fixes the itp indices.

### 1. Map
Run `map_aa2cg`. Trust the mapping as given: unmapped atoms, repeated atoms and included hydrogens are
deliberate. Don't write checks for them. Run the 2D render cell for the human (it writes
`cg_overlay.svg`).

### 2. SASA of the mapped model
Run the SASA cell. `plot_sasa_dir` prints each entry vs AA with a mark: ✓ within 5 %, ⚠ within
10 %, ✗ beyond. The mapped CG SASA should be ✓; if not, the mapping or bead sizes are off. Tell the
user, since that is outside the bonded scope.

### 3. Survey and choose the bonds
Run `shaker.bead_neighbours(GRO, mapping, resname)`. Its neighbours are the candidate bonds; paths of
three and four neighbouring beads are the candidate angles and dihedrals for later iterations; bead
rings of three are triangles (bonds/constraints on all three sides).

Beads joined through the same unmapped atom all appear as mutual neighbours (and as 3-bead rings).
That is expected with a deliberate mapping; don't bond every such pair. Pick the pairs that reflect
the chemistry (usually each bead to the one it hangs from), and check the rest in the comparison.

Measure the candidate bonds on the mapped trajectory with `shaker.measure_bonded_terms`, and get
stiffness estimates from a trial `bonded_estimator` run (its comments show k values and constraint
conversions). For each distribution note its sd and whether it's unimodal.

Also measure the non-neighbour bead pairs (as `dist_tgts` in the same call). Pairs closer than
~5.5 Å feel full LJ, since `nrexcl = 1` excludes only bonded pairs.

**Terms are introduced in stages, one stage per iteration** (see `martini_rules.md`). An iteration is
one `runSim` and its analysis.

1. **Iteration 1: bonds only.**
   - Every neighbour pair picked above gets a bond, or a constraint when it's very stiff (the estimator
     converts k > 25 000 automatically). Keep to at most 4 constraints per bead.
   - Close non-neighbour pairs get no term yet; they are comparison targets only.
   - No angles, no dihedrals, no explicit `[ exclusions ]`.
   - Exception: a fused rigid unit (see `martini_rules.md`, "Rigid and fused systems") is built in full
     now: frame constraints, virtual sites, the generator's exclusions and, for a 4-bead hinge frame,
     its improper.
2. **After iteration 1: survey angles and dihedrals.** List the candidate angles (paths of 3) and
   dihedrals (paths of 4) from the `bead_neighbours` output, plus skip-bead angles: triplets whose
   consecutive beads are at most two neighbour steps apart (measured now, added no earlier than
   iteration 3). Measure them on both the mapped AA and the CG trajectory with
   `measure_bonded_terms`. For each note its sd, whether it's unimodal, near-linear (angle mean > 150°
   or reaching 170°), flat (dihedral sd > 60°) or planar (dihedral near 0/180° with sd < 20°).
   Anything the CG run already reproduces from bonds and non-bonded interactions needs no term.
3. **Iteration 2: the first angles.** Add the obvious angles among those the CG run misses: stiff
   (sd ≲ 15°), unimodal, along connected beads. A near-linear one is fine if no dihedral is planned
   through it. When two angles fix the same geometry, take the one further from 180°. A close
   non-neighbour pair that the bonds-only run distorts gets the angle spanning it. Still no dihedrals.
4. **Iteration 3 onward: more complex angles and dihedrals**, only where the CG run shows they're
   needed. Candidates are terms the bonded report marks ✗/⚠ or `mean-off` while they are reasonably
   stiff in AA. Add one logical group at a time. This is also where **terms off the bonded frame**
   come in (`martini_rules.md`): skip-bead angles chosen from the survey, a second angle aimed at a
   region an existing angle already targets, and, more rarely, skip-bead dihedrals to set the relative
   plane of two rigid parts. Measure such candidates with `measure_bonded_terms` before choosing.
5. **No dihedral without its angles** (proper or improper). Before adding i-j-k-l, i-j-k and j-k-l
   must each be present as an angle term, **or** be replaced by a bond/constraint between the angle's
   outer beads (i-k or j-l), which closes a triangle. No other substitute counts. A dihedral needing
   a near-linear angle gets type 10 on that angle (the estimator does this for angles > 140° inside a
   listed dihedral), or a different quadruplet.
6. Flat torsions never get a proper dihedral. Multimodal terms follow the order in `martini_rules.md`
   ("Multimodal distributions").

Put the targets and a short comment explaining each choice (and what was deliberately left out) in the
estimator cell, run `bonded_estimator`, and print the footer. Keep `show=False` on the estimator; its
per-dihedral fit figures are saved as `dihedral_fit_<quadruplet>.svg`.

The itp is written from the footer once, in iteration 1. From iteration 2 on, use the estimator only to
get values for new terms (list just those in its targets), and paste them into
`CG_WAT/initial_CG.itp` by hand with a comment, as in step 5.

### 4. Build and run
- Write the itp. `shaker.write_initial_CGitp()` writes an initial .itp. You can apply further changes
  manually by opening and editing the file.
- Build the water box once, then run `runSim()`. Every run is checkpointed to `CG_WAT/iter_N_<stamp>/`.
- **Before each run**, check the close non-neighbour pairs from step 3 against the itp: any pair
  closer than ~5.5 Å that is neither bonded nor spanned by an angle feels full LJ. In iteration 1
  that's expected; include those pairs in the comparison targets and check them first.
- Watch `CG_WAT/runSim.status`. Its first line is `runSim · running <stage>`, `runSim · done` (with the
  new `iter_*` folder) or `runSim · failed <stage>` (with the reason), plus a timestamp; the finished
  stage lines follow. Note the time before starting the `runSim` cell and only trust a status written
  after it: until the queued cell actually starts, the file still shows the previous run. A `running`
  line whose timestamp and `gmx_run.log` stop changing means the kernel died.

### 5. Compare and refine
- After each run, re-measure AA and CG with the **comparison** targets (parameterized terms plus
  emergent observables such as end-to-end distances and unparameterized dihedrals). From iteration 1
  on, include all candidate angles and dihedrals so the next stage can be chosen from their CG
  distributions.
- Run `plot_bonded_distributions(AA_bonded, CG_bonded, prev_bonded, labels=["AA", "CG", "Prev. CG"])`
  and read its report: per term mean ± sd, Δ, OC, W, a ✓/⚠/✗ mark, `mean-off`, hints, and for
  Prev. CG which terms got better or worse. `only_flagged=True` lists only the problem terms.
- Recompute SASA (`run_SASA("CG_Simulated", ...)`, `plot_sasa_dir()`) and the overlap matrix
  (`assess_overlap_matrix`, which prints the mean OC, the worst beads and the worst pairs).
- Record the iteration:
  `shaker.record_iteration("CG_WAT", note="<what changed>", overlap=oc, bonded=(AA_bonded, CG_bonded), sasa=sasa_values)`.
  It prints the history table (SASA %, mean overlap, bonded ✓/⚠/✗ counts, note) for every iteration.

Diagnose before changing anything. Typical fixes, in order of preference:
1. **Missing term.** A 1-3 pair pulled together by LJ gets the angle spanning it. Switch to a 1-3
   bond/constraint only if the angle doesn't hold it (still ✗ or `mean-off` in the next run) or the
   estimator capped that angle's k (`capped from k=…` comment). Explicit exclusions are a last resort:
   use them sparingly, only when neither can fix the problem, and justify each one in a comment.
   An unmatched angle or torsion that is stiff in AA gets its term at the matching stage from step 3
   (obvious angles in iteration 2, other angles and dihedrals from iteration 3, dihedrals only once
   their angles are in place).
2. **Systematic offset.** When the CG mean is off, shift the reference value by the offset and keep k.
3. **Width mismatch.** Scale k by (σ_CG/σ_AA)².
4. **Near-linear instability.** When a run shows it (crash, LINCS warnings) or a dihedral is added
   through a near-linear angle, move that angle to restricted bending (type 10) or choose another one.

Make manual changes **directly in `CG_WAT/initial_CG.itp`**, with a comment on each edited line giving
the iteration, the observation, and the fix. One logical change per iteration where possible.
The writer cell only writes when `initial_CG.itp` is missing. To regenerate from the notebook, delete
the file first; that discards all hand edits (earlier versions stay in `CG_WAT/iter_*`).

### 6. Stop and report
Stop when all of these hold, or after **10 iterations** (then report what's left):
- every stage line of the last `runSim` (in `CG_WAT/runSim.status`) ends in ✓: no LINCS warnings, no
  early stop, and no crash files (`ls CG_WAT/step*.pdb CG_WAT/crash*` finds nothing);
- CG SASA ✓ (within 5 % of AA). ⚠ (within 10 %) is acceptable only if ✓ can't be reached without
  hurting the bonded match; say so in the report;
- every **unimodal** term is ✓ or ⚠ (OC ≥ 0.65) and not `mean-off` (CG mean within one AA sd);
- the mean overlap in the `record_iteration` table improved by < 0.01 over the previous iteration.

An overlap plateau never overrides an open unimodal ✗/`mean-off` term. Before calling such a term
unfixable, measure up to 3 alternative coordinates for the same degree of freedom (all that exist,
skip-bead terms first) with `measure_bonded_terms`, and record in the report why each was rejected.

Terms the report hints as `multimodal` follow `martini_rules.md` ("Multimodal distributions"). Those
left with a mean/sd term are exempt from the overlap criterion; say so in the report.

Finish with:
- a markdown results cell in the notebook: the `record_iteration` history table (change, SASA %, mean
  overlap, bonded counts) and remaining caveats;
- scratch cells deleted and the final itp at `CG_WAT/initial_CG.itp`;
- a short report to the user: the final terms and why, the iteration history, what's left, and anything
  they need to check (leftover cells).
