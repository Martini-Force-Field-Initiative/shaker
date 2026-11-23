# SHAKER quick reference (for agents)

Checked against branch `FriendlyAGI` @ `4e3c229`. `import shaker`. If a signature here looks wrong,
read the docstring (`help(shaker.fn)`); they are kept current.

## Start every notebook with
```python
shaker.set_options(
    show=False, progress=False
)  # no figures or progress bars in cell output
```
- `show`: figures are still **saved** (SVG by default); only notebook display is off.
- `progress`: all progress bars (mapping, measuring, overlap, alignment).
- `verbose` (default True): SHAKER's own text reports. Keep it on: they are the cheap way to read
  results. `set_options(verbose=False)` leaves only warnings and errors.
- Any function argument passed explicitly (`show=`, `verbose=`) overrides these.

## Data conventions
- Mapping: `{RESNAME: {BEAD: {"type": "SN3a", "charge": 0, "atoms": [...]}}}`. Bead order = itp
  index order. Repeating an atom name in `atoms` weights the COG (intended).
- Bead size from the first letter of the type: R (default), S, T, U = virtual (0 radius, 0 heavy atoms).
- Units: measurement dicts and reports use **Å** for distances; itp text uses **nm**. Angles and
  dihedrals in degrees; dihedrals in [-180, 180].
- Default histogram bins: distances 0–50 Å (0.2 Å), angles 0–180° (2°), dihedrals ±180° (2°).
  Values outside the bins trigger a warning naming the term; pass `bins_dist=` etc. to widen.
- Quality marks (bonded, overlap): ✓ OC ≥ 0.80, ⚠ ≥ 0.65, ✗ below. SASA: ✓ within 5 % of AA,
  ⚠ within 10 %, ✗ beyond. Marks are judged on the 2-decimal value shown.

## Functions (notebook order)
| call | does | returns / writes | notes |
|---|---|---|---|
| `map_aa2cg(gro, xtc, mapping, outname="cg_mapped")` | COG-maps the AA trajectory | writes `cg_mapped.gro/.xtc`, `cg_mapped_mapping.log`; returns None | slow (~1 min per 25k frames). Report: heavy atoms vs beads per residue |
| `bead_neighbours(gro, mapping, resname)` | which beads are bonded, from AA bonds (directly, through unmapped atoms, or sharing an atom) | `{bead: [neighbours]}` in mapping order; **prints** each bead's neighbours, the bonded-pair count and bead rings | needs only the AA structure, fast. Neighbours = candidate bonds/constraints; paths of 3 / 4 = candidate angles / dihedrals; 3-bead rings = triangles. An unmapped heavy atom joined to several beads makes them all neighbours (extra triangles); that's expected, not a mapping error |
| `render_2dMapping(pdb, resname, mapping, rotate=, mirror=)` | 2D mapping drawing | returns the **Path** of `cg_overlay.svg` (always written) | for the human |
| `render_mapping`, `render_connely_surface`, `render_ensemble` | nglview widgets | widget | for the human only; skip |
| `run_SASA(name, gro, xtc, resname, isCG=, mapping=)` | `gmx sasa` into `./SASA/<name>/` | files; gmx output in `gmx_sasa.log` | slow on long AA trajectories. Fails with the log tail in the error |
| `plot_sasa_dir()` | reads `./SASA/*/SASA.xvg` | `(fig, ax, [(name, mean_nm2, sd), ...])` sorted by folder name; saves `SASABar.svg`; **prints a report** | look values up by name, not position |
| `bonded_estimator(u, resname, dist_tgts, ang_tgts, harm_dihed_tgts, imp_dihed_tgts, use_indices=True)` | Boltzmann-inversion estimates | **str** (itp sections) | rules below. Saves one `dihedral_fit_<B1-B2-B3-B4>.svg` per proper dihedral; `outfile=None` to skip |
| `write_initial_CGitp(resname, mapping, filename="initial_CG.itp", header=, footer=)` | writes the itp | file (overwrites) | the stock notebook guards it with `if not exists` |
| `prepare_setup_water(cg_gro, n_mols=1, box_size=4)` | insert, solvate, neutralise, NaCl | `topol.top`, `memion_2.gro` in cwd; gmx output in `gmx_setup_water.log` | run once; `topol.top` `#include`s the itp by absolute path |
| `runSim(prodOPT="-pin on -nt 8", keep_last=-1)` | min → rel (10 ns) → prod (200 ns) in cwd | `m/r/p.*`, `pbc.pdb/.xtc`, `runSim.status`, new `iter_N_<stamp>/`; returns None | see "Running simulations" |
| `list_iterations(run_dir)` | checkpoint folders | `[Path, ...]` oldest → newest | each holds itp, `pbc.pdb/.xtc`, `mapping.json` (if passed), `metrics.json` (if recorded) |
| `measure_bonded_terms(u, resname, dist_tgts, ang_tgts, dihed_tgts, stride=1)` | histograms | `{'distances'/'angles'/'dihedrals': {'bins','hist','targets'}}` | `hist[i]` is a density over bin centres for `targets[i]`. **No PBC handling**: use PBC-fixed trajectories |
| `plot_bonded_distributions(AA, CG, [prev], labels=, colors=)` | comparison figure | `fig`; saves `cleanbonds.svg`; **prints a report** | datasets must share targets and bins. `only_flagged=True` lists only ⚠/✗/mean-off terms |
| `assess_overlap_matrix(u_aa, u_cg, resname)` | all bead-pair distance overlap | `(oc, fig, axes)`, `oc = {'matrix','overlaps','pairs','bead_names'}`; saves `overlap_matrix.svg`; **prints a summary** | PBC-correct. `print_matrix=True` adds the full N×N matrix (grows with N²) |
| `record_iteration(run_dir, note=, overlap=oc, bonded=(AA, CG), sasa=sasa_values)` | stores this iteration's scores | `metrics.json` in the latest `iter_*`; **prints the history table** | nothing is re-measured; call after the analysis of each run. Calling it again overwrites the latest folder's `metrics.json` (new timestamp, and the note you pass) |
| `align_mol_to_single_traj(gro, xtc, selection=, align_selection=)` | aligns all copies onto one | writes `aligned_molecules.xtc`, `reference_molecule.gro`, `average_molecule.gro`; returns None | first step for virtual sites |
| `generate_virtual_sites3(u, frame_names, mass_split=, mapping=, resname=)` | 3-bead-frame vsites (+ constraints, exclusions) | `(lines, mapping_or_None)` | `u = md.Universe("average_molecule.gro")` |
| `generate_virtual_sitesN(u, frame_names, ...)` | vsiten weights | `(lines, mapping_or_None)` | 2–4 frame beads. With `mass_split="equal"`, pass a `mapping` holding **only the rigid unit's beads**: every bead in it that isn't a frame bead gets mass 0. Put the returned masses in a copy of the mapping for the itp writer |

Every figure function takes `outfile=` (no extension → SVG; `"x.png"` → PNG; `None` → don't save;
existing files are overwritten) and `transparent=True`. To look at a figure, save a scratch PNG copy
and delete it afterwards (see SKILL.md); the SVGs are the user's.

## Running simulations
- Run from the run folder (tutorial: `CG_WAT`), which must hold `memion_2.gro` and `topol.top`.
- `runSim` blocks until done. It **first deletes** the previous run's `m.*`, `r.*`, `p.*`, `pbc.*`,
  `step*`, `crash*` in that folder (`cleanOldRun=True`). Earlier results survive only in `iter_*`.
- GROMACS output goes to `gmx_run.log`; the cell shows one line per stage:
  ```
  runSim · output → gmx_run.log
    min    3491 steps   Fmax 24.8 (target 10)   ✓
    rel    ETA 20:05 (~1 min) · 25 · 50 · 75        10.0/10.0 ns   16304 ns/day   0:00:53   ✓
    prod   ETA 20:14 (~9 min) · 25 · 50 · 75        200.0/200.0 ns   31733 ns/day   0:09:05   ✓
  ```
  `⚠` flags: `did not finish`, `stopped early` (fewer steps than planned, e.g. `-maxh`),
  `N LINCS warnings`. Minimization stopping above Fmax 10 at machine precision is normal.
- Failures raise with the last lines of GROMACS output in the message; the stage line ends `✗ failed`.
- **`runSim.status`** in the run folder tracks the run without the notebook. First line:
  `runSim · running <stage>`, then `runSim · done   <time>   → iter_N_<stamp>` or
  `runSim · failed <stage>   <time>   <reason>`; below it, the finished stage lines as above.
  Note the time before starting the `runSim` cell and only trust a status written after it: until a
  queued cell actually starts, the file still shows the previous run. A `running` line whose
  timestamp and `gmx_run.log` stop changing means the kernel died.
- A hung mdrun is killed after `hang_timeout=600` s without output, or 30 s after a `Fatal error`.
- Short test run: `runSim(prodOPT="-pin on -nt 8 -nsteps 5000")`. Wall-time cap: add `-maxh 2`.
- `stream_gmx=True` shows raw GROMACS output instead (floods the cell; avoid).

## Reading the reports
Bonded (`plot_bonded_distributions`), second dataset in full, further ones as changes:
```
Bonded: CG vs AA
                      AA mean ± sd  CG mean ± sd      Δ
distances (Å)
  WR1-LWR             13.90 ± 1.00  12.70 ± 1.80  -1.20  OC 0.60  W 1.25  ✗ mean-off
angles (°)
  WR5-WR6-WR7          161.5 ± 6.6  154.4 ± 10.1   -7.1  OC 0.64   W 7.2  ✗ mean-off
5 terms: 2 ✓  1 ⚠  2 ✗, 2 mean-off

Prev. CG vs AA: 5 terms: 3 ✓  2 ⚠  0 ✗, 0 mean-off
  CG worse:  WR1-LWR ✓→✗ (OC 0.84→0.60), ...
  CG better: WR1-WR2 ⚠→✓ (OC 0.76→0.95)
```
- `mean-off`: |Δ mean| > the reference's sd (a shifted peak even if OC looks fine).
- Dihedral mean, sd, Δ and W are circular (-179° vs 179° are 2° apart).
- Plot annotations show the same OC/W numbers.

Overlap (`assess_overlap_matrix`): mean OC, pair counts, per-bead mean OC (worst first), up to 5
worst non-✓ pairs. Beads with low mean OC point at where the 3D shape is off.

SASA (`plot_sasa_dir`): mean ± std per folder, and vs `AA`: Δ nm², %, mark.

## Iterating
1. Edit the itp in the run folder (numbers in nm / degrees / kJ mol⁻¹).
2. `runSim()` → new `iter_N`.
3. Measure CG, `plot_bonded_distributions(AA, CG, prev)` (prev from `list_iterations(...)[-2]`),
   `run_SASA("CG_Simulated", ...)` + `plot_sasa_dir()`, `assess_overlap_matrix(...)`.
4. `record_iteration(run_dir, note="what you changed", overlap=oc, bonded=(AA, CG), sasa=sasa_values)`:
   ```
   iter  SASA %  overlap  bonded ✓  ⚠  ✗  note
      3  98.5 ✓     0.82        28  8  3  re-anchor on WR5-WR7 axis
      4  98.6 ✓     0.83        30  6  2  WR5-WR7 r0 0.472 -> 0.480
   ```
   Iterations never recorded show "–". SASA defaults to the `CG_Simulated` entry vs `AA`.

## Estimator rules (`bonded_estimator`)
- Bond k > 25000 → written as a constraint.
- Angle k capped at 250 (original k in a comment).
- Angle θ0 > 140° that is part of a listed dihedral → type 10 (restricted bending), k ×1.1;
  if θ0 > 150° also k ×1.2 and θ0 capped at 150° with a warning comment (pick another dihedral).
- Impropers: harmonic (type 2), k capped at 250. Proper dihedrals: fitted, type 9 (several terms).
- `use_indices=True` numbers beads by their order in `cg_mapped.gro` (= mapping order); names kept
  as comments. Default is names.

## Gotchas
- MDAnalysis "Failed to guess the mass for atom types …" warnings are harmless.
- `prod.mdp`: RF electrostatics, eps_r 15, v-rescale / Parrinello-Rahman, `lincs_order 4`, no
  `verlet-buffer-tolerance`. Conserved-energy drift in the log is normal for this setup.
- Tutorial functions write into the current directory (`runSim` needs `os.chdir(run_dir)`).
