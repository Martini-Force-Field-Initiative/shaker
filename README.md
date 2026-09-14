# SHAKER

**SHAKER** is a Python toolkit for parameterizing and validating Martini coarse-grained (CG) force fields against atomistic or quantum-mechanical reference data.

It provides modular utilities for mapping atomistic trajectories to CG representations, measuring and fitting bonded distributions, preparing GROMACS simulation systems, and visualizing CG models — all designed for interactive use in Jupyter notebooks.

---

## Features

- **CG Mapping** — Map AA/QM trajectories to CG beads with flexible bead definitions; generates mapping quality reports with Martini 3 tolerance checks
- **Bonded Analysis** — Measure and compare distance, angle, and dihedral distributions between reference and CG simulations
- **Parameter Estimation** — Estimate initial GROMACS bonded parameters (bonds, angles, proper/improper dihedrals) via equipartition theorem and inverted Boltzmann
- **Dihedral Fitting** — Advanced dihedral fitting with Savitzky-Golay smoothing and multi-term cosine series
- **SASA** — Compute solvent-accessible surface areas with CG-specific van der Waals radii
- **Virtual Sites** — Generate and optimize type-3 virtual site definitions
- **Model Quality Assessment** — Evaluate CG model accuracy via pairwise intra-bead distance overlap matrices and ensemble visualization
- **System Setup** — Build solvated Martini simulation boxes and run GROMACS minimization/production MD with minimal boilerplate
- **Visualization** — Render 3D mapping overlays, Connolly surfaces, molecular ensembles, and 2D chemical structures

---

## Installation

### Step1:

Clone the repository
```bash
git clone https://github.com/Lp0lp/shaker.git

```
Create a virtual environment with the dependencies (e.g. with **conda**):
```bash
cd shaker
conda env create -f environment-user.yml
```
This will create a conda environment with the name shaker-env.

***OR*** with **venv**:
```bash
cd shaker
python3 -m venv shaker-venv
```
This will create a venv environment with the name shaker-env.

Then activate your respective environment.

### Step2:

Install SHAKER with pip:
```bash
pip install .
```

### Step3:

Test the package with pytest
```bash
pytest -v tests/
```

All the tests should pass, if not please open an issue.

## Requirements

- Python >= 3.10
- [MDAnalysis](https://www.mdanalysis.org/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [RDKit](https://www.rdkit.org/)
- [nglview](https://nglviewer.org/nglview/latest/) (for 3D visualization in notebooks)
- [tqdm](https://tqdm.github.io/)
- [GROMACS](https://www.gromacs.org/) (external; required for system setup and SASA)

---

## Developer Workflow

Install SHAKER in your environment along with all development dependency groups (lint, typecheck, test, docs):
```bash
python -m pip install --group all
```

Then install the linting/formatting hooks so your changes are checked automatically:
```bash
pre-commit install
```
This runs `ruff` (lint + format) on staged files at commit time; the same checks also run in CI on every PR.

---

# Typical Workflow

SHAKER is designed for iterative CG parameterization. A typical session in a Jupyter notebook follows these steps:

## 1. Map an atomistic trajectory to CG

```python
import shaker

mapping = {
    "MOL": {
        "B1": {"type": "SC3", "charge": 0, "atoms": ["C1", "C2", "C3"]},
        "B2": {"type": "SC3", "charge": 0, "atoms": ["C4", "C5", "C6"]},
        "B3": {"type": "TC4", "charge": 0, "atoms": ["C7", "C8"]},
    }
}

shaker.map_aa2cg("reference.gro", "reference.xtc", mapping, outname="cg_mapped")
```

This writes `cg_mapped.gro` and `cg_mapped.xtc`, and prints a mapping quality report
comparing your bead count against the Martini 3 ±1/10 heavy-atom tolerance.

## 2. Estimate bonded parameters

```python
import MDAnalysis as mda

u = mda.Universe("cg_mapped.gro", "cg_mapped.xtc")

topology_text = shaker.bonded_estimator(
    u,
    resname="MOL",
    dist_tgts=[("B1", "B2"), ("B2", "B3")],
    ang_tgts=[("B1", "B2", "B3")],
)

print(topology_text)
```

`bonded_estimator` measures distributions, applies the equipartition theorem for bonds and angles, fits dihedrals via inverted Boltzmann, and returns
GROMACS-formatted topology lines ready to paste into an `.itp` file.

## 3. Write an initial topology file

```python
shaker.write_initial_CGitp("cg_mapped.gro", mapping, outname="initial_CG.itp")
```

## 4. Set up and run a simulation

```python
shaker.prepare_setup_water(
    "cg_mapped.gro",
    structure_itp="initial_CG.itp",
    n_mols=1,
    box_size=5,
    NaCL_Conc=0.15,
)

shaker.runSim()
```

## 5. Compare distributions

```python
u_aa = mda.Universe("cg_mapped.gro", "cg_mapped.xtc")  # reference
u_cg = mda.Universe("sim.gro", "sim.xtc")  # CG simulation

aa_bonded = shaker.measure_bonded_terms(
    u_aa,
    resname="MOL",
    dist_tgts=[("B1", "B2"), ("B2", "B3")],
    ang_tgts=[("B1", "B2", "B3")],
)

cg_bonded = shaker.measure_bonded_terms(
    u_cg,
    resname="MOL",
    dist_tgts=[("B1", "B2"), ("B2", "B3")],
    ang_tgts=[("B1", "B2", "B3")],
)

fig = shaker.plot_bonded_distributions(
    aa_bonded,
    cg_bonded,
    labels=["AA reference", "CG"],
    colors=["tab:blue", "tab:red"],
)
```

## 6. Assess model quality

Visualize the CG ensemble and quantify how well the CG model reproduces the
reference structural distributions with the overlap matrix.

```python
# Overlay of aligned CG frames — good for spotting sampling issues
view = shaker.render_ensemble("sim.gro", "sim.xtc", n_frames=50)
view  # displays an interactive nglview widget in the notebook

# Pairwise overlap matrix between AA and CG intra-bead distance distributions
fig = shaker.assess_overlap_matrix(u_aa, u_cg, resname="MOL")
```

The overlap matrix produces an N×N heatmap of overlap coefficients (0–1) for
every bead pair, plus a per-bead mean column on the right. Values close to 1
indicate the CG model faithfully reproduces the reference geometry; low values
flag which beads need further refinement.

---

## Tutorials

Step-by-step Jupyter notebooks are available in [`tutorials/`](tutorials/):

| Notebook | Description |
|---|---|
| [1 — Basic Parameterization](tutorials/1_BasicParameterization/Parameterize_with_Shaker_Tutorial.ipynb) | Map and parameterize a simple molecule end-to-end |
| [2 — Complex Mapping](tutorials/2_ComplexMapping/Complex_forward_mapping_with_Shaker.ipynb) | Handle multi-residue and multi-bead mappings |
| [3 — Virtual Sites](tutorials/3_Type3VS/Type3VSwithShaker.ipynb) | Generate and optimize type-3 virtual sites |

---

## Modules

| Module | Description |
|---|---|
| `mapper` | Map AA/QM trajectories to CG representations |
| `measurer` | Measure bonded distributions across trajectories |
| `estimator` | Estimate GROMACS bonded parameters from distributions |
| `dihedral_fitting` | Inverted Boltzmann dihedral fitting with cosine series |
| `sasa` | SASA calculation with CG van der Waals radii |
| `vsites` | Virtual site generation and molecule alignment |
| `overlap` | Bead overlap matrix analysis |
| `system_builders` | GROMACS simulation setup and execution |
| `itp` | GROMACS topology (`.itp`) file generation |
| `plot` | Publication-quality distribution and SASA plots |
| `render` | 3D visualization and 2D chemical structure drawing |

---

## License

LGPLv2.1 License. See [LICENSE](LICENSE) for details.
