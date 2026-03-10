# SHAKER
Tools for mapping, parameterizing, and validating Martini coarse-grained models.

SHAKER is a Python toolkit designed to facilitate the systematic parameterization and benchmarking of coarse-grained (CG) models against atomistic simulations, with a particular focus on the Martini force field.

The package provides modular utilities for tasks such as mapping atomistic trajectories to CG representations, analyzing bonded distributions, preparing simulation systems, and visualizing mappings and structural properties.

SHAKER is primarily intended to be used interactively within Python environments, such as Jupyter notebooks, where users can combine its tools into flexible workflows for CG model development and validation.

This interactive approach makes it easy to iterate on mappings, inspect intermediate results, and rapidly test new parameterization strategies.

## Installation

### From PyPI (Soooooon... not yet.):  

### From Source:  

```bash
git clone https://github.com/YOURUSERNAME/shaker.git
cd shaker
pip install -e .
# or simply
pip install git+https://https://github.com/Lp0lp/shaker.git
```

## Main features

### Efficient CG Mapping
Map atomistic/QM trajectories onto coarse-grained representations using
flexible bead definitions.

```python
from shaker import mapper

mapping = {
    "MOL": {
        "R1": {"type": "SX3",  "charge": 0, "atoms": ['Cl1','Cl1','C0B','C0B','C0A','C05']},
        "R2": {"type": "SX3",  "charge": 0, "atoms": ['C06','C06','C05','C08','Cl0','Cl0']},
        ...
    },
}

shaker.mapper.map_aa2cg("aa.gro", "aa.xtc", mapping)
```

### Bonded distribution analysis
Quickly compare bonded distributions between reference AA/QM and CG simulations.

```python
from shaker import analysis

results = analysis.measure_bonded_terms(
    universe,
    resname="MOL",
    dist_tgts=[("B1","B2")],
    ang_tgts=[("B1","B2","B3")],
    dihed_tgts=[])

fig = shaker.plot.plot_bonded_distributions(
            AA_bonded,
            CG_bonded,
            labels=["AA", "CG"],
            colors=["tab:blue", "tab:red"],
            outfile="cleanbonds")
```

### Quick simulation setup for efficient iterative testing.
Quickly prepare and run Martini simulation systems, removing much of the
boilerplate involved in system preparation. This allows rapid iteration during
the parameterization process, making it easier to test and refine CG models.

```python
from shaker import simulation

simulation.prepare_setup_water("cg_structure.gro")
simulation.runSim()
```

### Plotting, visualization, rendering and more!
Visualize ensembles, calculate molecular SASA, generate virtual-site definitions...


## Requirements
WIP.

## License
WIP.