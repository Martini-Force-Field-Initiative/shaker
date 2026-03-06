# SHAKER
Tools for mapping, parameterizing, and validating Martini coarse-grained models.

SHAKER is a Python toolkit designed to facilitate the systematic parameterization and benchmarking of coarse-grained (CG) models against atomistic simulations, with a particular focus on Martini force field models. It provides utilities for mapping atomistic trajectories to CG representations, analyzing bonded distributions, preparing simulation systems, and visualizing CG mappings, enabling efficient development and validation of coarse-grained molecular models.

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

mapper.map_aa2cg(
    gro="aa.gro",
    xtc="aa.xtc",
    resnames=["PEPT"],
    bead_assignments=[beads],
    bead_names=[names])
```

### Bonded distribution analysis
Quickly compare bonded distributions between reference AA/QM and CG simulations.

```python
from shaker import analysis

results = analysis.measure_bonded_terms(
    universe,
    resname="PEPT",
    dist_tgts=[("B1","B2")],
    ang_tgts=[("B1","B2","B3")],
    dihed_tgts=[])
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

## Requirements
WIP.

## License
WIP.