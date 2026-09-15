import warnings
from importlib.metadata import version

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API",
    category=UserWarning,
    module=r"nglview",
)

__version__ = version("shaker")

__all__ = [
    "__version__",
    "align_mol_to_single_traj",
    "assess_overlap_matrix",
    "bonded_estimator",
    "dihedral_fitting",
    "estimator",
    "generate_virtual_sites3",
    "generate_virtual_sitesN",
    "helper",
    "itp",
    "list_iterations",
    "map_aa2cg",
    "mapper",
    "measure_bonded_terms",
    "measurer",
    "overlap",
    "plot",
    "plot_bonded_distributions",
    "plot_sasa_dir",
    "prepare_setup_water",
    "render",
    "render_2d",
    "render_2dMapping",
    "render_connely_surface",
    "render_ensemble",
    "render_mapping",
    "runSim",
    "run_SASA",
    "sasa",
    "system_builders",
    "vsites",
    "write_initial_CGitp",
]

from . import (
    dihedral_fitting,
    estimator,
    helper,
    itp,
    mapper,
    measurer,
    overlap,
    plot,
    render,
    render_2d,
    sasa,
    system_builders,
    vsites,
)
from .estimator import bonded_estimator
from .itp import write_initial_CGitp
from .mapper import map_aa2cg
from .measurer import measure_bonded_terms
from .overlap import assess_overlap_matrix
from .plot import plot_bonded_distributions, plot_sasa_dir
from .render import render_connely_surface, render_ensemble, render_mapping
from .render_2d import render_2dMapping
from .sasa import run_SASA
from .system_builders import list_iterations, prepare_setup_water, runSim
from .vsites import (
    align_mol_to_single_traj,
    generate_virtual_sites3,
    generate_virtual_sitesN,
)
