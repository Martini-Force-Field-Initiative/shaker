from importlib.metadata import version
__version__ = version("shaker")

from . import dihedral_fitting, itp, mapper, measurer, sasa, system_builders, vsites, helper, plot, render, vmd_render, estimator