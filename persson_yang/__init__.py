"""
persson_yang — Python port of the Persson-Yang (2008) seal leak-rate simulation.

Based on: Persson & Yang, J. Phys.: Condens. Matter 20 (2008) 315011

Modules
-------
parameters  : Params dataclass (load_parameters)
psd         : idealized_PSD, psd_2D
surface     : artificial_surf, calculate_rms, estimate_hurst_exponent
contact     : eval_contact_area
zoom        : zoom_surface_spatial
percolation : check_percolation_path
core        : compute_uc, compute_Qdot, find_critical
plots       : make_plots  (reproduces Figs 6-11)
fig14       : make_fig14_gif
seal_study  : analyze_seal  (applied example: rubber seal on a rough substrate)
"""

from .parameters import Params, load_parameters
from .psd import PSD, idealized_PSD, psd_2D
from .surface import artificial_surf, calculate_rms, estimate_hurst_exponent
from .contact import eval_contact_area, zoom_surface
from .zoom import zoom_surface_spatial
from .percolation import check_percolation_path
from .core import compute_uc, compute_Qdot, find_critical
from .plots import make_plots
from .fig14 import make_fig14_gif
from .seal_study import analyze_seal

__all__ = [
    "Params", "load_parameters",
    "PSD", "idealized_PSD", "psd_2D",
    "artificial_surf", "calculate_rms", "estimate_hurst_exponent",
    "eval_contact_area", "zoom_surface",
    "zoom_surface_spatial",
    "check_percolation_path",
    "compute_uc", "compute_Qdot", "find_critical",
    "make_plots",
    "make_fig14_gif",
    "analyze_seal",
]
