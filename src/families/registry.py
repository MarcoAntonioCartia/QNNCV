"""
Family registry
===============

Extracted 1:1 from train_2d_qgan.py (behavior-preserving refactor).
"""

from src.families.gaussian import GaussianFamily
from src.families.ring import RingFamily
from src.families.correlated import CorrelatedGaussianFamily
from src.families.four_gaussians import FourGaussiansFamily
from src.families.vibronic import VibronicFamily


def get_family(name, grid_size=40, x_range=3.0):
    """Get distribution family by name."""
    families = {
        'gaussian': GaussianFamily,
        'ring': RingFamily,
        'correlated': CorrelatedGaussianFamily,
        'four_gaussians': FourGaussiansFamily,
        'vibronic': VibronicFamily,
    }
    if name not in families:
        raise ValueError(f"Unknown family: {name}. Available: {list(families.keys())}")
    return families[name](grid_size=grid_size, x_range=x_range)
