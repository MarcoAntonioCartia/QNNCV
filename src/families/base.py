"""
Distribution-family base class
==============================

Extracted 1:1 from train_2d_qgan.py (behavior-preserving refactor).
"""

import numpy as np


class DistributionFamily:
    """Base class for distribution families."""

    def __init__(self, grid_size, x_range):
        self.grid_size = grid_size
        self.x_range = x_range
        self.xvec = np.linspace(-x_range, x_range, grid_size)
        self.yvec = np.linspace(-x_range, x_range, grid_size)
        self.X, self.Y = np.meshgrid(self.xvec, self.yvec)

    def sample(self):
        """Sample a distribution from the family."""
        raise NotImplementedError

    def get_canonical(self):
        """Get the canonical member."""
        raise NotImplementedError
