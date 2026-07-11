"""
Gaussian distribution family
============================

Extracted 1:1 from train_2d_qgan.py (behavior-preserving refactor).
"""

import numpy as np

from src.families.base import DistributionFamily


class GaussianFamily(DistributionFamily):
    """Independent 2D Gaussians with varying parameters."""

    def __init__(self, grid_size=40, x_range=3.0,
                 center_range=1.0, sigma_range=(0.3, 0.8)):
        super().__init__(grid_size, x_range)
        self.center_range = center_range
        self.sigma_range = sigma_range

    def sample(self):
        cx = np.random.uniform(-self.center_range, self.center_range)
        cy = np.random.uniform(-self.center_range, self.center_range)
        sigma = np.random.uniform(*self.sigma_range)

        dist = self._make_gaussian(cx, cy, sigma)
        params = {'cx': cx, 'cy': cy, 'sigma': sigma}
        return dist, params

    def get_canonical(self):
        sigma = (self.sigma_range[0] + self.sigma_range[1]) / 2
        dist = self._make_gaussian(0, 0, sigma)
        return dist, {'cx': 0, 'cy': 0, 'sigma': sigma}

    def _make_gaussian(self, cx, cy, sigma):
        dist = np.exp(-((self.X - cx)**2 + (self.Y - cy)**2) / (2 * sigma**2))
        return dist / (dist.sum() + 1e-10)
