"""
Four-Gaussians distribution family
==================================

Extracted 1:1 from train_2d_qgan.py (behavior-preserving refactor).
"""

import numpy as np

from src.families.base import DistributionFamily


class FourGaussiansFamily(DistributionFamily):
    """Four Gaussians at corners of a square. Tests multi-modal 2D learning."""

    def __init__(self, grid_size=40, x_range=3.0,
                 center_range=0.3, spread_range=(1.0, 1.8), sigma_range=(0.2, 0.4)):
        super().__init__(grid_size, x_range)
        self.center_range = center_range
        self.spread_range = spread_range
        self.sigma_range = sigma_range

    def sample(self):
        cx = np.random.uniform(-self.center_range, self.center_range)
        cy = np.random.uniform(-self.center_range, self.center_range)
        spread = np.random.uniform(*self.spread_range)
        sigma = np.random.uniform(*self.sigma_range)

        dist = self._make_four_gaussians(cx, cy, spread, sigma)
        params = {'cx': cx, 'cy': cy, 'spread': spread, 'sigma': sigma}
        return dist, params

    def get_canonical(self):
        spread = (self.spread_range[0] + self.spread_range[1]) / 2
        sigma = (self.sigma_range[0] + self.sigma_range[1]) / 2
        dist = self._make_four_gaussians(0, 0, spread, sigma)
        return dist, {'cx': 0, 'cy': 0, 'spread': spread, 'sigma': sigma}

    def _make_four_gaussians(self, cx, cy, spread, sigma):
        """Four Gaussians at corners of a square centered at (cx, cy)."""
        corners = [
            (cx - spread, cy - spread),
            (cx - spread, cy + spread),
            (cx + spread, cy - spread),
            (cx + spread, cy + spread),
        ]

        dist = np.zeros_like(self.X)
        for gx, gy in corners:
            dist += np.exp(-((self.X - gx)**2 + (self.Y - gy)**2) / (2 * sigma**2))

        return dist / (dist.sum() + 1e-10)
