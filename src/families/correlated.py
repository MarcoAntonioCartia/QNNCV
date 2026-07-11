"""
Correlated-Gaussian distribution family
=======================================

Extracted 1:1 from train_2d_qgan.py (behavior-preserving refactor).
"""

import numpy as np

from src.families.base import DistributionFamily


class CorrelatedGaussianFamily(DistributionFamily):
    """Correlated 2D Gaussians."""

    def __init__(self, grid_size=40, x_range=3.0,
                 center_range=0.5, rho_range=(0.4, 0.85), scale_range=(0.6, 1.2)):
        super().__init__(grid_size, x_range)
        self.center_range = center_range
        self.rho_range = rho_range
        self.scale_range = scale_range

    def sample(self):
        cx = np.random.uniform(-self.center_range, self.center_range)
        cy = np.random.uniform(-self.center_range, self.center_range)
        rho = np.random.uniform(*self.rho_range)
        scale = np.random.uniform(*self.scale_range)

        dist = self._make_correlated(cx, cy, rho, scale)
        params = {'cx': cx, 'cy': cy, 'rho': rho, 'scale': scale}
        return dist, params

    def get_canonical(self):
        rho = (self.rho_range[0] + self.rho_range[1]) / 2
        scale = (self.scale_range[0] + self.scale_range[1]) / 2
        dist = self._make_correlated(0, 0, rho, scale)
        return dist, {'cx': 0, 'cy': 0, 'rho': rho, 'scale': scale}

    def _make_correlated(self, cx, cy, rho, scale):
        cov = np.array([[1, rho], [rho, 1]]) * scale**2
        cov_inv = np.linalg.inv(cov)
        det = np.linalg.det(cov)

        dx = self.X - cx
        dy = self.Y - cy

        mahal = (cov_inv[0,0] * dx**2 +
                 2 * cov_inv[0,1] * dx * dy +
                 cov_inv[1,1] * dy**2)

        dist = np.exp(-0.5 * mahal) / (2 * np.pi * np.sqrt(det))
        return dist / (dist.sum() + 1e-10)
