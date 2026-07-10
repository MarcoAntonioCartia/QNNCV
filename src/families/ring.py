"""
Ring distribution family
========================

Extracted 1:1 from train_2d_qgan.py (behavior-preserving refactor).
"""

import numpy as np

from src.families.base import DistributionFamily


class RingFamily(DistributionFamily):
    """Ring distributions with varying parameters."""

    def __init__(self, grid_size=40, x_range=3.0,
                 center_range=0.5, radius_range=(0.8, 1.5), width_range=(0.15, 0.35)):
        super().__init__(grid_size, x_range)
        self.center_range = center_range
        self.radius_range = radius_range
        self.width_range = width_range

    def sample(self):
        cx = np.random.uniform(-self.center_range, self.center_range)
        cy = np.random.uniform(-self.center_range, self.center_range)
        radius = np.random.uniform(*self.radius_range)
        width = np.random.uniform(*self.width_range)

        dist = self._make_ring(cx, cy, radius, width)
        params = {'cx': cx, 'cy': cy, 'radius': radius, 'width': width}
        return dist, params

    def get_canonical(self):
        radius = (self.radius_range[0] + self.radius_range[1]) / 2
        width = (self.width_range[0] + self.width_range[1]) / 2
        dist = self._make_ring(0, 0, radius, width)
        return dist, {'cx': 0, 'cy': 0, 'radius': radius, 'width': width}

    def _make_ring(self, cx, cy, radius, width):
        r = np.sqrt((self.X - cx)**2 + (self.Y - cy)**2)
        dist = np.exp(-0.5 * ((r - radius) / width)**2)
        return dist / (dist.sum() + 1e-10)
