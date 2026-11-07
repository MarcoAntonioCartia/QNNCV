"""
Point-wise critics: D(x, y) -> R evaluated on grid points, scored against a
density by a Monte-Carlo sum over a sampled point set (the 12 Jul 2026
spec of the CV quantum critic, shared here by every point-wise critic so
the classical / CV / DV versions are compared on identical footing).

    score_b = (G^2 / |S|) * sum_{s in S} x_b(s) * D(s)

with S a set of n_points grid indices drawn per call (n_points = 0 means
the full grid, which is the exact sum_s x_b(s) D(s)). The critic input is
the trainer's critic tensor (B, G, G, 1) or (B, G, G). The score is linear
in the input, so the WGAN gradient penalty sees (G^2/|S|) D(S) scattered
on S. Instance noise and blur act on x before this point and therefore
average through the sum; the spec notes they do little on the point path.

Grid convention = src/families/base.py: X, Y = np.meshgrid(xvec, yvec), so
flat index k = i*G + j maps to (x = xvec[j], y = yvec[i]).
"""

import numpy as np
import tensorflow as tf

from src.backends.base import CriticBackend


class PointwiseCritic(CriticBackend):
    """Subclasses implement `evaluate(points) -> (n,)` for points (n, 2)
    (differentiable w.r.t. trainable_variables) and `describe_extra()`."""

    def __init__(self, *, xvec, yvec, n_points=64, seed=0):
        self.xvec = np.asarray(xvec, np.float32)
        self.yvec = np.asarray(yvec, np.float32)
        X, Y = np.meshgrid(self.xvec, self.yvec)
        self.coords = np.stack([X.ravel(), Y.ravel()], axis=1)  # (G^2, 2)
        self.n_cells = int(self.coords.shape[0])
        self.n_points = int(n_points)
        self._rng = np.random.default_rng(seed)
        self.last_points = None

    def sample_points(self):
        if self.n_points <= 0 or self.n_points >= self.n_cells:
            return np.arange(self.n_cells)
        return np.sort(self._rng.choice(self.n_cells, self.n_points, replace=False))

    def evaluate(self, points):
        raise NotImplementedError

    def __call__(self, x, training=False, points=None):
        x = tf.convert_to_tensor(x, tf.float32)
        B = int(x.shape[0])
        flat = tf.reshape(x, [B, -1])
        if int(flat.shape[1]) != self.n_cells:
            raise ValueError(f"critic input has {int(flat.shape[1])} cells, "
                             f"grid has {self.n_cells}")
        idx = self.sample_points() if points is None else np.asarray(points)
        self.last_points = idx
        d = self.evaluate(tf.constant(self.coords[idx]))          # (|S|,)
        xs = tf.gather(flat, idx, axis=1)                           # (B, |S|)
        scale = self.n_cells / float(len(idx))
        return scale * tf.reduce_sum(xs * d[None, :], axis=1, keepdims=True)

    def describe(self):
        d = {'backend': self.name,
             'n_params': int(sum(int(np.prod(v.shape)) for v in self.trainable_variables)),
             'hilbert_dim': None, 'n_points': self.n_points, 'pointwise': True}
        d.update(self.describe_extra())
        return d

    def describe_extra(self):
        return {}
