"""
Marginal-W1 metric for 2D densities
===================================

Extracted 1:1 from train_2d_qgan.py (behavior-preserving refactor).
"""

import numpy as np
from scipy.stats import wasserstein_distance


def compute_wasserstein_2d(p, q):
    """Approximate 2D Wasserstein distance via marginals."""
    p = np.asarray(p)
    q = np.asarray(q)

    if not np.isfinite(p).all() or not np.isfinite(q).all():
        return float('inf')

    p_sum = p.sum()
    q_sum = q.sum()

    if p_sum <= 0 or q_sum <= 0:
        return float('inf')

    p = p / p_sum
    q = q / q_sum

    # Marginals
    p_x = np.clip(p.sum(axis=0), 1e-10, None)
    p_y = np.clip(p.sum(axis=1), 1e-10, None)
    q_x = np.clip(q.sum(axis=0), 1e-10, None)
    q_y = np.clip(q.sum(axis=1), 1e-10, None)

    p_x = p_x / p_x.sum()
    p_y = p_y / p_y.sum()
    q_x = q_x / q_x.sum()
    q_y = q_y / q_y.sum()

    try:
        w_x = wasserstein_distance(range(len(p_x)), range(len(q_x)), p_x, q_x)
        w_y = wasserstein_distance(range(len(p_y)), range(len(q_y)), p_y, q_y)
        return (w_x + w_y) / 2
    except:
        return float('inf')
