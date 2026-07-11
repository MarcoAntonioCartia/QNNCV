"""
Hermite Basis for Fock -> Position transformation
==================================================

Extracted 1:1 from train_2d_qgan.py (behavior-preserving refactor).
"""

import numpy as np
import tensorflow as tf
from scipy.special import hermite
from math import factorial, sqrt, pi


def compute_hermite_basis(xvec, cutoff_dim):
    """
    Compute Hermite basis functions φ_n(x) for n = 0, 1, ..., cutoff-1.

    φ_n(x) = (1/sqrt(2^n n! sqrt(π))) H_n(x) exp(-x²/2)

    Args:
        xvec: (n_points,) array of x values
        cutoff_dim: Fock space cutoff

    Returns:
        basis: (n_points, cutoff) array of basis function values
    """
    xvec = np.asarray(xvec)
    n_points = len(xvec)
    basis = np.zeros((n_points, cutoff_dim), dtype=np.float64)

    for n in range(cutoff_dim):
        Hn = hermite(n)
        norm = 1.0 / sqrt(2**n * factorial(n) * sqrt(pi))
        basis[:, n] = norm * Hn(xvec) * np.exp(-xvec**2 / 2)

    return tf.constant(basis, dtype=tf.float32)


def recommend_cutoff(n_modes, base_cutoff=8, max_fock_states=5000):
    """
    Suggest cutoff dimension that keeps Fock space manageable.

    The Fock space has cutoff^n_modes states. For large n_modes, the default
    cutoff may use too much memory.

    Args:
        n_modes: Total number of qumodes
        base_cutoff: Desired cutoff (will be reduced if needed)
        max_fock_states: Maximum acceptable Fock space size

    Returns:
        Recommended cutoff dimension
    """
    cutoff = base_cutoff
    while cutoff ** n_modes > max_fock_states and cutoff > 4:
        cutoff -= 1
    return cutoff
