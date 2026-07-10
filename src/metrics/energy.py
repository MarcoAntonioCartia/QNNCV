"""
Energy-distance metric on the 2D grid
=====================================

Extracted 1:1 from train_2d_qgan.py (behavior-preserving refactor).
"""

import numpy as np


def build_energy_distance_context(X, Y, val_set, canonical_target):
    """
    Precompute what's needed to evaluate 2D energy distances cheaply.

    Energy distance between distributions p, q supported on the same grid:
        ED(p, q) = 2 p^T D q  -  p^T D p  -  q^T D q
    with D[i, j] = Euclidean distance between grid points i and j (in x/y
    units, NOT grid cells). It is a proper metric on distributions and is
    sensitive to the full 2D geometry -- unlike compute_wasserstein_2d,
    which only compares marginals and cannot distinguish e.g. a ring from
    a 4-lobe pattern with matching marginals.

    Precomputing D @ q for every validation member turns nearest-member
    search into one matrix-vector product per generated sample.
    """
    coords = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=-1))  # (G^2, G^2), ~10 MB at 40x40

    def _flat(p):
        p = np.asarray(p, dtype=np.float32).ravel()
        return p / (p.sum() + 1e-10)

    val_flat = np.stack([_flat(m) for m, _ in val_set], axis=1)  # (G^2, n_val)
    val_Dq = D @ val_flat                                        # (G^2, n_val)
    val_self = np.einsum('ij,ij->j', val_flat, val_Dq)           # (n_val,)

    canon_flat = _flat(canonical_target)
    canon_Dq = D @ canon_flat
    canon_self = float(canon_flat @ canon_Dq)

    return {'D': D, 'val_Dq': val_Dq, 'val_self': val_self,
            'canon_Dq': canon_Dq, 'canon_self': canon_self}


def energy_distances(p, ed_ctx):
    """Return (ED to canonical, ED to nearest validation member) for density p."""
    p = np.asarray(p, dtype=np.float32).ravel()
    p = p / (p.sum() + 1e-10)
    Dp = ed_ctx['D'] @ p
    p_self = float(p @ Dp)

    ed_canon = 2.0 * float(p @ ed_ctx['canon_Dq']) - p_self - ed_ctx['canon_self']
    ed_val = 2.0 * (p @ ed_ctx['val_Dq']) - p_self - ed_ctx['val_self']
    return ed_canon, float(ed_val.min())
