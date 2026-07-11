"""
Golden: metrics — marginal W1 (compute_wasserstein_2d), energy-distance
context + distances, and the full validate() function via a stub generator.

The stub generator returns fixed seeded densities, so all five validation
metrics (canonical_w1, nearest_w1, diversity, canonical_ed, nearest_ed) are
frozen end-to-end, including the nearest-member searches over the
precomputed grid-distance matrix.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (load_t2q, seed_all, save_golden_npz, load_golden_npz,
                      assert_exact, assert_close, main_dispatch)

GOLDEN = 'metrics.npz'


def gaussian_blob(X, Y, cx, cy, sx, sy):
    import numpy as np
    d = np.exp(-((X - cx) ** 2) / (2 * sx ** 2)
               - ((Y - cy) ** 2) / (2 * sy ** 2))
    return (d / d.sum()).astype(np.float64)


def fixed_setup():
    """Grid + 6 val densities + canonical + 10 stub-generator densities."""
    import numpy as np
    xvec = np.linspace(-3.0, 3.0, 40)
    X, Y = np.meshgrid(xvec, xvec)
    rs = np.random.RandomState(21)
    val = [gaussian_blob(X, Y,
                         rs.uniform(-1, 1), rs.uniform(-1, 1),
                         rs.uniform(0.3, 0.8), rs.uniform(0.3, 0.8))
           for _ in range(6)]
    canonical = gaussian_blob(X, Y, 0.0, 0.0, 0.55, 0.55)
    rs2 = np.random.RandomState(31)
    gen = [gaussian_blob(X, Y,
                         rs2.uniform(-1, 1), rs2.uniform(-1, 1),
                         rs2.uniform(0.3, 0.8), rs2.uniform(0.3, 0.8))
           for _ in range(10)]
    return xvec, X, Y, val, canonical, gen


class StubGenerator:
    """Yields fixed densities in order; ignores z (validate draws z anyway)."""
    batch_size = None
    latent_dim = 4

    def __init__(self, densities):
        import tensorflow as tf
        self._tensors = [tf.constant(d, dtype=tf.float32) for d in densities]
        self._i = 0

    def generate_distribution_2d(self, z, xvec, yvec):
        t = self._tensors[self._i % len(self._tensors)]
        self._i += 1
        return t


def compute():
    import numpy as np
    mod = load_t2q()
    xvec, X, Y, val, canonical, gen = fixed_setup()
    out = {}

    # --- marginal W1 on fixed pairs (pure numpy/scipy -> exact) ---
    pairs = [(0, 1), (2, 3), (4, 5), (0, 0)]
    out['w1_pairs'] = np.array(
        [mod.compute_wasserstein_2d(val[i], val[j]) for i, j in pairs])
    out['w1_canon'] = np.array(
        [mod.compute_wasserstein_2d(canonical, v) for v in val])
    # degenerate-input behavior (asserted inline, no golden needed)
    assert mod.compute_wasserstein_2d(np.zeros((40, 40)), val[0]) == float('inf')
    bad = val[0].copy(); bad[0, 0] = np.nan
    assert mod.compute_wasserstein_2d(bad, val[0]) == float('inf')

    # --- energy-distance context + distances ---
    val_set = [(v, {}) for v in val]
    ctx = mod.build_energy_distance_context(X, Y, val_set, canonical)
    out['ed_val_self'] = ctx['val_self']
    out['ed_canon_self'] = np.array(ctx['canon_self'])
    out['ed_canon_Dq'] = ctx['canon_Dq']
    out['ed_val_Dq'] = ctx['val_Dq']
    out['ed_D_spots'] = ctx['D'][::97, ::101]
    eds = [mod.energy_distances(g, ctx) for g in gen[:4]]
    out['ed_canon_dists'] = np.array([e[0] for e in eds])
    out['ed_near_dists'] = np.array([e[1] for e in eds])

    # --- full validate() with stub generator ---
    seed_all(0)  # validate consumes tf.random.normal per sample draw
    stub = StubGenerator(gen)
    metrics = mod.validate(stub, canonical, val_set, xvec, xvec, ed_ctx=ctx,
                           n_samples=10)
    for k in ('canonical_w1', 'nearest_w1', 'diversity',
              'canonical_ed', 'nearest_ed'):
        out[f'validate_{k}'] = np.array(metrics[k])

    return out


def generate(force=False):
    save_golden_npz(GOLDEN, compute(), force=force)


def test():
    golden = load_golden_npz(GOLDEN)
    current = compute()
    exact_keys = {'w1_pairs', 'w1_canon'}
    for key in golden.files:
        if key in exact_keys:
            assert_exact(current[key], golden[key], label=key)
        else:
            assert_close(current[key], golden[key], label=key)
        print(f"  ok: {key}")


if __name__ == '__main__':
    main_dispatch(generate, test, 'test_metrics')
