"""
Golden: full SF circuit forward pass with FIXED weights and FIXED z.

For 2 and 3 total modes (2 layers, cutoff 6):
  - weights are stored IN the golden and assigned on replay (decouples the
    test from TF RNG stream evolution),
  - batched (batch_size=4) and sequential engines both run,
  - each output is compared to its own golden at 1e-6, and batched vs
    sequential agreement is asserted at 1e-4 (mirroring
    scripts/verify_batching.py).

This pins the symbolic program itself: encoding Dgates (z_i * latent_scale),
interferometer/squeeze/displacement/Kerr gate order, parameter names z_i/w_i.
SEAM 1 (Phase 6) must leave every value here bit-compatible.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (load_t2q, seed_all, save_golden_npz, load_golden_npz,
                      assert_close, main_dispatch)

GOLDEN = 'circuit_forward.npz'

MODES = [2, 3]
LAYERS = 2
CUTOFF = 6
BATCH = 4


def make_weights_and_z(mod, m):
    """Fresh generator weights (seeded) + fixed z batch, as numpy."""
    import numpy as np
    seed_all(0)
    g = mod.CVQGANGenerator(n_modes=m, n_layers=LAYERS, cutoff_dim=CUTOFF)
    weights = g.weights.numpy()
    rs = np.random.RandomState(5 + m)
    z = rs.standard_normal((BATCH, g.latent_dim)).astype(np.float32) * 0.5
    return weights, z


def forward(mod, m, weights, z):
    import tensorflow as tf
    seed_all(0)
    g_seq = mod.CVQGANGenerator(n_modes=m, n_layers=LAYERS, cutoff_dim=CUTOFF)
    g_seq.weights.assign(weights)
    seed_all(0)
    g_bat = mod.CVQGANGenerator(n_modes=m, n_layers=LAYERS, cutoff_dim=CUTOFF,
                                batch_size=BATCH)
    g_bat.weights.assign(weights)

    import numpy as np
    xvec = np.linspace(-3.0, 3.0, 40)
    zt = tf.constant(z)
    p_seq, n_seq = g_seq.generate_batch(zt, xvec, xvec, return_ket_norm=True)
    p_bat, n_bat = g_bat.generate_batch(zt, xvec, xvec, return_ket_norm=True)
    return (p_seq.numpy(), n_seq.numpy(), p_bat.numpy(), n_bat.numpy())


def compute(golden=None):
    """If golden given, replay its stored weights/z; else create them."""
    mod = load_t2q()
    out = {}
    for m in MODES:
        if golden is None:
            weights, z = make_weights_and_z(mod, m)
        else:
            weights, z = golden[f'weights_m{m}'], golden[f'z_m{m}']
        p_seq, n_seq, p_bat, n_bat = forward(mod, m, weights, z)
        out[f'weights_m{m}'] = weights
        out[f'z_m{m}'] = z
        out[f'prob_seq_m{m}'] = p_seq
        out[f'ketnorm_seq_m{m}'] = n_seq
        out[f'prob_bat_m{m}'] = p_bat
        out[f'ketnorm_bat_m{m}'] = n_bat
    return out


def generate(force=False):
    save_golden_npz(GOLDEN, compute(), force=force)


def test():
    import numpy as np
    golden = load_golden_npz(GOLDEN)
    current = compute(golden=golden)
    for m in MODES:
        for key in (f'prob_seq_m{m}', f'ketnorm_seq_m{m}',
                    f'prob_bat_m{m}', f'ketnorm_bat_m{m}'):
            assert_close(current[key], golden[key], label=key)
            print(f"  ok: {key}")
        # batched vs sequential agreement (verify_batching.py's bound)
        err = float(np.max(np.abs(
            current[f'prob_seq_m{m}'] - current[f'prob_bat_m{m}'])))
        assert err < 1e-4, f"batched/sequential disagree for m={m}: {err}"
        print(f"  ok: batched==sequential m={m} (max err {err:.2e})")


if __name__ == '__main__':
    main_dispatch(generate, test, 'test_circuit_forward')
