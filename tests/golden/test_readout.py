"""
Golden: Hermite-function readout (Fock ket -> 40x40 position density),
with and without ancilla partial trace, for 2, 3 and 4 total modes.

Mechanism: build a real CVQGANGenerator (so shapes/attrs are authentic),
then monkeypatch the INSTANCE's _run_circuit to return a fixed, seeded
fake ket. generate_distribution_2d / generate_batch always route through
self._run_circuit, so this patch survives every extraction phase.

Covers: sequential path, batched-engine path, and the single-z-on-a-
batched-engine slice path (ket[0]).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (load_t2q, seed_all, FakeState, save_golden_npz,
                      load_golden_npz, assert_close, main_dispatch)

GOLDEN = 'readout.npz'

CUTOFF = 4
MODES = [2, 3, 4]
BATCH = 3  # != CUTOFF so shape bugs can't cancel out


def fixed_ket(shape, seed):
    import numpy as np
    rs = np.random.RandomState(seed)
    ket = rs.standard_normal(shape) + 1j * rs.standard_normal(shape)
    ket = ket / np.sqrt((np.abs(ket) ** 2).sum())
    return ket.astype(np.complex64)


def compute():
    import numpy as np
    import tensorflow as tf
    mod = load_t2q()
    xvec = np.linspace(-3.0, 3.0, 40)
    out = {'xvec': xvec}

    for m in MODES:
        ket_np = fixed_ket((CUTOFF,) * m, seed=7 + m)
        ket_b_np = fixed_ket((BATCH,) + (CUTOFF,) * m, seed=70 + m)

        # --- sequential engine ---
        # latent_scale=1.0: former signature default, passed explicitly since
        # the default unification; readout.npz was generated under it
        seed_all(0)
        g = mod.CVQGANGenerator(n_modes=m, n_layers=1, cutoff_dim=CUTOFF,
                                latent_scale=1.0)
        g._run_circuit = lambda z, _k=ket_np: FakeState(tf.constant(_k))
        prob, ket_norm = g.generate_distribution_2d(
            tf.zeros([g.latent_dim]), xvec, xvec, return_ket_norm=True)
        out[f'prob_seq_m{m}'] = prob.numpy()
        out[f'ketnorm_seq_m{m}'] = np.array(float(ket_norm))

        # --- batched engine ---
        seed_all(0)
        gb = mod.CVQGANGenerator(n_modes=m, n_layers=1, cutoff_dim=CUTOFF,
                                 latent_scale=1.0, batch_size=BATCH)
        gb._run_circuit = lambda z, _k=ket_b_np: FakeState(tf.constant(_k))
        probs_b, norms_b = gb.generate_batch(
            tf.zeros([BATCH, gb.latent_dim]), xvec, xvec,
            return_ket_norm=True)
        out[f'prob_batch_m{m}'] = probs_b.numpy()
        out[f'ketnorm_batch_m{m}'] = norms_b.numpy()

        # --- single z on a batched engine (slices ket[0]) ---
        prob_single = gb.generate_distribution_2d(
            tf.zeros([gb.latent_dim]), xvec, xvec)
        out[f'prob_single_on_batched_m{m}'] = prob_single.numpy()

    return out


def generate(force=False):
    save_golden_npz(GOLDEN, compute(), force=force)


def test():
    golden = load_golden_npz(GOLDEN)
    current = compute()
    assert set(golden.files) == set(current.keys()), (
        f"key mismatch: {sorted(set(golden.files) ^ set(current.keys()))}")
    for key in golden.files:
        assert_close(current[key], golden[key], label=key)
        print(f"  ok: {key}")


if __name__ == '__main__':
    main_dispatch(generate, test, 'test_readout')
