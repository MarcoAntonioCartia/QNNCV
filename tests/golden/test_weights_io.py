"""
Structural suite for src/quantum/weights_io.py (no golden data):

- v1 (input_displacement): save_weights with w_s=None writes a raw flat
  array loadable WITHOUT allow_pickle — guards the legacy byte contract the
  e2e best_weights goldens rely on.
- v2 (interleaved): dict round-trips weights and w_s exactly.
- v1 file -> interleaved generator: weights assigned, w_s stays zero.
- v2 file -> input_displacement generator: ValueError.
- Shape mismatches: ValueError.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (load_t2q, seed_all, assert_exact, scratch_dir,
                      main_dispatch)

MODES = 2
LAYERS = 2
CUTOFF = 4


def build(mod, encoding=None):
    seed_all(0)
    return mod.CVQGANGenerator(n_modes=MODES, n_layers=LAYERS,
                               cutoff_dim=CUTOFF, latent_scale=1.0,
                               encoding=encoding)


def test():
    import numpy as np
    mod = load_t2q()  # inserts the repo root on sys.path
    from src.quantum import weights_io
    EncodingSpec = mod.EncodingSpec

    tmp = scratch_dir('weights_io')
    g_id = build(mod)
    g_is = build(mod, encoding=EncodingSpec('interleaved_squeeze'))
    w = g_id.weights.numpy()
    ws = np.random.RandomState(3).standard_normal(
        (LAYERS, MODES)).astype(np.float32) * 0.1

    # --- v1: raw array, loads WITHOUT allow_pickle -------------------------
    v1 = os.path.join(tmp, 'v1_weights.npy')
    weights_io.save_weights(v1, w, w_s=None)
    raw = np.load(v1)  # allow_pickle defaults to False: must not raise
    assert_exact(raw, w, label='v1 raw bytes')
    loaded = weights_io.load_weights(v1)
    assert loaded['w_s'] is None, "v1 load produced a w_s"
    assert_exact(loaded['weights'], w, label='v1 weights')
    print("  ok: v1 raw array round-trip (allow_pickle NOT required)")

    # --- v2: dict round-trip exact -----------------------------------------
    v2 = os.path.join(tmp, 'v2_weights.npy')
    weights_io.save_weights(v2, w, w_s=ws)
    loaded = weights_io.load_weights(v2)
    assert loaded['w_s'] is not None, "v2 load lost w_s"
    assert_exact(loaded['weights'], w, label='v2 weights')
    assert_exact(loaded['w_s'], ws, label='v2 w_s')
    assert loaded['meta'].get('format') == weights_io.FORMAT_V2
    assert loaded['meta'].get('encoding') == 'interleaved_squeeze'
    print("  ok: v2 dict round-trip exact (format/encoding metadata present)")

    # --- v1 -> interleaved generator: warm start, w_s stays 0 --------------
    g_is.weights.assign(np.zeros_like(w))
    weights_io.apply_weights(g_is, weights_io.load_weights(v1), source=v1)
    assert_exact(g_is.weights.numpy(), w, label='v1->interleaved weights')
    assert float(np.max(np.abs(g_is.w_s.numpy()))) == 0.0, (
        "v1 -> interleaved must leave w_s at zero")
    print("  ok: v1 -> interleaved warm start (w_s stays 0)")

    # --- v2 -> interleaved generator: both assigned ------------------------
    weights_io.apply_weights(g_is, weights_io.load_weights(v2), source=v2)
    assert_exact(g_is.w_s.numpy(), ws, label='v2->interleaved w_s')
    print("  ok: v2 -> interleaved assigns w_s")

    # --- v2 -> input_displacement generator: hard error --------------------
    try:
        weights_io.apply_weights(g_id, weights_io.load_weights(v2), source=v2)
    except ValueError as e:
        assert 'interleaved_squeeze' in str(e)
        print("  ok: v2 -> input_displacement raises ValueError")
    else:
        raise AssertionError("v2 -> input_displacement did not raise")

    # --- shape mismatches: hard errors -------------------------------------
    try:
        weights_io.apply_weights(
            g_id, {'weights': w[:-1], 'w_s': None, 'meta': {}}, source='bad')
    except ValueError as e:
        assert 'shape' in str(e)
        print("  ok: weights shape mismatch raises ValueError")
    else:
        raise AssertionError("weights shape mismatch did not raise")
    try:
        weights_io.apply_weights(
            g_is, {'weights': w, 'w_s': ws[:1], 'meta': {}}, source='bad')
    except ValueError as e:
        assert 'w_s shape' in str(e)
        print("  ok: w_s shape mismatch raises ValueError")
    else:
        raise AssertionError("w_s shape mismatch did not raise")


def generate(force=False):
    # Structural test — no golden data to generate; run the checks instead.
    test()


if __name__ == '__main__':
    main_dispatch(generate, test, 'test_weights_io')
