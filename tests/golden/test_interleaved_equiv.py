"""
Structural/behavioral suite for the interleaved_squeeze encoding (no golden
data — every check is a self-contained property):

1. EQUIVALENCE: with w_s == 0, interleaved output equals input_displacement
   output for identical assigned weights and fixed z — sequential and
   batched engines, at the circuit-golden tolerance (1e-6).
2. GRADIENT LIVENESS: at w_s == 0 with generic z, d(loss)/d(w_s) exists and
   is nonzero (tanh(z) != 0, so the modulation gradient does not vanish at
   the frozen-baseline init).
3. Z-RESPONSE: with w_s nonzero, +z vs -z changes the density's CENTRAL
   second moment (width/photon content), not merely its position — beyond
   what the pure front-displacement (w_s == 0) case shows.
4. BATCHED == SEQUENTIAL at nonzero w_s (1e-4, mirroring verify_batching).
5. INJ-LAYERS SUBSET: injecting into layer 0 only differs from injecting
   into all layers (same nonzero w_s).
6. R-MAX CLIP: asserted directly on the effective r tensor (the binding
   formula) — the unclipped value exceeds r_max, the clipped value obeys
   max|r| <= r_max — plus a behavioral check that the clip reaches the
   circuit (outputs with/without r_max differ under saturating w_s).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import load_t2q, seed_all, assert_close, main_dispatch

MODES = 2
LAYERS = 2
CUTOFF = 6
BATCH = 4


def build(mod, encoding=None, batch_size=None, use_kerr=True):
    seed_all(0)
    return mod.CVQGANGenerator(n_modes=MODES, n_layers=LAYERS,
                               cutoff_dim=CUTOFF, latent_scale=1.0,
                               batch_size=batch_size, encoding=encoding,
                               use_kerr=use_kerr)


def fixed_weights_and_z(mod):
    import numpy as np
    g = build(mod)
    weights = g.weights.numpy()
    rs = np.random.RandomState(7)
    z = rs.standard_normal((BATCH, g.latent_dim)).astype(np.float32) * 0.5
    return weights, z


def central_second_moment(p, xvec):
    """Trace of the density's central (about-centroid) second moment."""
    import numpy as np
    X = np.asarray(xvec)[:, None]
    Y = np.asarray(xvec)[None, :]
    mass = p.sum()
    mux = (p * X).sum() / mass
    muy = (p * Y).sum() / mass
    return float((p * ((X - mux) ** 2 + (Y - muy) ** 2)).sum() / mass)


def test():
    import numpy as np
    import tensorflow as tf
    mod = load_t2q()
    EncodingSpec = mod.EncodingSpec
    weights, z = fixed_weights_and_z(mod)
    xvec = np.linspace(-3.0, 3.0, 40)
    zt = tf.constant(z)

    # --- (1) w_s == 0 equivalence, sequential and batched ------------------
    for bs in (None, BATCH):
        g_id = build(mod, batch_size=bs)
        g_id.weights.assign(weights)
        g_is = build(mod, encoding=EncodingSpec('interleaved_squeeze'),
                     batch_size=bs)
        g_is.weights.assign(weights)
        assert float(tf.reduce_max(tf.abs(g_is.w_s))) == 0.0, "w_s init != 0"
        p_id, n_id = g_id.generate_batch(zt, xvec, xvec, return_ket_norm=True)
        p_is, n_is = g_is.generate_batch(zt, xvec, xvec, return_ket_norm=True)
        tag = 'batched' if bs else 'sequential'
        assert_close(p_is.numpy(), p_id.numpy(), label=f'equiv prob {tag}')
        assert_close(n_is.numpy(), n_id.numpy(), label=f'equiv ketnorm {tag}')
        print(f"  ok: w_s=0 equivalence ({tag})")

    # --- (2) gradient liveness at w_s == 0 ---------------------------------
    g_is = build(mod, encoding=EncodingSpec('interleaved_squeeze'),
                 batch_size=BATCH)
    g_is.weights.assign(weights)
    with tf.GradientTape() as tape:
        prob, kn = g_is.generate_batch(zt, xvec, xvec, return_ket_norm=True)
        loss = (tf.reduce_mean(tf.square(prob))
                + tf.reduce_mean(tf.square(1.0 - kn)))
    grads = tape.gradient(loss, g_is.trainable_variables)
    assert len(grads) == 2, f"expected 2 grads, got {len(grads)}"
    assert grads[1] is not None, "d(loss)/d(w_s) is None at w_s=0"
    ws_gnorm = float(tf.linalg.global_norm([grads[1]]))
    assert ws_gnorm > 0.0, "d(loss)/d(w_s) vanishes at w_s=0 for generic z"
    print(f"  ok: gradient liveness at w_s=0 (|grad w_s| = {ws_gnorm:.2e})")

    # --- (3) z-response: central second moment varies with z ---------------
    # Kerr-free circuit: the layers are then Gaussian-symplectic, so at
    # w_s == 0 the output covariance is EXACTLY displacement-independent —
    # +z vs -z is a pure rigid motion and the central-m2 difference is grid
    # noise. With w_s != 0, z drives the squeeze magnitudes and the width
    # must respond. (With Kerr the baseline is confounded: the nonlinearity
    # makes +/- displaced inputs evolve non-mirror-symmetrically.)
    zi = g_is.encoding.z_index_for_mode(0)
    z_plus = np.zeros(g_is.latent_dim, dtype=np.float32)
    z_plus[zi] = 1.5
    z_minus = -z_plus
    weights_nk = build(mod, use_kerr=False).weights.numpy()

    def m2_diff(ws_value):
        g = build(mod, encoding=EncodingSpec('interleaved_squeeze'),
                  use_kerr=False)
        g.weights.assign(weights_nk)
        g.w_s.assign(np.full((LAYERS, MODES), ws_value, dtype=np.float32))
        m2 = []
        for zv in (z_plus, z_minus):
            p = g.generate_distribution_2d(tf.constant(zv), xvec, xvec)
            m2.append(central_second_moment(p.numpy(), xvec))
        return abs(m2[0] - m2[1])

    diff0 = m2_diff(0.0)     # rigid motion only: ~0 up to grid numerics
    diff3 = m2_diff(0.3)     # z now drives squeeze: width must respond
    assert diff3 > 10 * diff0 + 1e-3, (
        f"z-response too weak: |dm2|(w_s=0.3)={diff3:.4g} vs "
        f"|dm2|(w_s=0)={diff0:.4g}")
    print(f"  ok: z-response (central m2 diff {diff3:.4f} vs "
          f"rigid-motion baseline {diff0:.2e})")

    # --- (4) batched == sequential at nonzero w_s --------------------------
    ws = np.random.RandomState(11).standard_normal(
        (LAYERS, MODES)).astype(np.float32) * 0.2
    outs = {}
    for bs in (None, BATCH):
        g = build(mod, encoding=EncodingSpec('interleaved_squeeze'),
                  batch_size=bs)
        g.weights.assign(weights)
        g.w_s.assign(ws)
        outs[bs] = g.generate_batch(zt, xvec, xvec).numpy()
    err = float(np.max(np.abs(outs[None] - outs[BATCH])))
    assert err < 1e-4, f"batched/sequential disagree at w_s!=0: {err}"
    print(f"  ok: batched==sequential at w_s!=0 (max err {err:.2e})")

    # --- (5) inj_layers subset differs from all-layers ---------------------
    g_sub = build(mod, encoding=EncodingSpec('interleaved_squeeze',
                                             inj_layers=(0,)))
    g_sub.weights.assign(weights)
    g_sub.w_s.assign(ws)
    p_sub = g_sub.generate_batch(zt, xvec, xvec).numpy()
    diff = float(np.max(np.abs(p_sub - outs[None])))
    assert diff > 1e-6, "inj_layers=(0,) output identical to all-layers"
    print(f"  ok: inj_layers subset differs from all-layers "
          f"(max diff {diff:.2e})")

    # --- (6) r_max clip on the effective r tensor --------------------------
    R_MAX = 0.1
    big_ws = np.full((LAYERS, MODES), 5.0, dtype=np.float32)
    g_clip = build(mod, encoding=EncodingSpec('interleaved_squeeze',
                                              r_max=R_MAX))
    g_clip.weights.assign(weights)
    g_clip.w_s.assign(big_ws)
    # Effective r exactly as _run_circuit binds it, per layer/mode:
    M = g_clip.interferometer_params
    z_sat = tf.constant(z_plus)  # tanh(1.5) ~ 0.905: saturating drive
    r_unclipped, r_clipped = [], []
    for l in range(LAYERS):
        for m in range(MODES):
            idx = l * g_clip.params_per_layer + M + 2 * m
            zc = z_sat[g_clip.encoding.z_index_for_mode(m)]
            r = g_clip.weights[idx] + g_clip.w_s[l, m] * tf.tanh(zc)
            r_unclipped.append(float(r))
            r_clipped.append(float(tf.clip_by_value(r, -R_MAX, R_MAX)))
    assert max(abs(r) for r in r_unclipped) > R_MAX, (
        "test not probative: unclipped r never exceeds r_max")
    assert max(abs(r) for r in r_clipped) <= R_MAX + 1e-7, (
        f"clipped r exceeds r_max: {r_clipped}")
    # And the clip reaches the circuit: with saturating w_s, output with
    # r_max differs from output without it.
    g_noclip = build(mod, encoding=EncodingSpec('interleaved_squeeze'))
    g_noclip.weights.assign(weights)
    g_noclip.w_s.assign(big_ws)
    p_clip = g_clip.generate_distribution_2d(z_sat, xvec, xvec).numpy()
    p_noclip = g_noclip.generate_distribution_2d(z_sat, xvec, xvec).numpy()
    dclip = float(np.max(np.abs(p_clip - p_noclip)))
    assert dclip > 1e-6, "r_max clip has no effect on the circuit output"
    print(f"  ok: r_max clip (max|r| unclipped "
          f"{max(abs(r) for r in r_unclipped):.2f} -> clipped <= {R_MAX}; "
          f"output effect {dclip:.2e})")


def generate(force=False):
    # Structural test — no golden data to generate; run the checks instead.
    test()


if __name__ == '__main__':
    main_dispatch(generate, test, 'test_interleaved_equiv')
