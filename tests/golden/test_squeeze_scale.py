"""
Structural suite for the z-squeeze gain modes of 'interleaved_squeeze'
(no golden data):

1. FIXED scale (the default mode): EncodingSpec(squeeze_scale=s) fills w_s
   with s, w_s is NOT trainable (one trainable variable, no gradient is
   produced for w_s), and the batched output equals a trainable-gain
   generator whose w_s was assigned to s — exact (same seed, same weights,
   same z: the bound tensors are identical).
2. Channel on/off: with s != 0 the output differs from the
   input_displacement baseline for generic z; with the default s == 0 it is
   exactly the baseline (the control arm).
3. TRAINABLE gain with nonzero init: train_squeeze_scale=True and
   squeeze_scale=s gives w_s == s, trainable, with a finite nonzero
   d(loss)/d(w_s).
4. EncodingSpec validation + get_config: input_displacement rejects a
   nonzero squeeze_scale or train_squeeze_scale; get_config reports both.
5. CLI surface: --squeeze-scale (float, default 0.0) and
   --train-squeeze-scale (store_true, default False) exist, and
   build_hp_suffix tags them 'ss<val>' / 'sstrain' (absent at defaults).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import load_t2q, seed_all, assert_exact, main_dispatch

MODES = 2
LAYERS = 2
CUTOFF = 6
BATCH = 4
SCALE = 0.3


def build(mod, encoding=None):
    seed_all(0)
    return mod.CVQGANGenerator(n_modes=MODES, n_layers=LAYERS,
                               cutoff_dim=CUTOFF, latent_scale=1.0,
                               batch_size=BATCH, encoding=encoding)


def test():
    import numpy as np
    import tensorflow as tf
    mod = load_t2q()
    EncodingSpec = mod.EncodingSpec

    g_base = build(mod)
    weights = g_base.weights.numpy()
    z = np.random.RandomState(7).standard_normal(
        (BATCH, g_base.latent_dim)).astype(np.float32) * 0.5
    zt = tf.constant(z)
    xvec = np.linspace(-3.0, 3.0, 40)
    full_s = np.full((LAYERS, MODES), SCALE, dtype=np.float32)

    def probs(g):
        g.weights.assign(weights)
        p, _ = g.generate_batch(zt, xvec, xvec, return_ket_norm=True)
        return p.numpy()

    def loss_and_grads(g):
        g.weights.assign(weights)
        with tf.GradientTape(persistent=True) as tape:
            p, kn = g.generate_batch(zt, xvec, xvec, return_ket_norm=True)
            loss = (tf.reduce_mean(tf.square(p))
                    + tf.reduce_mean(tf.square(1.0 - kn)))
        grads = tape.gradient(loss, g.trainable_variables)
        g_ws = tape.gradient(loss, g.w_s)
        del tape
        return grads, g_ws

    p_base = probs(g_base)

    # --- (1) fixed scale: value, non-trainability, == assigned trainable ---
    g_fix = build(mod, encoding=EncodingSpec('interleaved_squeeze',
                                             squeeze_scale=SCALE))
    assert_exact(g_fix.w_s.numpy(), full_s, label='fixed w_s value')
    assert not g_fix.w_s.trainable, "fixed-scale w_s must not be trainable"
    assert len(g_fix.trainable_variables) == 1, (
        f"fixed-scale generator exposes {len(g_fix.trainable_variables)} "
        f"trainable variables, expected 1")
    grads, g_ws = loss_and_grads(g_fix)
    assert len(grads) == 1 and grads[0] is not None, "weights grad missing"
    assert g_ws is None, "a FIXED gain must receive no gradient"
    g_tr = build(mod, encoding=EncodingSpec('interleaved_squeeze',
                                            train_squeeze_scale=True))
    g_tr.w_s.assign(full_s)
    p_fix = probs(g_fix)
    assert_exact(p_fix, probs(g_tr),
                 label='fixed s == trainable gain assigned to s')
    print(f"  ok: fixed scale s={SCALE}: w_s filled, non-trainable, "
          f"output == trainable gain at s")

    # --- (2) channel on at s != 0, off (== baseline) at the default --------
    diff_on = float(np.max(np.abs(p_fix - p_base)))
    assert diff_on > 1e-6, (
        f"fixed scale {SCALE} left the output identical to the baseline")
    g_off = build(mod, encoding=EncodingSpec('interleaved_squeeze'))
    assert not g_off.w_s.trainable, "default gain must be fixed"
    assert float(np.max(np.abs(g_off.w_s.numpy()))) == 0.0, "default s != 0"
    assert_exact(probs(g_off), p_base, label='default (s=0, fixed) == baseline')
    print(f"  ok: channel on at s={SCALE} (max |dP| = {diff_on:.2e}); "
          f"default s=0 is exactly the baseline")

    # --- (3) trainable gain with nonzero init -----------------------------
    g_tr2 = build(mod, encoding=EncodingSpec('interleaved_squeeze',
                                             squeeze_scale=SCALE,
                                             train_squeeze_scale=True))
    assert g_tr2.w_s.trainable, "train_squeeze_scale=True must be trainable"
    assert_exact(g_tr2.w_s.numpy(), full_s, label='trainable init value')
    assert len(g_tr2.trainable_variables) == 2
    grads, g_ws = loss_and_grads(g_tr2)
    assert len(grads) == 2 and grads[1] is not None, "w_s grad missing"
    gn = float(tf.linalg.global_norm([grads[1]]))
    assert np.isfinite(gn) and gn > 0.0, f"bad d(loss)/d(w_s): {gn}"
    print(f"  ok: trainable gain init at s={SCALE} (|grad w_s| = {gn:.2e})")

    # --- (4) validation + get_config ---------------------------------------
    for kw in ({'squeeze_scale': 0.1}, {'train_squeeze_scale': True}):
        try:
            EncodingSpec('input_displacement', **kw)
            raise AssertionError(f"input_displacement accepted {kw}")
        except ValueError:
            pass
    cfg = g_tr2.get_config()
    assert cfg['squeeze_scale'] == SCALE and cfg['train_squeeze_scale'] is True
    cfg = g_fix.get_config()
    assert cfg['squeeze_scale'] == SCALE and cfg['train_squeeze_scale'] is False
    print("  ok: EncodingSpec validation and get_config")

    # --- (5) CLI surface ---------------------------------------------------
    parser = mod.build_parser()
    assert parser.get_default('squeeze_scale') == 0.0
    assert parser.get_default('train_squeeze_scale') is False
    args = parser.parse_args(['--family', 'ring', '--encoding',
                              'interleaved_squeeze', '--squeeze-scale', '0.2',
                              '--train-squeeze-scale'])
    args.seed = mod.resolve_seed(args.seed)
    parts = mod.build_hp_suffix(args, parser).split('_')
    assert 'encIS' in parts and 'ss0.2' in parts and 'sstrain' in parts, parts
    args = parser.parse_args(['--family', 'ring', '--encoding',
                              'interleaved_squeeze'])
    args.seed = mod.resolve_seed(args.seed)
    suffix = mod.build_hp_suffix(args, parser)
    assert 'ss' not in suffix.split('_') and 'sstrain' not in suffix, suffix
    assert not any(p.startswith('ss') for p in suffix.split('_')), suffix
    print("  ok: CLI defaults (0.0 / False) and folder tags ss<val> / sstrain")


def generate(force=False):
    # Structural test — no golden data to generate; run the checks instead.
    test()


if __name__ == '__main__':
    main_dispatch(generate, test, 'test_squeeze_scale')
