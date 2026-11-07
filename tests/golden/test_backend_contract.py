"""
Contract suite for src/backends (Phase 0 of docs/benchmark_plan.md).
No golden data: the contract is structural, plus one exactness check that
pins the CV adapter to the raw CVQGANGenerator.

For every generator in registry.toy_factories():
  1. shapes/dtype: (B, G, G) float32, finite, non-negative; `aux` accepted
     by `regularizer`, which returns a finite scalar.
  2. generate_distribution_2d(z[0]) equals generate_batch(z)[0] (sequential
     vs batched engines agree to 1e-4, as verify_batching pins).
  3. gradients: a loss on the output reaches every trainable variable.
  4. state_dict round-trip exact; after perturbing weights and reloading,
     the output is restored exactly.
  5. describe(): required keys with sane values (n_params == size of
     trainable_variables).
For every critic: scores (B, 1), gradients reach every variable, describe().
CV adapter exactness: adapter output == raw CVQGANGenerator output for the
same seed/weights/z (bit-identical), and the registry must list every
backend in toy_factories (and vice versa).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import load_t2q, seed_all, assert_exact, assert_close, main_dispatch

BATCH = 4
GRID = 40
REQUIRED_GEN_KEYS = {'backend', 'n_params', 'hilbert_dim', 'latent_dim'}
REQUIRED_CRITIC_KEYS = {'backend', 'n_params', 'hilbert_dim'}


def check_generator(name, factory):
    import numpy as np
    import tensorflow as tf
    seed_all(0)
    g = factory()
    xvec = np.linspace(-3.0, 3.0, GRID)
    z = tf.constant(np.random.RandomState(7).standard_normal(
        (BATCH, g.latent_dim)).astype(np.float32) * 0.5)

    # (1) shapes, dtype, regularizer
    probs, aux = g.generate_batch(z, xvec, xvec)
    p = probs.numpy()
    assert p.shape == (BATCH, GRID, GRID), f"{name}: shape {p.shape}"
    assert p.dtype == np.float32, f"{name}: dtype {p.dtype}"
    assert np.isfinite(p).all() and (p >= 0).all(), f"{name}: bad values"
    reg = float(g.regularizer(aux))
    assert np.isfinite(reg), f"{name}: regularizer not finite"

    # (2) single-vector path agrees with the batched path
    single = g.generate_distribution_2d(z[0], xvec, xvec).numpy()
    assert_close(single, p[0], label=f'{name}: single vs batch', rtol=1e-4, atol=1e-4)

    # (3) gradients reach every trainable variable
    with tf.GradientTape() as tape:
        pr, ax = g.generate_batch(z, xvec, xvec)
        loss = tf.reduce_mean(tf.square(pr)) + g.regularizer(ax)
    grads = tape.gradient(loss, g.trainable_variables)
    assert len(grads) == len(g.trainable_variables) > 0
    for v, gr in zip(g.trainable_variables, grads):
        assert gr is not None, f"{name}: no gradient for {v.name}"
        assert np.isfinite(gr.numpy()).all(), f"{name}: non-finite grad {v.name}"

    # (4) state round-trip
    state = g.state_dict()
    before = p.copy()
    for v in g.trainable_variables:
        v.assign(v + 0.05)
    assert np.max(np.abs(g.generate_batch(z, xvec, xvec)[0].numpy() - before)) > 0, (
        f"{name}: perturbing the variables did not change the output")
    g.load_state_dict(state)
    assert_exact(g.generate_batch(z, xvec, xvec)[0].numpy(), before,
                 label=f'{name}: output after state_dict reload')
    state2 = g.state_dict()
    for k in state:
        if isinstance(state[k], np.ndarray):
            assert_exact(state2[k], state[k], label=f'{name}: state[{k}]')

    # (5) describe
    d = g.describe()
    missing = REQUIRED_GEN_KEYS - set(d)
    assert not missing, f"{name}: describe() lacks {missing}"
    assert d['n_params'] == g.count_params(g.trainable_variables), d
    assert d['latent_dim'] == g.latent_dim
    print(f"  ok: generator {name:16s} params={d['n_params']} "
          f"hilbert={d['hilbert_dim']} latent={d['latent_dim']}")


def check_critic(name, factory):
    import numpy as np
    import tensorflow as tf
    seed_all(0)
    c = factory()
    x = tf.constant(np.random.RandomState(3).rand(BATCH, GRID, GRID, 1).astype(np.float32))
    with tf.GradientTape() as tape:
        s = c(x, training=False)
        loss = tf.reduce_mean(s)
    assert tuple(s.shape) == (BATCH, 1), f"{name}: score shape {s.shape}"
    grads = tape.gradient(loss, c.trainable_variables)
    assert len(grads) == len(c.trainable_variables) > 0
    assert all(gr is not None for gr in grads), f"{name}: missing gradient"
    d = c.describe()
    missing = REQUIRED_CRITIC_KEYS - set(d)
    assert not missing, f"{name}: describe() lacks {missing}"
    print(f"  ok: critic    {name:16s} params={d['n_params']}")


def test():
    import numpy as np
    import tensorflow as tf
    mod = load_t2q()  # repo root on sys.path + TF/SF prologue
    from src.backends import registry
    gens, critics = registry.toy_factories(batch_size=BATCH)

    # registry <-> toy factories completeness (base names, before any '/')
    reg_g = set(registry.GENERATOR_REGISTRY)
    toy_g = {k.split('/')[0] for k in gens}
    assert reg_g == toy_g, f"generator registry {reg_g} != toy factories {toy_g}"
    assert set(registry.CRITIC_REGISTRY) == set(critics), "critic registry != toy factories"

    for name, factory in gens.items():
        check_generator(name, factory)
    for name, factory in critics.items():
        check_critic(name, factory)

    # CV adapter exactness against the raw generator
    xvec = np.linspace(-3.0, 3.0, GRID)
    seed_all(0)
    raw = mod.CVQGANGenerator(n_modes=2, n_layers=2, cutoff_dim=6,
                              latent_scale=1.0, batch_size=BATCH)
    seed_all(0)
    adapted = gens['cv']()
    z = tf.constant(np.random.RandomState(7).standard_normal(
        (BATCH, raw.latent_dim)).astype(np.float32) * 0.5)
    p_raw, k_raw = raw.generate_batch(z, xvec, xvec, return_ket_norm=True)
    p_ad, k_ad = adapted.generate_batch(z, xvec, xvec)
    assert_exact(p_ad.numpy(), p_raw.numpy(), label='cv adapter probs')
    assert_exact(k_ad.numpy(), k_raw.numpy(), label='cv adapter ket norms')
    # the regularizer is the trainer's ket penalty formula
    assert_exact(float(adapted.regularizer(k_ad)),
                 float(tf.reduce_mean(tf.square(1.0 - k_raw))),
                 label='cv regularizer == ket penalty')
    print("  ok: cv adapter bit-identical to CVQGANGenerator; regularizer == ket penalty")


def generate(force=False):
    test()


if __name__ == '__main__':
    main_dispatch(generate, test, 'test_backend_contract')
