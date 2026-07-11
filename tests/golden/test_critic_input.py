"""
Golden: critic input pipeline — to_critic_input (peak normalization, the
load-bearing WGAN fix), build_blur_kernel, critic_blur, and
compute_gradient_penalty.

Covers the three tensor paths that feed the critic (real, fake, and the GP
interpolants — GP is computed on the same normalized tensors D sees) via
fixed seeded inputs.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (load_t2q, seed_all, save_golden_npz, load_golden_npz,
                      assert_exact, assert_close, main_dispatch)

GOLDEN = 'critic_input.npz'


def fixed_densities(shape, seed):
    import numpy as np
    rs = np.random.RandomState(seed)
    d = rs.uniform(0.0, 1.0, size=shape).astype(np.float32)
    # normalize to sum 1 per sample like real densities
    axes = tuple(range(len(shape) - 2, len(shape)))
    d = d / d.sum(axis=axes, keepdims=True)
    return d


def compute():
    import numpy as np
    import tensorflow as tf
    mod = load_t2q()
    out = {}

    # --- to_critic_input: single (gx, gy) and batched (B, gx, gy) ---
    x2 = fixed_densities((8, 8), 11)
    x3 = fixed_densities((3, 8, 8), 12)
    out['tci_single_in'] = x2
    out['tci_batch_in'] = x3
    out['tci_single_out'] = mod.to_critic_input(tf.constant(x2)).numpy()
    out['tci_batch_out'] = mod.to_critic_input(tf.constant(x3)).numpy()

    # --- blur kernel ---
    k = mod.build_blur_kernel(0.7)
    out['blur_kernel_0p7'] = k.numpy()
    assert mod.build_blur_kernel(0) is None
    assert mod.build_blur_kernel(-1.0) is None
    assert mod.build_blur_kernel(None) is None

    # --- critic_blur: single and batched, plus None-kernel no-op ---
    d40 = fixed_densities((40, 40), 13)
    b40 = fixed_densities((2, 40, 40), 14)
    out['blur_single_in'] = d40
    out['blur_batch_in'] = b40
    out['blur_single_out'] = mod.critic_blur(tf.constant(d40), k).numpy()
    out['blur_batch_out'] = mod.critic_blur(tf.constant(b40), k).numpy()
    noop = mod.critic_blur(tf.constant(d40), None).numpy()
    assert np.array_equal(noop, d40), "critic_blur(None) must be a no-op"

    # --- gradient penalty (consumes tf.random.uniform + D weight init) ---
    seed_all(0)
    disc = mod.Discriminator2D(dropout_rate=0.0)
    real = tf.constant(fixed_densities((4, 40, 40), 15))
    fake = tf.constant(fixed_densities((4, 40, 40), 16))
    real_n = mod.to_critic_input(real)
    fake_n = mod.to_critic_input(fake)
    gp = mod.compute_gradient_penalty(disc, real_n, fake_n)
    out['gp_value'] = np.array(float(gp))

    return out


def generate(force=False):
    save_golden_npz(GOLDEN, compute(), force=force)


def test():
    golden = load_golden_npz(GOLDEN)
    current = compute()
    exact_keys = {'tci_single_in', 'tci_batch_in', 'blur_kernel_0p7',
                  'blur_single_in', 'blur_batch_in'}
    for key in golden.files:
        if key in exact_keys:
            assert_exact(current[key], golden[key], label=key)
        else:
            assert_close(current[key], golden[key], label=key)
        print(f"  ok: {key}")


if __name__ == '__main__':
    main_dispatch(generate, test, 'test_critic_input')
