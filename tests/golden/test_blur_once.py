"""
Golden + invariant: blur is applied EXACTLY ONCE per critic tensor path.

critic_blur is the only conv2d user in the whole pipeline (the critic is
Dense-only), and it resolves tf.nn.conv2d by attribute lookup at call time,
so counting tf.nn.conv2d calls is call-site independent — the check survives
every extraction phase no matter which module ends up calling critic_blur.

Expected count for a run with critic_blur_sigma > 0:
    epochs * (2 * n_critic + 1)
= one blur for the real batch + one for the fake batch per critic step,
plus one for the generator's adversarial pass per epoch. The instance-noise
re-normalization inside the critic tape must NOT blur again, and validate()
/ plotting never blur.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (load_t2q, save_golden_npz, load_golden_npz,
                      scratch_dir, main_dispatch)

GOLDEN = 'blur_once.npz'

EPOCHS = 2
N_CRITIC = 2
TRAIN_KWARGS = dict(
    seed=0, family_name='gaussian', n_train=8, n_val=6, n_total_modes=2,
    n_layers=2, cutoff_dim=6, epochs=EPOCHS, val_every=2, plot_every=999,
    batch_size=4, n_critic=N_CRITIC, d_dropout=0.0, critic_blur_sigma=0.7,
)


def compute():
    import numpy as np
    import tensorflow as tf
    mod = load_t2q()

    calls = {'n': 0}
    _orig = tf.nn.conv2d

    def _counting(*a, **k):
        calls['n'] += 1
        return _orig(*a, **k)

    tf.nn.conv2d = _counting
    try:
        gen, hist = mod.train_2d_qgan(log_dir=scratch_dir('bluronce'),
                                      **TRAIN_KWARGS)
    finally:
        tf.nn.conv2d = _orig

    return {
        'conv2d_calls': np.array(calls['n'], dtype=np.int64),
        'g_loss': np.array(hist['g_loss']),
        'd_loss': np.array(hist['d_loss']),
    }


def generate(force=False):
    out = compute()
    expected = EPOCHS * (2 * N_CRITIC + 1)
    assert int(out['conv2d_calls']) == expected, (
        f"golden generation sanity: conv2d called {int(out['conv2d_calls'])} "
        f"times, expected {expected} — blur-once invariant broken in the "
        f"CURRENT code?")
    save_golden_npz(GOLDEN, out, force=force)


def test():
    from _harness import assert_close
    golden = load_golden_npz(GOLDEN)
    current = compute()
    expected = EPOCHS * (2 * N_CRITIC + 1)
    n = int(current['conv2d_calls'])
    assert n == int(golden['conv2d_calls']), (
        f"conv2d call count drifted: {n} != golden "
        f"{int(golden['conv2d_calls'])}")
    assert n == expected, (
        f"blur-once invariant violated: {n} conv2d calls, expected {expected}")
    print(f"  ok: conv2d call count == {n} == epochs*(2*n_critic+1)")
    assert_close(current['g_loss'], golden['g_loss'], label='g_loss')
    assert_close(current['d_loss'], golden['d_loss'], label='d_loss')
    print("  ok: g_loss/d_loss arrays match golden")


if __name__ == '__main__':
    main_dispatch(generate, test, 'test_blur_once')
