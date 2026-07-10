"""Verification for the critic-input peak-scaling + SF TF batching changes.

Disposable -- delete after the checks pass. Training output goes to
<system temp>/qnncv_verify_*, not ./logs.

Checks, in order:
  1. Batched vs sequential generation EQUIVALENCE (2 modes and 3 modes,
     same weights, same z) -- the batched measurement path (einsums,
     partial trace, per-element normalization) is numerically correct.
  2. Single-z calls on a batched engine match the unbatched engine
     (the broadcast-and-slice path used by plots and validate()).
  3. Gradients flow through the batched circuit AND peak-normalization
     to the generator weights -- the TF tape is NOT broken.
  4. to_critic_input: peak=1 per sample, differentiable.
  5. Tiny GAN-mode run (batch_size=4, n_critic=2): D and G gradient
     norms non-zero every epoch, D outputs move, metrics finite.
  6. Tiny Path A run (default batch_size=1): banner prints, D never
     trains -- legacy behavior preserved.
"""
import io
import os
import sys
import tempfile
from contextlib import redirect_stdout
import importlib.util
import numpy as np
import tensorflow as tf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = tempfile.gettempdir()

spec = importlib.util.spec_from_file_location(
    't2q', os.path.join(REPO, 'train_2d_qgan.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

xvec = np.linspace(-3, 3, 40)

# --- 1 & 2. batched == sequential ---
for n_modes in (2, 3):
    print(f'=== Equivalence check, {n_modes} modes ===')
    g_seq = mod.CVQGANGenerator(n_modes=n_modes, n_layers=2, cutoff_dim=5)
    g_bat = mod.CVQGANGenerator(n_modes=n_modes, n_layers=2, cutoff_dim=5,
                                batch_size=4)
    g_bat.weights.assign(g_seq.weights)

    z = tf.random.normal([4, g_seq.latent_dim])
    p_bat, kn_bat = g_bat.generate_batch(z, xvec, xvec, return_ket_norm=True)

    p_seq, kn_seq = [], []
    for i in range(4):
        p, kn = g_seq.generate_distribution_2d(z[i], xvec, xvec,
                                               return_ket_norm=True)
        p_seq.append(p.numpy())
        kn_seq.append(float(kn))
    p_seq = np.stack(p_seq)

    err = float(np.max(np.abs(p_bat.numpy() - p_seq)))
    kerr = float(np.max(np.abs(kn_bat.numpy() - np.array(kn_seq))))
    assert err < 1e-4, f'{n_modes}-mode batched != sequential (max err {err})'
    assert kerr < 1e-4, f'{n_modes}-mode ket norms differ (max err {kerr})'
    print(f'PASS: batched == sequential (max prob err {err:.2e}, '
          f'ket-norm err {kerr:.2e})')

    # Single-z on the batched engine (broadcast + slice path)
    p_single_bat = g_bat.generate_distribution_2d(z[0], xvec, xvec).numpy()
    err1 = float(np.max(np.abs(p_single_bat - p_seq[0])))
    assert err1 < 1e-4, f'single-z on batched engine differs (max err {err1})'
    print(f'PASS: single-z on batched engine matches (max err {err1:.2e})')

# --- 3. gradient through batched circuit + peak-normalization ---
print('=== Gradient-flow check (batched engine) ===')
with tf.GradientTape() as tape:
    p, kn = g_bat.generate_batch(  # g_bat is the 3-mode generator here
        tf.random.normal([4, g_bat.latent_dim]), xvec, xvec,
        return_ket_norm=True)
    # Mimic the G loss path: critic input scaling + ket penalty
    loss = (tf.reduce_mean(tf.square(mod.to_critic_input(p)))
            + tf.reduce_mean(tf.square(1.0 - kn)))
grads = tape.gradient(loss, g_bat.trainable_variables)
assert grads[0] is not None, 'no gradient through batched circuit!'
gnorm = float(tf.linalg.global_norm(grads))
assert np.isfinite(gnorm) and gnorm > 0, f'bad G grad norm: {gnorm}'
print(f'PASS: gradient flows through batched circuit (norm {gnorm:.4f})')

# --- 4. to_critic_input ---
x = tf.constant(np.random.rand(3, 8, 8).astype('float32')) * 0.01
y = mod.to_critic_input(x).numpy()
assert np.allclose(y.max(axis=(1, 2)), 1.0, atol=1e-5), 'peak != 1'
v = tf.Variable(x)
with tf.GradientTape() as t2:
    out = tf.reduce_sum(tf.square(mod.to_critic_input(v)))
g = t2.gradient(out, v)
assert g is not None and float(tf.reduce_max(tf.abs(g))) > 0
print('PASS: to_critic_input scales to peak=1 and is differentiable')

# --- 5. GAN-mode training run, batch_size=4 ---
print('=== GAN-mode run (batch_size=4, n_critic=2, d_dropout=0) ===')
common = dict(family_name='gaussian', n_train=8, n_val=6, n_total_modes=2,
              n_layers=2, cutoff_dim=6, epochs=4, val_every=2, plot_every=999)
gen, hist = mod.train_2d_qgan(
    g_lr=0.005, d_lr=0.005, n_critic=2, d_dropout=0.0,
    supervised_weight=0.0, batch_size=4,
    log_dir=os.path.join(OUT, 'qnncv_verify_batchgan'), **common)

assert min(hist['d_grad_norm']) > 0, \
    f"D gradients zero: {hist['d_grad_norm']}"
assert min(hist['g_grad_norm']) > 0, \
    f"G gradients zero -- tape broken: {hist['g_grad_norm']}"
assert any(x != hist['d_real_score'][0] for x in hist['d_real_score']), \
    'D outputs never changed'
for key in ('val_nearest_w1', 'val_diversity', 'val_nearest_ed'):
    assert all(np.isfinite(hist[key])), (key, hist[key])
print('PASS: D grad norms:', [round(x, 4) for x in hist['d_grad_norm']])
print('PASS: G grad norms:', [round(x, 4) for x in hist['g_grad_norm']])
print('PASS: D(r) trajectory:', [round(x, 4) for x in hist['d_real_score']])
print('PASS: D(f) trajectory:', [round(x, 4) for x in hist['d_fake_score']])

# --- 6. Path A run, default batch_size=1 (legacy path) ---
print('=== Path A run (batch_size=1 default) ===')
buf = io.StringIO()


class Tee:
    def write(self, s):
        buf.write(s)
        sys.__stdout__.write(s)

    def flush(self):
        sys.__stdout__.flush()


with redirect_stdout(Tee()):
    gen2, hist2 = mod.train_2d_qgan(
        supervised_weight=1.0, supervised_warmup=4,
        log_dir=os.path.join(OUT, 'qnncv_verify_batchpatha'), **common)
out = buf.getvalue()

assert 'PATH A MODE' in out, 'Path A banner missing'
assert all(x == 0.0 for x in hist2['d_grad_norm']), 'D trained in Path A!'
assert min(hist2['g_grad_norm']) > 0, 'G gradients zero in Path A!'
print('PASS: Path A banner printed, D never trained, G still learns')

print('\nALL CHECKS PASSED')
