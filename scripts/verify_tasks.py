"""Verification for the 2026-07-09 edits to train_2d_qgan.py (Tasks 1-3).

Disposable -- delete after the checks pass. Runs two tiny end-to-end
trainings (output goes to <system temp>/qnncv_verify_*, not ./logs):

  1. GAN mode with strong-critic knobs (n_critic=5, d_dropout=0, d_lr=g_lr):
     - D receives non-zero gradients every epoch (critic tape refactor)
     - D outputs change across epochs (D actually updates)
     - new validation metrics (nearest W1, diversity, energy distances)
       are recorded and finite
     - best_weights.npy is persisted
  2. Path A mode (supervised_weight=1.0, warmup >= epochs):
     - PATH A MODE banner prints, discriminator disabled
     - D gradient norms and D loss are identically zero
"""
import io
import os
import sys
import tempfile
from contextlib import redirect_stdout
import importlib.util
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = tempfile.gettempdir()

spec = importlib.util.spec_from_file_location(
    't2q', os.path.join(REPO, 'train_2d_qgan.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

common = dict(family_name='gaussian', n_train=8, n_val=6, n_total_modes=2,
              n_layers=2, cutoff_dim=6, epochs=4, val_every=2, plot_every=999)

# --- 1. GAN mode, strong critic knobs ---
# batch_size=1 / latent_scale=1.0 (and the Path-A values below) are the former
# signature defaults, passed explicitly since the default unification so this
# script's behavior is unchanged.
print('=== GAN-mode run (n_critic=5, d_dropout=0, d_lr=g_lr) ===')
gen, hist = mod.train_2d_qgan(
    g_lr=0.005, d_lr=0.005, n_critic=5, d_dropout=0.0,
    batch_size=1, latent_scale=1.0,
    supervised_weight=0.0, log_dir=os.path.join(OUT, 'qnncv_verify_gan'),
    **common)

assert len(hist['d_grad_norm']) == 4, hist['d_grad_norm']
assert min(hist['d_grad_norm']) > 0, \
    f"D gradients zero after tape refactor: {hist['d_grad_norm']}"
assert any(x != 0.0 for x in hist['d_loss']), 'D loss identically zero'
assert any(x != hist['d_real_score'][0] for x in hist['d_real_score']), \
    'D outputs never changed -- D not updating'
for key in ('val_nearest_w1', 'val_diversity', 'val_canonical_ed',
            'val_nearest_ed'):
    assert key in hist and len(hist[key]) == 2, (key, hist.get(key))
    assert all(np.isfinite(hist[key])), (key, hist[key])
assert os.path.exists(os.path.join(OUT, 'qnncv_verify_gan',
                                   'best_weights.npy'))
print('PASS: D grad norm per epoch:',
      [round(x, 4) for x in hist['d_grad_norm']])
print('PASS: D real-score trajectory:',
      [round(x, 4) for x in hist['d_real_score']])
print('PASS: val metrics:',
      {k: [round(float(v), 4) for v in hist[k]]
       for k in ('val_nearest_w1', 'val_diversity', 'val_nearest_ed')})
print('PASS: best_weights.npy persisted')

# --- 2. Path A mode ---
print('\n=== Path A run (supervised_weight=1.0, warmup >= epochs) ===')
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
        d_lr=0.0002, n_critic=1, batch_size=1, d_dropout=0.3,
        latent_scale=1.0,
        log_dir=os.path.join(OUT, 'qnncv_verify_patha'), **common)
out = buf.getvalue()

assert 'PATH A MODE' in out, 'Path A banner missing'
assert 'discriminator DISABLED' in out, 'Path A D-disabled line missing'
assert all(x == 0.0 for x in hist2['d_grad_norm']), 'D trained in Path A!'
assert all(x == 0.0 for x in hist2['d_loss']), 'D loss nonzero in Path A!'
print('PASS: Path A banner printed, discriminator never trained')

print('\nALL CHECKS PASSED')
