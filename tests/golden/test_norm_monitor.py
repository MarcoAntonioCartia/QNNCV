"""
Structural suite for the per-sample ket-norm monitor (no golden data):

Two tiny in-process interleaved training runs (gaussian, 2 layers, cutoff 6,
2 epochs) into scratch log dirs:

- norm_floor=1.5 (unreachable: ket norms are <= 1): the stderr warning MUST
  fire every generator step and norm_monitor.npz must record warn_count ==
  batch_size for every epoch, plus the offending z.
- norm_floor=0.0: the warning must stay silent and warn_count == 0.
- Both runs: norm_monitor.npz next to history.npz with per-epoch arrays
  satisfying ket_norm_min <= ket_norm_mean <= ket_norm_max, and the
  history.npz key set UNCHANGED (monitor keys live in the separate file).
"""

import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import load_t2q, scratch_dir, main_dispatch

EPOCHS = 2
BATCH = 4

# history.npz keys frozen by the E2E goldens (+ the seed scalar).
HISTORY_KEYS = {
    'g_loss', 'd_loss', 'wasserstein', 'val_wasserstein', 'val_nearest_w1',
    'val_diversity', 'val_canonical_ed', 'val_nearest_ed', 'g_grad_norm',
    'd_grad_norm', 'supervised_loss', 'adversarial_loss', 'gp_value',
    'gp_weight_current', 'd_real_score', 'd_fake_score', 'ket_norm',
    'supervised_weight_current', 'instance_noise_current', 'seed',
}


def tiny_run(mod, norm_floor):
    """One tiny interleaved run; returns (log_dir, stderr_text)."""
    log_dir = scratch_dir(f'normmon_{norm_floor}')
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        mod.train_2d_qgan(
            family_name='gaussian', n_train=8, n_val=4,
            n_total_modes=2, n_layers=2, cutoff_dim=6,
            epochs=EPOCHS, d_lr=0.0002, n_critic=1, batch_size=BATCH,
            d_dropout=0.0, latent_scale=1.0,
            plot_every=1000, val_every=1, seed=0, log_dir=log_dir,
            encoding='interleaved_squeeze', norm_floor=norm_floor,
        )
    return log_dir, err.getvalue()


def test():
    import numpy as np
    mod = load_t2q()

    # --- floor 1.5: warning must fire, warn_count == batch each epoch ------
    log_dir, stderr_text = tiny_run(mod, norm_floor=1.5)
    assert 'WARNING' in stderr_text and 'ket_norm < 1.5' in stderr_text, (
        f"norm-floor warning did not fire on stderr:\n{stderr_text[-500:]}")
    assert 'worst z =' in stderr_text, "warning does not report offending z"
    mon_path = os.path.join(log_dir, 'norm_monitor.npz')
    assert os.path.exists(mon_path), "norm_monitor.npz not written"
    mon = np.load(mon_path)
    assert set(mon.files) == {'ket_norm_min', 'ket_norm_mean', 'ket_norm_max',
                              'warn_count', 'worst_z', 'norm_floor'}, (
        f"unexpected norm_monitor keys: {sorted(mon.files)}")
    assert mon['warn_count'].shape == (EPOCHS,)
    assert np.all(mon['warn_count'] == BATCH), (
        f"expected warn_count == {BATCH} every epoch at floor 1.5, got "
        f"{mon['warn_count']}")
    assert mon['worst_z'].shape == (EPOCHS, 4), (
        f"worst_z shape {mon['worst_z'].shape}")
    assert float(mon['norm_floor']) == 1.5
    assert np.all(mon['ket_norm_min'] <= mon['ket_norm_mean'] + 1e-12)
    assert np.all(mon['ket_norm_mean'] <= mon['ket_norm_max'] + 1e-12)
    print("  ok: floor=1.5 fires stderr warning; warn_count == batch; "
          "min <= mean <= max; worst_z recorded")

    # history.npz key set unchanged (monitor keys live in the separate file)
    hist = np.load(os.path.join(log_dir, 'history.npz'))
    assert set(hist.files) == HISTORY_KEYS, (
        f"history.npz keys drifted: {sorted(set(hist.files) ^ HISTORY_KEYS)}")
    print("  ok: history.npz key set unchanged")

    # --- floor 0.0: silent, warn_count == 0 --------------------------------
    log_dir, stderr_text = tiny_run(mod, norm_floor=0.0)
    assert 'WARNING' not in stderr_text, (
        f"warning fired at floor 0.0:\n{stderr_text[-500:]}")
    mon = np.load(os.path.join(log_dir, 'norm_monitor.npz'))
    assert np.all(mon['warn_count'] == 0), (
        f"warn_count nonzero at floor 0.0: {mon['warn_count']}")
    print("  ok: floor=0.0 silent, warn_count == 0")


def generate(force=False):
    # Structural test — no golden data to generate; run the checks instead.
    test()


if __name__ == '__main__':
    main_dispatch(generate, test, 'test_norm_monitor')
