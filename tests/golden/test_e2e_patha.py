"""
End-to-end golden: Path A (pure_supervised) mode, tiny config, in-process
via e2e_patha_child.py in a fresh subprocess.

Asserts the auto-detection contract (supervised_weight >= 1.0 AND
supervised_warmup >= epochs => discriminator disabled + banner), then
freezes history arrays, best_weights, and a fixed-z generated density.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (HERE, run_subprocess, scratch_dir, save_golden_dir,
                      save_golden_text, load_golden_text, golden_path,
                      assert_close, main_dispatch)

GOLDEN_DIR = 'e2e_patha'
CHILD = os.path.join(HERE, 'e2e_patha_child.py')


def run_child():
    out_dir = scratch_dir('e2e_patha')
    proc = run_subprocess([sys.executable, CHILD, out_dir])
    console = proc.stdout + '\n--- STDERR ---\n' + proc.stderr
    assert proc.returncode == 0, f"Path A child failed:\n{console[-4000:]}"
    assert 'CHILD DONE' in proc.stdout
    return out_dir, proc.stdout


def check_patha_contract(stdout, history):
    import numpy as np
    assert 'PATH A MODE' in stdout, "Path A banner missing"
    assert 'discriminator DISABLED' in stdout, "D-disabled banner missing"
    assert np.all(np.asarray(history['d_loss']) == 0.0), (
        "d_loss not identically zero in Path A")
    assert np.all(np.asarray(history['d_grad_norm']) == 0.0), (
        "d_grad_norm not identically zero in Path A")


def generate(force=False):
    import numpy as np
    out_dir, stdout = run_child()
    history = np.load(os.path.join(out_dir, 'history.npz'))
    check_patha_contract(stdout, history)
    save_golden_dir(GOLDEN_DIR, {
        'history.npz': os.path.join(out_dir, 'history.npz'),
        'best_weights.npy': os.path.join(out_dir, 'best_weights.npy'),
        'fixed_z_density.npy': os.path.join(out_dir, 'fixed_z_density.npy'),
    }, force=force)
    save_golden_text(os.path.join(GOLDEN_DIR, 'stdout.txt'), stdout,
                     force=force)


def test():
    import numpy as np
    out_dir, stdout = run_child()
    cur_h = np.load(os.path.join(out_dir, 'history.npz'))
    check_patha_contract(stdout, cur_h)
    print("  ok: Path A banners + D identically zero")

    gold_h = np.load(golden_path(os.path.join(GOLDEN_DIR, 'history.npz')))
    assert set(cur_h.files) == set(gold_h.files), (
        f"history keys drifted: {sorted(set(cur_h.files) ^ set(gold_h.files))}")
    for key in gold_h.files:
        assert_close(cur_h[key], gold_h[key], label=f'history[{key}]')
    print(f"  ok: history.npz ({len(gold_h.files)} arrays)")

    cur_w = np.load(os.path.join(out_dir, 'best_weights.npy'))
    gold_w = np.load(golden_path(os.path.join(GOLDEN_DIR, 'best_weights.npy')))
    assert_close(cur_w, gold_w, label='best_weights')
    print("  ok: best_weights.npy")

    cur_d = np.load(os.path.join(out_dir, 'fixed_z_density.npy'))
    gold_d = np.load(golden_path(
        os.path.join(GOLDEN_DIR, 'fixed_z_density.npy')))
    assert_close(cur_d, gold_d, label='fixed_z_density')
    print("  ok: fixed_z_density.npy")


if __name__ == '__main__':
    main_dispatch(generate, test, 'test_e2e_patha')
