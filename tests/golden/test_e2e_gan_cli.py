"""
End-to-end golden: GAN mode (WGAN-GP) via the real CLI, tiny config.

Runs `python train_2d_qgan.py <frozen ring flags + tiny sizes>` as a
subprocess with cwd = a scratch dir (so ./logs is isolated), then freezes:
  - every history.npz array (atol 1e-6; seed exact),
  - best_weights.npy,
  - run_config.json minus 'timestamp' (exact — pins seed/env-var provenance,
    i.e. the --deterministic argv-scan-before-TF-import behavior),
  - the run-folder name suffix with the timestamp stripped,
  - console markers (Seed:/Output:, no PATH A banner).

This exercises argparse, build_hp_suffix, main() dispatch, the training
loop, checkpointing and all output files in one shot.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (REPO, TRAIN_PATH, run_subprocess, scratch_dir,
                      save_golden_dir, golden_path, save_golden_text,
                      load_golden_text, assert_close, main_dispatch)

GOLDEN_DIR = 'e2e_gan'

CLI_ARGS = [
    '--family', 'ring', '--n-modes', '3', '--n-train', '8', '--n-val', '6',
    '--n-layers', '2', '--cutoff-dim', '6', '--epochs', '4',
    '--val-every', '2', '--plot-every', '999', '--seed', '0',
    '--deterministic', '--d-lr', '0.0005', '--gp-weight', '10',
    '--n-critic', '5', '--batch-size', '8', '--d-dropout', '0',
]

TS_RE = re.compile(r'_\d{8}_\d{6}')


def run_cli():
    """Run the CLI in a scratch cwd; return (run_dir, console, folder_suffix)."""
    cwd = scratch_dir('e2e_gan')
    proc = run_subprocess([sys.executable, TRAIN_PATH] + CLI_ARGS, cwd=cwd)
    console = proc.stdout + '\n--- STDERR ---\n' + proc.stderr
    assert proc.returncode == 0, f"CLI run failed:\n{console[-4000:]}"
    m = re.search(r'^Output: (.+)$', proc.stdout, re.MULTILINE)
    assert m, f"no 'Output:' line in console:\n{proc.stdout[-2000:]}"
    run_dir = m.group(1).strip()
    if not os.path.isabs(run_dir):
        run_dir = os.path.normpath(os.path.join(cwd, run_dir))
    assert os.path.isdir(run_dir), f"run dir not found: {run_dir}"
    folder_suffix = TS_RE.sub('', os.path.basename(run_dir))
    return run_dir, proc.stdout, folder_suffix


def generate(force=False):
    run_dir, stdout, folder_suffix = run_cli()
    save_golden_dir(GOLDEN_DIR, {
        'history.npz': os.path.join(run_dir, 'history.npz'),
        'best_weights.npy': os.path.join(run_dir, 'best_weights.npy'),
        'run_config.json': os.path.join(run_dir, 'run_config.json'),
    }, force=force)
    save_golden_text(os.path.join(GOLDEN_DIR, 'console.txt'), stdout,
                     force=force)
    save_golden_text(os.path.join(GOLDEN_DIR, 'folder_suffix.txt'),
                     folder_suffix, force=force)


def load_config_no_ts(path):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    cfg.pop('timestamp', None)
    return cfg


def test():
    import numpy as np
    run_dir, stdout, folder_suffix = run_cli()

    # console markers
    assert 'Seed: 0' in stdout, "console missing 'Seed: 0'"
    assert 'PATH A MODE' not in stdout, "GAN run unexpectedly in Path A mode"
    print("  ok: console markers")

    # folder suffix (timestamp-stripped) exact
    gold_suffix = load_golden_text(
        os.path.join(GOLDEN_DIR, 'folder_suffix.txt'))
    assert folder_suffix == gold_suffix, (
        f"folder suffix drifted: {folder_suffix!r} != {gold_suffix!r}")
    print(f"  ok: folder suffix {folder_suffix!r}")

    # run_config.json exact minus timestamp
    cur_cfg = load_config_no_ts(os.path.join(run_dir, 'run_config.json'))
    gold_cfg = load_config_no_ts(golden_path(
        os.path.join(GOLDEN_DIR, 'run_config.json')))
    assert cur_cfg == gold_cfg, (
        "run_config.json drifted:\n"
        + json.dumps({k: (gold_cfg.get(k), cur_cfg.get(k))
                      for k in sorted(set(gold_cfg) | set(cur_cfg))
                      if gold_cfg.get(k) != cur_cfg.get(k)}, indent=2))
    print("  ok: run_config.json")

    # history arrays
    cur_h = np.load(os.path.join(run_dir, 'history.npz'))
    gold_h = np.load(golden_path(os.path.join(GOLDEN_DIR, 'history.npz')))
    assert set(cur_h.files) == set(gold_h.files), (
        f"history keys drifted: {sorted(set(cur_h.files) ^ set(gold_h.files))}")
    assert int(cur_h['seed']) == int(gold_h['seed']) == 0
    for key in gold_h.files:
        assert_close(cur_h[key], gold_h[key], label=f'history[{key}]')
    print(f"  ok: history.npz ({len(gold_h.files)} arrays)")

    # best weights
    cur_w = np.load(os.path.join(run_dir, 'best_weights.npy'))
    gold_w = np.load(golden_path(
        os.path.join(GOLDEN_DIR, 'best_weights.npy')))
    assert_close(cur_w, gold_w, label='best_weights')
    print("  ok: best_weights.npy")


if __name__ == '__main__':
    main_dispatch(generate, test, 'test_e2e_gan_cli')
