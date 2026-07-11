"""Verify seed plumbing in train_2d_qgan.py (2026-07-10 reproducibility edit).

Two independent checks, both tiny end-to-end runs (output under <system temp>,
never ./logs except the CLI round-trip which cleans up after itself):

  1. SAME-MACHINE DETERMINISM
     Run train_2d_qgan(seed=0, ...) twice, each in a FRESH subprocess (to avoid
     in-process TF/Keras/SF global-state bleed), with identical args. Load both
     history.npz and assert the RNG-driven arrays match within tight tolerance
     (ideally bit-identical). If they don't match with a fixed seed on one
     machine, something isn't being seeded -- find it before declaring done.

  2. SEED ROUND-TRIP
     Run the actual CLI once with --seed 0 and confirm the seed value is
     identical across all four provenance surfaces:
       folder-name tag (_seed0)  ==  run_config.json  ==  history.npz  ==  console

Usage:  python scripts/verify_determinism.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(REPO, 'train_2d_qgan.py')

# Tiny, fast, but exercises every RNG path (dataset gen, weight init, z, noise,
# GP, batch-index draws, validation).
TRAIN_KWARGS = dict(
    seed=0, family_name='gaussian', n_train=8, n_val=6, n_total_modes=2,
    n_layers=2, cutoff_dim=6, epochs=4, val_every=2, plot_every=999,
    batch_size=4, n_critic=2, d_dropout=0.0,
)

# Arrays whose values are driven by the seeded RNGs; these must reproduce.
RNG_KEYS = ('g_loss', 'd_loss', 'wasserstein', 'g_grad_norm', 'd_grad_norm',
            'gp_value')

# Keep both subprocesses in the same, quiet environment so the only variable
# under test is the seed.
CHILD_ENV = dict(os.environ)
CHILD_ENV.update(PYTHONIOENCODING='utf-8', TF_CPP_MIN_LOG_LEVEL='3',
                 TF_ENABLE_ONEDNN_OPTS='0', PYTHONUNBUFFERED='1')


def run_train_subprocess(out_dir):
    """Invoke train_2d_qgan(**TRAIN_KWARGS, log_dir=out_dir) in a fresh process."""
    code = (
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('t2q', {TRAIN!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        f"mod.train_2d_qgan(log_dir={out_dir!r}, **{TRAIN_KWARGS!r})\n"
    )
    r = subprocess.run([sys.executable, '-W', 'ignore', '-c', code],
                       cwd=REPO, env=CHILD_ENV,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + '\n' + r.stderr + '\n')
        raise SystemExit(f"train subprocess failed (rc={r.returncode}) for {out_dir}")
    return np.load(os.path.join(out_dir, 'history.npz'), allow_pickle=True)


def check_determinism(tmp):
    print('=== Check 1: same-machine determinism (two --seed 0 runs) ===')
    a = run_train_subprocess(os.path.join(tmp, 'run_a'))
    b = run_train_subprocess(os.path.join(tmp, 'run_b'))

    bit_identical = True
    for k in RNG_KEYS:
        assert k in a.files and k in b.files, f"missing history key: {k}"
        va, vb = np.asarray(a[k]), np.asarray(b[k])
        if not np.array_equal(va, vb):
            bit_identical = False
            if not np.allclose(va, vb, rtol=1e-6, atol=1e-8):
                raise SystemExit(
                    f"FAIL: '{k}' differs between two seed-0 runs beyond tolerance -- "
                    f"a seeding path is missing.\n  a={va}\n  b={vb}")
    # Seed scalar must be recorded and correct in both.
    assert int(a['seed']) == 0 and int(b['seed']) == 0, \
        f"seed not recorded as 0: a={a['seed']} b={b['seed']}"

    print('PASS: both runs match on', list(RNG_KEYS),
          '(bit-identical)' if bit_identical else '(within 1e-6/1e-8 tolerance)')
    print('PASS: history.npz["seed"] == 0 in both runs')


def check_roundtrip(tmp):
    print('\n=== Check 2: seed round-trips across all four provenance surfaces ===')
    logs_root = os.path.join(tmp, 'logs')
    # main() writes to ./logs relative to cwd; use an isolated cwd so we can
    # clean up without touching the repo's real ./logs.
    cwd = os.path.join(tmp, 'cli')
    os.makedirs(cwd)
    cmd = [sys.executable, '-W', 'ignore', TRAIN,
           '--family', 'gaussian', '--seed', '0', '--epochs', '4',
           '--val-every', '2', '--n-train', '8', '--n-val', '6',
           '--n-layers', '2', '--cutoff-dim', '6', '--batch-size', '4',
           '--n-critic', '2']
    r = subprocess.run(cmd, cwd=cwd, env=CHILD_ENV, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + '\n' + r.stderr + '\n')
        raise SystemExit(f"CLI run failed (rc={r.returncode})")

    console = r.stdout
    # Console: header line "Seed: 0"
    assert re.search(r'^Seed:\s*0\s*$', console, re.M), 'console header "Seed: 0" missing'
    m = re.search(r'^Output:\s*(.+?)\s*$', console, re.M)
    assert m, 'could not find "Output:" line to locate the run folder'
    log_dir = os.path.join(cwd, m.group(1)) if not os.path.isabs(m.group(1)) else m.group(1)
    folder = os.path.basename(log_dir.rstrip('/\\'))

    # Folder tag
    assert 'seed0' in folder, f"folder tag missing 'seed0': {folder}"
    # run_config.json
    with open(os.path.join(log_dir, 'run_config.json')) as f:
        cfg = json.load(f)
    assert cfg['seed'] == 0, f"run_config.json seed != 0: {cfg['seed']}"
    # history.npz
    hist = np.load(os.path.join(log_dir, 'history.npz'), allow_pickle=True)
    assert int(hist['seed']) == 0, f"history.npz seed != 0: {hist['seed']}"

    print(f'PASS: folder tag  -> {folder!r} contains "seed0"')
    print(f'PASS: run_config.json seed = {cfg["seed"]}')
    print(f'PASS: history.npz  seed = {int(hist["seed"])}')
    print('PASS: console printed "Seed: 0"')
    print('PASS: all four surfaces agree (seed == 0)')


def main():
    tmp = tempfile.mkdtemp(prefix='qnncv_determinism_')
    try:
        check_determinism(tmp)
        check_roundtrip(tmp)
        print('\nALL CHECKS PASSED')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
