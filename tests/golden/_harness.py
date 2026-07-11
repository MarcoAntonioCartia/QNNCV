"""
Shared helpers for the golden-reference test harness.

Every test_*.py in this directory is a plain-assert script (no pytest) that
doubles as its own golden generator:

    python tests/golden/test_X.py --generate   # write golden (refuses overwrite
                                               # unless --force)
    python tests/golden/test_X.py              # compare against golden

Run everything via tests/golden/run_all.py (fresh subprocess per suite).

Tests load the code under test through the ROOT MONOLITH/SHIM train_2d_qgan.py
(importlib, like scripts/verify_batching.py) so the same tests are valid in
every extraction phase: they always exercise whatever the root file currently
re-exports.

IMPORTANT: run with the qnncv conda python (TF 2.18 / NumPy 2.0.2 / SF 0.23),
not the base Anaconda python.
"""

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA_DIR = os.path.join(HERE, 'data')
TRAIN_PATH = os.path.join(REPO, 'train_2d_qgan.py')

# Child-process environment: force UTF-8 stdout (legacy src/__init__ prints
# '✗' which crashes cp1252 pipes) and quiet, thread-stable TF. Mirrors
# scripts/verify_determinism.py's CHILD_ENV.
def child_env():
    env = dict(os.environ)
    env.update({
        'PYTHONUTF8': '1',
        'PYTHONIOENCODING': 'utf-8',
        'PYTHONUNBUFFERED': '1',
        'TF_CPP_MIN_LOG_LEVEL': '3',
        'TF_ENABLE_ONEDNN_OPTS': '0',
    })
    return env


def run_subprocess(cmd, cwd=None, timeout=1800):
    """Run a child python process with the harness env; return CompletedProcess."""
    return subprocess.run(
        cmd, cwd=cwd or REPO, env=child_env(),
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        timeout=timeout)


def load_t2q():
    """Load the root train_2d_qgan.py as a module (the strangler surface).

    Repo root is inserted at sys.path[0] FIRST so that hard `from src...`
    imports inside the (partially extracted) monolith resolve regardless of
    how this test process was launched. Goldens are generated under this same
    context, so comparisons are self-consistent across phases.
    """
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    import importlib.util
    spec = importlib.util.spec_from_file_location('t2q', TRAIN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def seed_all(seed):
    """Local mirror of the monolith's seed_everything (usable pre-load)."""
    import random
    import numpy as np
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class FakeState:
    """Stub for an SF state: .ket() returns the injected tensor."""

    def __init__(self, ket):
        self._ket = ket

    def ket(self):
        return self._ket


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

RTOL = 1e-6
ATOL = 1e-8


def assert_exact(actual, desired, label=''):
    import numpy as np
    a = np.asarray(actual)
    d = np.asarray(desired)
    assert a.shape == d.shape, f"{label}: shape {a.shape} != {d.shape}"
    assert np.array_equal(a, d), f"{label}: arrays differ (exact compare)"


def assert_close(actual, desired, label='', rtol=RTOL, atol=ATOL):
    import numpy as np
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(desired),
        rtol=rtol, atol=atol, equal_nan=True, err_msg=label)


# ---------------------------------------------------------------------------
# Golden storage
# ---------------------------------------------------------------------------

def golden_path(name):
    return os.path.join(DATA_DIR, name)


def save_golden_npz(name, arrays, force=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = golden_path(name)
    if os.path.exists(path) and not force:
        raise SystemExit(
            f"REFUSING to overwrite existing golden {path} (use --force)")
    import numpy as np
    np.savez(path, **arrays)
    print(f"golden written: {path}")


def load_golden_npz(name):
    import numpy as np
    path = golden_path(name)
    assert os.path.exists(path), (
        f"golden {path} missing — run this script with --generate first")
    return np.load(path, allow_pickle=False)


def save_golden_dir(name, src_files, force=False):
    """Copy a dict {dest_name: src_path} into data/<name>/."""
    dest = golden_path(name)
    if os.path.isdir(dest) and os.listdir(dest) and not force:
        raise SystemExit(
            f"REFUSING to overwrite existing golden dir {dest} (use --force)")
    os.makedirs(dest, exist_ok=True)
    for dest_name, src_path in src_files.items():
        shutil.copyfile(src_path, os.path.join(dest, dest_name))
    print(f"golden dir written: {dest} ({sorted(src_files)})")


def save_golden_text(name, text, force=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = golden_path(name)
    if os.path.exists(path) and not force:
        raise SystemExit(
            f"REFUSING to overwrite existing golden {path} (use --force)")
    with io.open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"golden written: {path}")


def load_golden_text(name):
    path = golden_path(name)
    assert os.path.exists(path), (
        f"golden {path} missing — run this script with --generate first")
    with io.open(path, 'r', encoding='utf-8') as f:
        return f.read()


def save_golden_json(name, obj, force=False):
    save_golden_text(name, json.dumps(obj, indent=2, sort_keys=True), force)


def load_golden_json(name):
    return json.loads(load_golden_text(name))


def scratch_dir(tag):
    return tempfile.mkdtemp(prefix=f'qnncv_golden_{tag}_')


def main_dispatch(generate_fn, test_fn, name):
    """Standard entry point: --generate [--force] writes goldens, else test."""
    if '--generate' in sys.argv:
        generate_fn(force='--force' in sys.argv)
        print(f"GENERATED {name}")
    else:
        test_fn()
        print(f"PASS {name}")
