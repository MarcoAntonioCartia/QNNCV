"""
Run every golden test suite, each in a FRESH subprocess (no TF/Keras
global-state bleed between suites), with UTF-8 stdout forced.

Usage (from repo root, with the qnncv conda python):
    C:\\Users\\mende\\.conda\\envs\\qnncv\\python.exe tests/golden/run_all.py

Exit code 0 iff every suite passes.
"""

import glob
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _harness import REPO, run_subprocess  # noqa: E402


def main():
    if importlib.util.find_spec('tensorflow') is None:
        print("ERROR: tensorflow not importable — run with the qnncv python:")
        print(r"  C:\Users\mende\.conda\envs\qnncv\python.exe "
              r"tests\golden\run_all.py")
        return 2

    suites = sorted(glob.glob(os.path.join(HERE, 'test_*.py')))
    if not suites:
        print("no test suites found")
        return 2

    results = []
    for path in suites:
        name = os.path.basename(path)
        t0 = time.time()
        proc = run_subprocess([sys.executable, path], cwd=REPO)
        dt = time.time() - t0
        ok = (proc.returncode == 0)
        results.append((name, ok, dt))
        print(f"{'PASS' if ok else 'FAIL'}  {name:28s} ({dt:6.1f}s)")
        if not ok:
            tail = (proc.stdout + '\n--- STDERR ---\n' + proc.stderr)
            print(tail[-4000:])

    print("-" * 50)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    total = sum(dt for _, _, dt in results)
    print(f"{len(results) - n_fail}/{len(results)} suites passed "
          f"({total:.0f}s total)")
    if n_fail == 0:
        print("ALL GOLDEN SUITES PASSED")
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
