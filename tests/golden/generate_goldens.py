"""
Generate ALL golden references from the CURRENT code. Run ONCE against the
pristine monolith (Phase 0), before any extraction.

Each test_*.py owns its generation logic (`--generate`); this script just
runs them all in fresh subprocesses. Refuses to overwrite existing goldens
unless --force is given.

Usage (from repo root, qnncv python):
    C:\\Users\\mende\\.conda\\envs\\qnncv\\python.exe tests/golden/generate_goldens.py [--force]
"""

import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _harness import REPO, DATA_DIR, run_subprocess  # noqa: E402


def main():
    force = '--force' in sys.argv
    if os.path.isdir(DATA_DIR) and os.listdir(DATA_DIR) and not force:
        print(f"REFUSING: {DATA_DIR} already contains goldens "
              f"(use --force to regenerate)")
        return 2

    suites = sorted(glob.glob(os.path.join(HERE, 'test_*.py')))
    failed = []
    for path in suites:
        name = os.path.basename(path)
        args = [sys.executable, path, '--generate'] + (
            ['--force'] if force else [])
        t0 = time.time()
        proc = run_subprocess(args, cwd=REPO)
        dt = time.time() - t0
        ok = (proc.returncode == 0)
        print(f"{'DONE' if ok else 'FAIL'}  {name:28s} ({dt:6.1f}s)")
        if not ok:
            failed.append(name)
            print((proc.stdout + '\n--- STDERR ---\n' + proc.stderr)[-4000:])

    if failed:
        print(f"GENERATION FAILED for: {failed}")
        return 1
    print("ALL GOLDENS GENERATED")
    return 0


if __name__ == '__main__':
    sys.exit(main())
