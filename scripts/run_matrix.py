"""
Run a list of train_2d_qgan.py argument lines N at a time (benchmark matrices
whose arms differ in categorical / per-arm flags that run_sweep.py can only
pass through to every run at once).

    python scripts/run_matrix.py jobs.txt --parallel 12 --tag dv_ladder
    python scripts/run_matrix.py jobs.txt --python C:/.../qnncv-dv/python.exe ...

jobs.txt: one train_2d_qgan.py argument line per run; blank lines and lines
starting with '#' are ignored. Each run's console goes to
logs/matrix_<tag>/<index>.log; the run itself writes its own folder under
logs/ as usual. --dry-run prints the commands. Exit code 1 if any run failed.

--min-free-gb G (default 0 = off) is a memory guard: a new run is started only
while the machine's free RAM is above G GB, and at most one run is started per
--start-gap seconds so a fresh process can ramp up before the next check. Runs
already started are never touched; the scheduler simply waits. Needs psutil.
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def free_gb():
    import psutil  # imported here so the script runs without it when the guard is off
    return psutil.virtual_memory().available / 2 ** 30


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('jobs', help='text file, one train_2d_qgan.py argument line per run')
    ap.add_argument('--parallel', type=int, default=8)
    ap.add_argument('--tag', default=time.strftime('%Y%m%d_%H%M%S'))
    ap.add_argument('--python', default=sys.executable, help='interpreter (e.g. the qnncv-dv env)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--min-free-gb', type=float, default=0.0,
                    help='start a new run only while free RAM exceeds this many GB (0 = off)')
    ap.add_argument('--start-gap', type=float, default=20.0,
                    help='seconds between consecutive starts while the memory guard is on')
    args = ap.parse_args()
    if args.min_free_gb > 0:
        free_gb()  # fail early if psutil is missing

    lines = [ln.strip() for ln in open(args.jobs, encoding='utf-8')]
    jobs = [ln for ln in lines if ln and not ln.startswith('#')]
    log_dir = REPO / 'logs' / f'matrix_{args.tag}'
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1',
               TF_CPP_MIN_LOG_LEVEL='3', PYTHONUNBUFFERED='1')
    print(f"{len(jobs)} runs, {args.parallel} at a time, logs in {log_dir}")
    if args.dry_run:
        for i, j in enumerate(jobs):
            print(f"  [{i:03d}] python train_2d_qgan.py {j}")
        return 0
    log_dir.mkdir(parents=True, exist_ok=True)

    pending = list(enumerate(jobs))
    running, failed, t0 = [], [], time.time()
    last_start, waiting = 0.0, False
    while pending or running:
        while pending and len(running) < args.parallel:
            if args.min_free_gb > 0:
                if time.time() - last_start < args.start_gap:
                    break
                free = free_gb()
                if free < args.min_free_gb:
                    if not waiting:
                        print(f"[{time.strftime('%H:%M:%S')}] wait   free RAM {free:.1f} GB < "
                              f"{args.min_free_gb:g} GB; {len(running)} running, "
                              f"{len(pending)} pending", flush=True)
                        waiting = True
                    break
                waiting = False
            i, j = pending.pop(0)
            fh = open(log_dir / f'{i:03d}.log', 'w', encoding='utf-8')
            fh.write(f"python train_2d_qgan.py {j}\n\n"); fh.flush()
            proc = subprocess.Popen([args.python, '-u', '-W', 'ignore', 'train_2d_qgan.py']
                                    + shlex.split(j), stdout=fh, stderr=subprocess.STDOUT,
                                    cwd=str(REPO), env=env)
            running.append((i, proc, fh, time.time()))
            last_start = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] start  {i:03d}  {j}", flush=True)
        time.sleep(5)
        still = []
        for i, proc, fh, ts in running:
            rc = proc.poll()
            if rc is None:
                still.append((i, proc, fh, ts))
                continue
            fh.close()
            mins = (time.time() - ts) / 60
            print(f"[{time.strftime('%H:%M:%S')}] done   {i:03d}  rc={rc}  {mins:.1f} min"
                  f"  ({len(pending)} pending)", flush=True)
            if rc != 0:
                failed.append(i)
        running = still
    print(f"finished in {(time.time() - t0) / 60:.1f} min; failed: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
