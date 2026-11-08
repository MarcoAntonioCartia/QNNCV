"""
Group a rescore.csv (scripts/rescore.py) by arm and print a markdown table
with the pre-registered "reliable baseline" criterion of docs/benchmark_plan.md:

  pass = every seed of the arm has diversity within a factor 2 of the
         validation self-diversity (no collapse, no blow-up), the arm's
         median fidelity W1 is within 1.5x the best arm's median on the
         same grid, and no seed produced a non-finite score.

    python scripts/summarize_rescore.py analysis/rescore/rescore.csv
    python scripts/summarize_rescore.py analysis/rescore_g16/rescore.csv --filter gdv --sort cost

--filter keeps groups whose name contains the substring. Wall time per run
comes from backend_info.json next to each run folder when --logs is given.
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


def med_rng(v):
    v = np.asarray(v, float)
    return f"{np.median(v):.2f} ({v.min():.2f}-{v.max():.2f})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv')
    ap.add_argument('--filter', default=None)
    ap.add_argument('--logs', default='logs', help='run folders (for wall time); "" to skip')
    ap.add_argument('--sort', choices=['fid', 'cost', 'name'], default='fid')
    ap.add_argument('--fid-factor', type=float, default=1.5)
    ap.add_argument('--div-factor', type=float, default=2.0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding='utf-8')))
    if args.filter:
        rows = [r for r in rows if args.filter in r['group']]
    groups = defaultdict(list)
    for r in rows:
        groups[r['group']].append(r)

    def f(r, k):
        try:
            return float(r[k])
        except (KeyError, ValueError, TypeError):
            return float('nan')

    def wall(r):
        if not args.logs:
            return float('nan')
        p = os.path.join(args.logs, r['run'], 'backend_info.json')
        try:
            return float(json.load(open(p))['wall_seconds']) / 60.0
        except (OSError, KeyError, ValueError):
            return float('nan')

    summ = []
    for g, rs in groups.items():
        fid = [f(r, 'fid_w1') for r in rs]
        cov = [f(r, 'cov_w1') for r in rs]
        ratio = [f(r, 'diversity_w1') / f(r, 'val_self_div_w1') for r in rs]
        grid = rs[0].get('grid') or rs[0].get('train_grid') or '?'
        summ.append(dict(group=g, n=len(rs), grid=grid, params=rs[0].get('n_params'),
                         hilbert=rs[0].get('hilbert_dim'), fid=fid, cov=cov, ratio=ratio,
                         mmd_p=min(f(r, 'mmd_p') for r in rs),
                         wall=np.nanmedian([wall(r) for r in rs]),
                         finite=all(np.isfinite(fid + cov + ratio))))
    by_grid_best = {}
    for s in summ:
        if s['finite']:
            m = np.median(s['fid'])
            by_grid_best[s['grid']] = min(by_grid_best.get(s['grid'], np.inf), m)
    for s in summ:
        s['pass'] = (s['finite']
                     and all(1 / args.div_factor <= x <= args.div_factor for x in s['ratio'])
                     and np.median(s['fid']) <= args.fid_factor * by_grid_best[s['grid']])
    key = {'fid': lambda s: np.median(s['fid']), 'cost': lambda s: s['wall'],
           'name': lambda s: s['group']}[args.sort]
    summ.sort(key=key)

    print(f"| arm | grid | params | hilbert | n | fidelity W1 med (range) | coverage W1 med (range) "
          f"| diversity / val (range) | min MMD p | min/run | pass |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summ:
        print(f"| {s['group']} | {s['grid']} | {s['params']} | {s['hilbert']} | {s['n']} | "
              f"{med_rng(s['fid'])} | {med_rng(s['cov'])} | {med_rng(s['ratio'])} | "
              f"{s['mmd_p']:.3f} | {s['wall']:.0f} | {'yes' if s['pass'] else 'no'} |")
    n_pass = sum(s['pass'] for s in summ)
    print(f"\n{len(summ)} arms, {n_pass} pass (div within x{args.div_factor} of val on every seed, "
          f"median fid within x{args.fid_factor} of the best arm on the same grid).")


if __name__ == '__main__':
    main()
