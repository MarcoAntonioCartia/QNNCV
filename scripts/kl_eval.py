"""
KL scoring in the style of Riofrío et al. (2024), for comparing with their
parameter-parity result (eval-only; reuses scripts/rescore.py's loaders, so the
validation split and the checkpoint are rebuilt exactly as for the re-score).

Riofrío et al. score a model by KL(Q || P): Q the histogram of the training data,
P that of the generated samples, on a 16 x 16 grid. Here densities are known
exactly, so no sampling is needed; grids of 32 are sum-pooled to 16 x 16.
A floor of 1e-10 per cell (then renormalised) keeps KL finite where a model
puts no mass; that floor dominates KL for models with empty cells, so read
KL together with W1.

Modes
  single  (Path A runs, one target): the target is the family's canonical
          member. Reports KL(target || mean generated density) at the run's
          grid and at 16 x 16, the mean per-sample KL, and the canonical W1.
  family  (GAN runs): mixture KL(mean validation density || mean generated
          density) at 16 x 16 -- the closest analogue of Riofrío's single-
          distribution score (their target IS the data mixture) -- plus
          nearest-member KL (mean over fakes of min over members) and its
          coverage analogue (mean over members of min over fakes).

    python scripts/kl_eval.py --mode single logs/<Path-A runs>
    python scripts/kl_eval.py --mode family logs/<GAN runs>
Writes analysis/kl_eval/kl_<mode>.csv (one row per run) and prints group medians.
"""

import argparse
import csv
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rescore as R  # noqa: E402  (also bootstraps the repo path)

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

FLOOR = 1e-10


def to16(P):
    """(n, G, G) -> (n, 16, 16) by sum pooling (G a multiple of 16)."""
    n, G, _ = P.shape
    f = G // 16
    return P.reshape(n, 16, f, 16, f).sum(axis=(2, 4))


def norm(P):
    P = np.clip(P.astype(np.float64), FLOOR, None)
    return P / P.sum(axis=(-2, -1), keepdims=True)


def kl(q, p):
    """KL(q || p) over the last two axes, broadcasting."""
    q, p = norm(q), norm(p)
    return (q * np.log(q / p)).sum(axis=(-2, -1))


def load(run_dir, n_samples, seed):
    cfg = json.load(open(os.path.join(run_dir, 'run_config.json')))
    R.seed_everything(int(cfg['seed']))
    fam = R.get_family(cfg['family'], int(cfg['grid_size']), 3.0)
    data = R.generate_dataset(fam, cfg['n_train'] + cfg['n_val'])
    val = np.array([m for m, _ in data[cfg['n_train']:]])
    canon, _ = fam.get_canonical()
    g = R.build_generator(cfg)
    g.load(os.path.join(run_dir, 'best_weights.npy'))
    B = g.batch_size or 8
    tf.random.set_seed(seed)
    fakes = []
    while len(fakes) < n_samples:
        p, _ = g.generate_batch(tf.random.normal([B, g.latent_dim]), fam.xvec, fam.yvec)
        for b in range(B):
            pb = p[b].numpy()
            if np.isfinite(pb).all() and len(fakes) < n_samples:
                fakes.append(pb)
    return cfg, np.array(fakes), val, np.asarray(canon)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--mode', choices=['single', 'family'], required=True)
    ap.add_argument('--n-samples', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join('analysis', 'kl_eval'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = sorted({r for pat in a.runs for r in (glob.glob(pat) or [pat])
                   if os.path.exists(os.path.join(r, 'best_weights.npy'))})
    rows = []
    for rd in runs:
        name = os.path.basename(os.path.normpath(rd))
        cfg, F, V, C = load(rd, a.n_samples, a.seed)
        G = F.shape[-1]
        row = {'run': name, 'group': R.group_of(name), 'seed': cfg['seed'], 'grid': G,
               'generator': cfg.get('generator', 'cv')}
        if a.mode == 'single':
            Fm = F.mean(0)
            row.update(kl_native=float(kl(C, Fm)), kl16=float(kl(to16(C[None])[0], to16(Fm[None])[0])),
                       kl16_per_sample=float(kl(to16(C[None]), to16(F)).mean()),
                       canon_w1=float(np.mean([R.compute_wasserstein_2d(C, f) for f in F])))
        else:
            F16, V16 = to16(F), to16(V)
            M = kl(V16[None, :, :, :], F16[:, None, :, :])        # (n_fake, n_val): KL(member || fake)
            row.update(kl16_mixture=float(kl(V16.mean(0), F16.mean(0))),
                       kl16_nearest=float(M.min(1).mean()), kl16_coverage=float(M.min(0).mean()))
        rows.append(row)
        print(name, {k: round(v, 4) for k, v in row.items() if isinstance(v, float)}, flush=True)
    path = os.path.join(a.out, f'kl_{a.mode}.csv')
    old = []
    if os.path.exists(path):
        done = {r['run'] for r in rows}
        old = [r for r in csv.DictReader(open(path, encoding='utf-8')) if r['run'] not in done]
    allrows = old + rows
    if allrows:
        keys = list(dict.fromkeys(k for r in allrows for k in r))
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(allrows)
    groups = {}
    for r in rows:
        groups.setdefault(r['group'], []).append(r)
    print('\ngroup medians:')
    for gname, rs in sorted(groups.items()):
        cols = [k for k in rs[0] if k.startswith(('kl', 'canon'))]
        print(f"  {gname} (n={len(rs)}): " + ", ".join(
            f"{c} {statistics.median(float(r[c]) for r in rs):.4f}" for c in cols))
    print('written', path)


if __name__ == '__main__':
    main()
