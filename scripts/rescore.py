"""
Re-score saved checkpoints with more samples and family-level metrics
(Phase 1f of docs/benchmark_plan.md). Read-only on run folders; writes
analysis/rescore/<run>.json and analysis/rescore/rescore.csv (one row per
run folder, joinable with analysis/sweep/runs.csv on `run`).

Why: the trainer validates with 10 z-draws, which gives nearest-member W1
a noise of about +-0.08 per reading, and the "best checkpoint" is the
minimum of 15 such readings (optimistic by ~0.03). Here every checkpoint
gets the SAME 200 fixed-seed draws and the same scores:

  fidelity  (nearest-member): mean over fakes of the distance to the
            closest validation member -- "is each fake a plausible member?"
  coverage  : mean over validation members of the distance to the closest
            fake -- "is every member approximated by some fake?" (detects
            collapse, which fidelity alone cannot see)
  chamfer   = fidelity + coverage
  both in marginal W1 (grid cells, as the sweep) and in energy distance
  (full 2D, x/y units).
  diversity_w1 / val_self_div_w1: mean pairwise W1 among fakes / among
            validation members (the reference the trainer only prints).
  mmd2, mmd_p: two-sample MMD (RBF on flattened densities, median
            bandwidth) between fakes and validation members with a
            permutation p-value -- "could a statistician tell them apart?"
  ket_norm_min/mean: Fock-truncation health over the 200 samples (CV).

The validation set is rebuilt from run_config.json (same seed, same
family, same split as training).

    python scripts/rescore.py logs/qgan_2d_ring_2026110*_ep300_*
    python scripts/rescore.py --n-samples 200 --seed 0 logs/<run> ...
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from src.backends.cv import CVGeneratorBackend  # noqa: E402
from src.quantum.circuit import EncodingSpec  # noqa: E402
from src.training.qgan_2d import seed_everything  # noqa: E402
from src.families.registry import get_family  # noqa: E402
from src.data.dataset import generate_dataset  # noqa: E402
from src.metrics.wasserstein import compute_wasserstein_2d  # noqa: E402
from src.metrics.energy import build_energy_distance_context  # noqa: E402

TS_RE = re.compile(r'^(?P<prefix>.*?)_(?P<ts>\d{8}_\d{6})(?P<suffix>.*)$')
SEED_RE = re.compile(r'_seed(?P<seed>\d+)$')


def group_of(name):
    """Folder name minus timestamp and _seedN (= analyze_sweep's tag)."""
    base = SEED_RE.sub('', name)
    m = TS_RE.match(base)
    return (m.group('prefix') + m.group('suffix')) if m else base


class _ShimLoader:
    """Non-CV backends: build from run_config, load the backend-state file."""

    def __init__(self, cfg):
        from src.backends.trainer_shim import build_generator as _bg, load_backend_state
        self.backend = _bg(cfg)
        self._load = load_backend_state

    def load(self, path):
        self.backend.load_state_dict(self._load(path)['state'])

    def __getattr__(self, name):
        return getattr(self.backend, name)


def build_generator(cfg):
    backend = cfg.get('generator', 'cv')
    if backend != 'cv':
        return _ShimLoader(cfg)
    enc = cfg.get('encoding', 'input_displacement')
    if enc == 'input_displacement':
        spec = EncodingSpec(enc)
    else:
        inj = cfg.get('inj_layers')
        if isinstance(inj, str):
            inj = tuple(int(s) for s in inj.split(',') if s.strip())
        legacy_trainable = 'squeeze_scale' not in cfg   # pre-fixed-scale runs
        spec = EncodingSpec(enc, inj_layers=inj, r_max=cfg.get('r_max'),
                            squeeze_scale=cfg.get('squeeze_scale', 0.0),
                            train_squeeze_scale=cfg.get('train_squeeze_scale', legacy_trainable),
                            inj_latent=cfg.get('inj_latent', 'shared'))
    return CVGeneratorBackend(n_modes=cfg['n_modes'], n_layers=cfg['n_layers'],
                              cutoff_dim=cfg['cutoff_dim'], use_kerr=not cfg.get('no_kerr', False),
                              latent_scale=cfg['latent_scale'], batch_size=cfg['batch_size'],
                              encoding=spec)


def rbf_mmd_test(A, B, n_perm, rng):
    """Unbiased MMD^2 with an RBF kernel (median-heuristic bandwidth over
    the pooled sample) and a permutation p-value."""
    Z = np.concatenate([A, B], axis=0).astype(np.float64)
    sq = (Z ** 2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0)
    med = np.median(D2[np.triu_indices(len(Z), 1)])
    K = np.exp(-D2 / (med + 1e-12))
    n, m = len(A), len(B)

    def mmd2(idx):
        a, b = idx[:n], idx[n:]
        Kaa, Kbb, Kab = K[np.ix_(a, a)], K[np.ix_(b, b)], K[np.ix_(a, b)]
        return ((Kaa.sum() - np.trace(Kaa)) / (n * (n - 1))
                + (Kbb.sum() - np.trace(Kbb)) / (m * (m - 1))
                - 2 * Kab.mean())
    obs = mmd2(np.arange(n + m))
    perms = np.array([mmd2(rng.permutation(n + m)) for _ in range(n_perm)])
    p = (1 + np.sum(perms >= obs)) / (n_perm + 1)
    return float(obs), float(p)


def rescore(run_dir, n_samples, seed, n_perm, verbose=True, grid=None):
    """grid=None scores on the run's own grid; grid=G re-renders the
    checkpoint on a GxG grid (CV and classical generators are grid-free;
    the DV backend refuses a grid it cannot produce) against a validation
    set rebuilt at that grid, so arms trained at different grids can be
    compared on one. W1 values are in cells of the grid used."""
    name = os.path.basename(os.path.normpath(run_dir))
    cfg = json.load(open(os.path.join(run_dir, 'run_config.json')))
    t0 = time.time()
    rerender = grid is not None and int(grid) != int(cfg['grid_size'])
    G = int(grid) if grid is not None else int(cfg['grid_size'])
    # validation set exactly as training built it: seed, family, split
    seed_everything(int(cfg['seed']))
    fam = get_family(cfg['family'], G, 3.0)
    data = generate_dataset(fam, cfg['n_train'] + cfg['n_val'])
    val = [m for m, _ in data[cfg['n_train']:]]
    canon, _ = fam.get_canonical()
    ed_ctx = build_energy_distance_context(fam.X, fam.Y, data[cfg['n_train']:], canon)

    g = build_generator(cfg)
    g.load(os.path.join(run_dir, 'best_weights.npy'))
    # assert the rebuilt validation split when the run recorded its fingerprint
    info_path = os.path.join(run_dir, 'backend_info.json')
    if os.path.exists(info_path) and not rerender:
        import hashlib
        want = json.load(open(info_path)).get('val_set_sha1_16')
        got = hashlib.sha1(np.concatenate(
            [np.asarray(m, np.float32).ravel() for m in val]).tobytes()).hexdigest()[:16]
        if want and want != got:
            raise RuntimeError(f"{name}: rebuilt validation set ({got}) differs from the "
                               f"one used in training ({want}); rescore would be wrong")
    B = g.batch_size or 8
    tf.random.set_seed(seed)
    fakes, kets = [], []
    while len(fakes) < n_samples:
        z = tf.random.normal([B, g.latent_dim])
        p, k = g.generate_batch(z, fam.xvec, fam.yvec)
        for b in range(B):
            pb = p[b].numpy()
            if np.isfinite(pb).all() and len(fakes) < n_samples:
                fakes.append(pb)
                kets.append(float(k[b]) if k is not None else 1.0)  # no Fock space => no truncation
    fakes = np.array(fakes)
    nf, nv = len(fakes), len(val)

    # marginal W1 matrix fake x val, and fake x fake / val x val for diversity
    W = np.array([[compute_wasserstein_2d(v, f) for v in val] for f in fakes])
    canon_w1 = np.array([compute_wasserstein_2d(canon, f) for f in fakes])
    rng = np.random.default_rng(seed)
    pairs_f = rng.choice(nf, size=(min(300, nf * (nf - 1) // 2), 2))
    div_f = np.mean([compute_wasserstein_2d(fakes[i], fakes[j]) for i, j in pairs_f if i != j])
    pairs_v = rng.choice(nv, size=(min(300, nv * (nv - 1) // 2), 2))
    div_v = np.mean([compute_wasserstein_2d(val[i], val[j]) for i, j in pairs_v if i != j])

    # energy-distance matrix fake x val via the precomputed context
    flat = lambda a: (a.ravel() / (a.sum() + 1e-10)).astype(np.float32)  # noqa: E731
    E = np.empty((nf, nv))
    for i, f in enumerate(fakes):
        pf = flat(f)
        Dp = ed_ctx['D'] @ pf
        E[i] = 2.0 * (pf @ ed_ctx['val_Dq']) - float(pf @ Dp) - ed_ctx['val_self']
    E = np.maximum(E, 0.0)

    mmd2, mmd_p = rbf_mmd_test(np.array([flat(f) for f in fakes]),
                               np.array([flat(v) for v in val]), n_perm, rng)
    d = g.describe()
    row = {
        'run': name, 'group': group_of(name), 'family': cfg['family'], 'seed': cfg['seed'],
        'grid': G, 'train_grid': cfg['grid_size'],
        'epochs': cfg.get('epochs'), 'n_samples': nf, 'rescore_seed': seed,
        'fid_w1': float(W.min(1).mean()), 'cov_w1': float(W.min(0).mean()),
        'chamfer_w1': float(W.min(1).mean() + W.min(0).mean()),
        'fid_ed': float(E.min(1).mean()), 'cov_ed': float(E.min(0).mean()),
        'chamfer_ed': float(E.min(1).mean() + E.min(0).mean()),
        'canon_w1_mean': float(canon_w1.mean()),
        'diversity_w1': float(div_f), 'val_self_div_w1': float(div_v),
        'mmd2': mmd2, 'mmd_p': mmd_p,
        'ket_norm_min': float(np.min(kets)), 'ket_norm_mean': float(np.mean(kets)),
        'n_params': d['n_params'], 'hilbert_dim': d['hilbert_dim'], 'latent_dim': d['latent_dim'],
        'encoding': d.get('encoding'), 'inj_latent': d.get('inj_latent'),
        'squeeze_scale': cfg.get('squeeze_scale'), 'train_squeeze_scale': cfg.get('train_squeeze_scale'),
        'seconds': round(time.time() - t0, 1),
    }
    if verbose:
        print(f"  {name}: fid_w1 {row['fid_w1']:.3f} cov_w1 {row['cov_w1']:.3f} | "
              f"fid_ed {row['fid_ed']:.4f} cov_ed {row['cov_ed']:.4f} | div {row['diversity_w1']:.2f} "
              f"(val {row['val_self_div_w1']:.2f}) | mmd p={row['mmd_p']:.3f} | ket min {row['ket_norm_min']:.3f} "
              f"({row['seconds']}s)")
    return row


def write_csv(out_dir):
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out_dir, '*.json')))]
    if not rows:
        return
    with open(os.path.join(out_dir, 'rescore.csv'), 'w', newline='', encoding='utf-8') as fh:
        fields = list(rows[0].keys())
        fields += sorted({k for r in rows for k in r} - set(fields))  # keys added later
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('runs', nargs='+', help='run folders (globs allowed)')
    ap.add_argument('--n-samples', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0, help='seed of the z draws (same for every run)')
    ap.add_argument('--n-perm', type=int, default=200, help='permutations for the MMD p-value')
    ap.add_argument('--out', default=os.path.join('analysis', 'rescore'))
    ap.add_argument('--force', action='store_true', help='re-score runs that already have a json')
    ap.add_argument('--grid', type=int, default=None,
                    help="re-render every checkpoint on this grid (default: the run's own); "
                         'with the default --out the folder becomes analysis/rescore_g<G>')
    args = ap.parse_args()
    dirs = []
    for pat in args.runs:
        dirs += [d for d in sorted(glob.glob(pat)) if os.path.isdir(d)]
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, 'best_weights.npy'))
            and os.path.exists(os.path.join(d, 'run_config.json'))]
    if not dirs:
        print("no run folders with best_weights.npy + run_config.json"); return 2
    if args.grid is not None and args.out == os.path.join('analysis', 'rescore'):
        args.out = os.path.join('analysis', f'rescore_g{args.grid}')
    os.makedirs(args.out, exist_ok=True)
    print(f"re-scoring {len(dirs)} runs with {args.n_samples} samples (seed {args.seed}) -> {args.out}")
    for d in dirs:
        name = os.path.basename(os.path.normpath(d))
        jpath = os.path.join(args.out, name + '.json')
        if os.path.exists(jpath) and not args.force:
            print(f"  {name}: already scored (use --force)"); continue
        row = rescore(d, args.n_samples, args.seed, args.n_perm, grid=args.grid)
        with open(jpath, 'w', encoding='utf-8') as fh:
            json.dump(row, fh, indent=1)
    write_csv(args.out)
    print(f"written {os.path.join(args.out, 'rescore.csv')}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
