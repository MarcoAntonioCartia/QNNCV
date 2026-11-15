"""
Wigner functions and Wigner negativity of trained CV generators (eval-only;
no training, never writes into run folders).

For every run folder given, loads best_weights.npy, runs the circuit for z = 0
and for N_Z fixed-seed latent draws, and computes for each OUTPUT mode (0, 1)
the single-mode reduced state's Wigner function W(x, p) in the readout's
hbar = 1 units (same x as the Hermite readout: vacuum W = exp(-x^2-p^2)/pi),
its negative volume  delta = integral of max(-W, 0) dx dp  (Kenfack &
Zyczkowski 2004: delta = (integral |W| - 1) / 2), mean photon number and purity.

Reduced states of an entangled 3-mode state are mixed, so a zero single-mode
negativity does NOT mean the joint state is Gaussian or Wigner-positive; it is
a lower-level witness only. The reduced state is renormalised to trace 1 (the
ket norm inside cutoff is recorded).

    python scripts/wigner_eval.py "logs/qgan_2d_ring_*_seed*" [...]
    python scripts/wigner_eval.py --check <one run>     # validate against SF
Outputs: analysis/wigner/wigner.csv (one row per run) and
         analysis/wigner/grids/<run>.npz (z = 0: W for both modes, joint P(x1, x2)).
"""

import argparse
import contextlib
import csv
import glob
import io
import json
import os
import re
import sys
import time

import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np                                    # noqa: E402
import tensorflow as tf                               # noqa: E402

tf.get_logger().setLevel('ERROR')

from src.quantum.circuit import CVQGANGenerator, EncodingSpec   # noqa: E402
from src.quantum import weights_io                              # noqa: E402

OUT = os.path.join(REPO, 'analysis', 'wigner')
XW = np.linspace(-5.0, 5.0, 201)            # Wigner grid (x and p), hbar = 1
XP = np.linspace(-4.5, 4.5, 121)            # joint output density grid
N_Z, Z_SEED = 16, 123
RUN_RE = re.compile(r'^(qgan_2d_ring)_\d{8}_\d{6}_(.*)_seed(\d+)$')


def wigner(rho, xvec, pvec, hbar=1.0):
    """Single-mode Wigner function from a Fock density matrix: the iterative
    Laguerre recursion of strawberryfields BaseFockState.wigner, with hbar as a
    parameter. Returns W[p, x] (meshgrid(xvec, pvec) layout)."""
    c = rho.shape[0]
    Q, P = np.meshgrid(xvec, pvec)
    A = (Q + 1j * P) / (2 * np.sqrt(hbar / 2))
    Wl = [np.zeros(A.shape, complex) for _ in range(c)]
    Wl[0] = np.exp(-2.0 * np.abs(A) ** 2) / np.pi
    W = np.real(rho[0, 0]) * np.real(Wl[0])
    for n in range(1, c):
        Wl[n] = 2.0 * A * Wl[n - 1] / np.sqrt(n)
        W += 2 * np.real(rho[0, n] * Wl[n])
    for m in range(1, c):
        temp = Wl[m].copy()
        Wl[m] = (2 * np.conj(A) * temp - np.sqrt(m) * Wl[m - 1]) / np.sqrt(m)
        W += np.real(rho[m, m] * Wl[m])
        for n in range(m + 1, c):
            temp2 = (2 * A * Wl[n - 1] - np.sqrt(m) * temp) / np.sqrt(n)
            temp = Wl[n].copy()
            Wl[n] = temp2
            W += 2 * np.real(rho[m, n] * Wl[n])
    return W / hbar


def reduced(ket, mode):
    """Reduced density matrix of one mode from an n-mode ket (numpy)."""
    k = np.moveaxis(ket, mode, 0).reshape(ket.shape[mode], -1)
    rho = k @ k.conj().T
    return rho / np.trace(rho).real


def negativity(W, xvec, pvec):
    return float(np.sum(np.clip(-W, 0, None)) * (xvec[1] - xvec[0]) * (pvec[1] - pvec[0]))


def build(cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        g = CVQGANGenerator(n_modes=int(cfg['n_modes']), n_output_modes=2,
                            n_layers=int(cfg['n_layers']), cutoff_dim=int(cfg['cutoff_dim']),
                            use_kerr=not cfg.get('no_kerr', False),
                            latent_scale=float(cfg['latent_scale']), batch_size=None,
                            encoding=EncodingSpec('input_displacement'))
    return g


def ket_of(g, z):
    return np.asarray(g._run_circuit(tf.constant(z, tf.float32)).ket())


def mode_stats(ket):
    out = {}
    for m in (0, 1):
        rho = reduced(ket, m)
        W = wigner(rho, XW, XW)
        n = np.arange(rho.shape[0])
        out[m] = dict(W=W, neg=negativity(W, XW, XW), nbar=float(np.sum(n * np.diag(rho).real)),
                      purity=float(np.real(np.trace(rho @ rho))))
    return out


def run_one(run_dir):
    name = os.path.basename(os.path.normpath(run_dir))
    m = RUN_RE.match(name)
    cfg = json.load(open(os.path.join(run_dir, 'run_config.json')))
    if cfg.get('generator', 'cv') != 'cv' or cfg.get('encoding', 'input_displacement') != 'input_displacement':
        return None
    g = build(cfg)
    g.weights.assign(weights_io.load_weights(os.path.join(run_dir, 'best_weights.npy'))['weights'].astype(np.float32))
    rng = np.random.default_rng(Z_SEED)
    zs = [np.zeros(g.latent_dim, np.float32)] + list(rng.standard_normal((N_Z, g.latent_dim)).astype(np.float32))
    stats, norms = [], []
    for z in zs:
        ket = ket_of(g, z)
        norms.append(float(np.sum(np.abs(ket) ** 2)))
        stats.append(mode_stats(ket))
    s0 = stats[0]
    P0 = g.generate_distribution_2d(tf.constant(zs[0]), XP, XP).numpy()
    os.makedirs(os.path.join(OUT, 'grids'), exist_ok=True)
    np.savez_compressed(os.path.join(OUT, 'grids', f'{name}.npz'),
                        xw=XW, W0=s0[0]['W'].astype(np.float32), W1=s0[1]['W'].astype(np.float32),
                        xp=XP, P=P0.astype(np.float32))
    row = dict(run=name, group=f'{m.group(1)}_{m.group(2)}' if m else name, seed=cfg.get('seed'),
               n_z=N_Z, ket_norm_min=min(norms))
    for k in (0, 1):
        draws = stats[1:]
        row[f'neg{k}_z0'] = s0[k]['neg']
        row[f'neg{k}_mean'] = float(np.mean([s[k]['neg'] for s in draws]))
        row[f'neg{k}_max'] = float(np.max([s[k]['neg'] for s in draws]))
        row[f'nbar{k}_mean'] = float(np.mean([s[k]['nbar'] for s in draws]))
        row[f'purity{k}_mean'] = float(np.mean([s[k]['purity'] for s in draws]))
    return row


def check(run_dir):
    """Validate wigner()/reduced() against SF's own state.wigner (hbar = 2) and
    the p-marginal against the Hermite readout density."""
    cfg = json.load(open(os.path.join(run_dir, 'run_config.json')))
    g = build(cfg)
    g.weights.assign(weights_io.load_weights(os.path.join(run_dir, 'best_weights.npy'))['weights'].astype(np.float32))
    z = np.random.default_rng(0).standard_normal(g.latent_dim).astype(np.float32)
    state = g._run_circuit(tf.constant(z))
    ket = np.asarray(state.ket())
    tr = float(np.sum(np.abs(ket) ** 2))
    for m in (0, 1):
        Wm = wigner(reduced(ket, m), XW, XW)
        Wsf = np.asarray(state.wigner(m, XW * np.sqrt(2), XW * np.sqrt(2))) * 2 / tr   # hbar 2 -> 1
        print(f'mode {m}: max |W_ours - W_SF(rescaled)| = {np.max(np.abs(Wm - Wsf)):.2e} '
              f'(max |W| {np.max(np.abs(Wm)):.3f}); integral W = {Wm.sum() * (XW[1] - XW[0]) ** 2:.4f}')
    # x-marginal of mode 0 (integrate W over p) vs Hermite readout marginal over x2
    P = g.generate_distribution_2d(tf.constant(z), XW, XW).numpy()
    px_read = P.sum(axis=1) / (P.sum() * (XW[1] - XW[0]))
    px_w = wigner(reduced(ket, 0), XW, XW).sum(axis=0) * (XW[1] - XW[0])
    print(f'mode 0 x-marginal: max |from W - from readout| = {np.max(np.abs(px_w - px_read)):.2e} '
          f'(peak {px_read.max():.3f})')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    dirs = sorted({d for p in a.runs for d in glob.glob(p)
                   if os.path.exists(os.path.join(d, 'best_weights.npy'))})
    if a.check:
        check(dirs[0])
        return
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'wigner.csv')
    old = {}
    if os.path.exists(path):
        old = {r['run']: r for r in csv.DictReader(open(path, encoding='utf-8'))}
    t0 = time.time()
    for i, d in enumerate(dirs):
        row = run_one(d)
        if row is None:
            continue
        old[row['run']] = row
        print(f'[{i + 1}/{len(dirs)}] {row["run"]}  neg z0 {row["neg0_z0"]:.4f}/{row["neg1_z0"]:.4f}  '
              f'mean {row["neg0_mean"]:.4f}/{row["neg1_mean"]:.4f}  ({time.time() - t0:.0f} s)', flush=True)
    rows = list(old.values())
    keys = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print('written', os.path.relpath(path, REPO))


if __name__ == '__main__':
    main()
