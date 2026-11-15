"""
Trainability diagnostics for the CV generator (eval-only: never trains,
never writes into run folders).

Question (15 Nov 2026): the CV generator stalls while its gradients GROW and
the critic's gap widens. Is it (a) a barren plateau, (b) a capacity ceiling,
(c) adversarial non-convergence, or (d) Fock-cutoff truncation?

Cost used throughout = the E1 supervised objective, exactly as the trainer
computes it: mean over latents of KL(canonical ring || generated density)
(src/training/qgan_2d.py, generator_epoch), plus, separately, the trainer's
ket-norm penalty 20 * mean((1 - ket_norm)^2). For GAN runs the critic is not
saved, so this cost is a proxy for their objective; for E1 runs it is the
objective.

Modes
  cutoff  (D4) re-evaluate trained weights at cutoff 12, 16, 20 on fixed
          latents: ket norm, TV and W1 between the cutoff-12 and cutoff-20
          densities, KL to the canonical ring at each cutoff.
  grad    (D2) gradient at the trained weights (and at a fresh init) split by
          gate type and layer; global norm before clipping; norm of and
          cosine between the supervised and ket-penalty gradients.
  slices  (D3) cost along the descent direction and along random unit
          directions around the trained weights (E1 runs).
  variance (D1) gradient variance over random initialisations vs layers,
          modes, cutoff and init width (the barren-plateau test of McClean
          et al. 2018).

    python scripts/diag_trainability.py cutoff  <run dirs...>
    python scripts/diag_trainability.py grad    <run dirs...>
    python scripts/diag_trainability.py slices  <run dirs...>
    python scripts/diag_trainability.py variance --configs 0-19 [--n-init 100]
Writes analysis/trainability/<mode>[_<tag>].csv (+ .npz for variance).
"""

import argparse
import csv
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

# The gradient of the real loss flows through complex amplitudes and TF warns
# on every complex64 -> float32 cast back to the real weights (same as during
# training; the real part is the correct derivative). Keep the logs readable.
tf.get_logger().setLevel('ERROR')

from src.quantum.circuit import CVQGANGenerator, EncodingSpec  # noqa: E402
from src.quantum import weights_io  # noqa: E402
from src.families.registry import get_family  # noqa: E402
from src.metrics.wasserstein import compute_wasserstein_2d  # noqa: E402

OUT = os.path.join('analysis', 'trainability')
EPS = 1e-10
KET_W = 20.0
GROUPS = ['bs_theta', 'bs_phi', 'rot', 'sq_r', 'sq_phi', 'disp_r', 'disp_phi', 'kerr']


def gate_groups(N, L, kerr=True):
    """Parameter indices per gate type and per layer, in CIRCUIT CONSUMPTION
    order (src/quantum/circuit.py _build_symbolic_circuit): per layer
    [int1 | (r, phi) x N squeeze | int2 | (r, phi) x N displacement | kerr x N],
    an interferometer being [BS theta x nbs | BS phi x nbs | R x (N-1)]."""
    nbs, Mi = N * (N - 1) // 2, N * (N - 1) + N - 1
    ppl = 2 * Mi + 4 * N + (N if kerr else 0)
    g = {k: [] for k in GROUPS}
    layer = np.zeros(ppl * L, int)
    for l in range(L):
        b = l * ppl
        layer[b:b + ppl] = l
        for off in (0, Mi + 2 * N):
            s = b + off
            g['bs_theta'] += range(s, s + nbs)
            g['bs_phi'] += range(s + nbs, s + 2 * nbs)
            g['rot'] += range(s + 2 * nbs, s + Mi)
        s = b + Mi
        g['sq_r'] += range(s, s + 2 * N, 2)
        g['sq_phi'] += range(s + 1, s + 2 * N, 2)
        s = b + 2 * Mi + 2 * N
        g['disp_r'] += range(s, s + 2 * N, 2)
        g['disp_phi'] += range(s + 1, s + 2 * N, 2)
        if kerr:
            g['kerr'] += range(b + 2 * Mi + 4 * N, b + ppl)
    return {k: np.array(v, int) for k, v in g.items()}, layer, ppl


def build(N, L, cutoff, latent_scale=0.3):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        g = CVQGANGenerator(n_modes=N, n_output_modes=2, n_layers=L, cutoff_dim=cutoff,
                            use_kerr=True, latent_scale=latent_scale, batch_size=None,
                            encoding=EncodingSpec('input_displacement'))
    grp, _, ppl = gate_groups(N, L)
    assert ppl * L == g.total_params, (ppl, L, g.total_params)
    assert sum(len(v) for v in grp.values()) == g.total_params
    return g


def target(grid):
    fam = get_family('ring', grid, 3.0)
    C, _ = fam.get_canonical()
    return fam, tf.constant(np.asarray(C), tf.float32)


def forward(g, zs, fam):
    ps, ks = [], []
    for z in zs:
        p, k = g.generate_distribution_2d(z, fam.xvec, fam.yvec, return_ket_norm=True)
        ps.append(p)
        ks.append(k)
    return tf.stack(ps), tf.stack(ks)


def sup_cost(P, T):
    return tf.reduce_mean(tf.reduce_sum(T * (tf.math.log(T + EPS) - tf.math.log(P + EPS)), axis=[-2, -1]))


def ket_pen(K):
    return KET_W * tf.reduce_mean(tf.square(1.0 - K))


def latents(n, dim, seed=0):
    return [tf.constant(v, tf.float32) for v in np.random.default_rng(seed).standard_normal((n, dim))]


def cfg_of(rd):
    return json.load(open(os.path.join(rd, 'run_config.json')))


def write(rows, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print('written', path, flush=True)


# ---------------------------------------------------------------- D4
def mode_cutoff(runs, n_z, cutoffs=(12, 16, 20)):
    rows = []
    for rd in runs:
        c = cfg_of(rd)
        fam, T = target(int(c['grid_size']))
        W = weights_io.load_weights(os.path.join(rd, 'best_weights.npy'))
        out = {}
        for cut in cutoffs:
            g = build(c['n_modes'], c['n_layers'], cut, c['latent_scale'])
            weights_io.apply_weights(g, W, source=rd)
            zs = latents(n_z, g.latent_dim)
            P, K = forward(g, zs, fam)
            out[cut] = (P.numpy(), K.numpy(), float(sup_cost(P, T)))
        P12, P20 = out[cutoffs[0]][0], out[cutoffs[-1]][0]
        tv = 0.5 * np.abs(P12 - P20).sum(axis=(1, 2))
        w1 = [float(compute_wasserstein_2d(a, b)) for a, b in zip(P12, P20)]
        row = {'run': os.path.basename(rd), 'seed': c['seed'], 'supervised': c.get('supervised_weight', 0) >= 1,
               'tv_12_20_median': float(np.median(tv)), 'tv_12_20_max': float(tv.max()),
               'w1_12_20_median': float(np.median(w1))}
        for cut in cutoffs:
            row[f'ket_norm_{cut}_median'] = float(np.median(out[cut][1]))
            row[f'ket_norm_{cut}_min'] = float(out[cut][1].min())
            row[f'kl_canon_{cut}'] = out[cut][2]
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return rows


# ---------------------------------------------------------------- D2
def grads(g, zs, fam, T):
    with tf.GradientTape(persistent=True) as tape:
        P, K = forward(g, zs, fam)
        s, k = sup_cost(P, T), ket_pen(K)
    gs = tape.gradient(s, g.weights).numpy()
    gk = tape.gradient(k, g.weights).numpy()
    del tape
    return float(s), float(k), gs, gk, K.numpy()


def summarise_grad(prefix, w, gs, gk, grp, layer, L):
    gt = gs + gk
    row = {f'{prefix}_norm_sup': float(np.linalg.norm(gs)), f'{prefix}_norm_ket': float(np.linalg.norm(gk)),
           f'{prefix}_norm_total': float(np.linalg.norm(gt)),
           f'{prefix}_cos_sup_ket': float(gs @ gk / (np.linalg.norm(gs) * np.linalg.norm(gk) + 1e-30))}
    for k, idx in grp.items():
        row[f'{prefix}_rms_g_{k}'] = float(np.sqrt(np.mean(gt[idx] ** 2)))
        row[f'{prefix}_rms_w_{k}'] = float(np.sqrt(np.mean(w[idx] ** 2)))
    for l in range(L):
        row[f'{prefix}_norm_layer{l}'] = float(np.linalg.norm(gt[layer == l]))
    return row


def mode_grad(runs, n_z):
    rows = []
    for rd in runs:
        c = cfg_of(rd)
        fam, T = target(int(c['grid_size']))
        N, L = c['n_modes'], c['n_layers']
        grp, layer, _ = gate_groups(N, L)
        g = build(N, L, c['cutoff_dim'], c['latent_scale'])
        zs = latents(n_z, g.latent_dim)
        row = {'run': os.path.basename(rd), 'seed': c['seed'], 'supervised': c.get('supervised_weight', 0) >= 1,
               'g_lr': c.get('g_lr'), 'clip': c.get('g_grad_clip')}
        tf.random.set_seed(int(c['seed']))                      # fresh init like the run's own (std 0.1)
        g.weights.assign(g._init_weights(0.1, 0.1))
        s, k, gs, gk, K = grads(g, zs, fam, T)
        row.update(init_sup=s, init_ket=k, **summarise_grad('init', g.weights.numpy(), gs, gk, grp, layer, L))
        weights_io.apply_weights(g, weights_io.load_weights(os.path.join(rd, 'best_weights.npy')), source=rd)
        s, k, gs, gk, K = grads(g, zs, fam, T)
        row.update(trained_sup=s, trained_ket=k, trained_ket_norm_min=float(K.min()),
                   **summarise_grad('trained', g.weights.numpy(), gs, gk, grp, layer, L))
        rows.append(row)
        print(row['run'], {k: round(v, 4) for k, v in row.items()
                           if isinstance(v, float) and ('norm_' in k[:13] or k.endswith(('_sup', '_ket', 'cos_sup_ket')))},
              flush=True)
    return rows


# ---------------------------------------------------------------- D3
def mode_slices(runs, n_z, n_dir=4, ts=np.linspace(-0.5, 0.5, 11)):
    rows = []
    for rd in runs:
        c = cfg_of(rd)
        fam, T = target(int(c['grid_size']))
        g = build(c['n_modes'], c['n_layers'], c['cutoff_dim'], c['latent_scale'])
        weights_io.apply_weights(g, weights_io.load_weights(os.path.join(rd, 'best_weights.npy')), source=rd)
        zs = latents(n_z, g.latent_dim)
        w0 = g.weights.numpy().copy()
        _, _, gs, gk, _ = grads(g, zs, fam, T)
        gt = gs + gk
        rng = np.random.default_rng(1)
        dirs = [('descent', -gt / np.linalg.norm(gt))]
        dirs += [(f'random{i}', v / np.linalg.norm(v)) for i, v in enumerate(rng.standard_normal((n_dir, w0.size)))]
        for name, d in dirs:
            for t in ts:
                g.weights.assign(w0 + t * d)
                P, K = forward(g, zs, fam)
                rows.append({'run': os.path.basename(rd), 'seed': c['seed'], 'direction': name, 't': float(t),
                             'sup': float(sup_cost(P, T)), 'ket': float(ket_pen(K))})
        g.weights.assign(w0)
        base = [r for r in rows if r['run'] == os.path.basename(rd) and r['t'] == 0.0][0]['sup']
        best = {n: min((r['sup'], r['t']) for r in rows if r['run'] == os.path.basename(rd) and r['direction'] == n)
                for n, _ in dirs}
        print(os.path.basename(rd), 'sup at 0:', round(base, 4), {n: (round(v, 4), t) for n, (v, t) in best.items()},
              flush=True)
    return rows


# ---------------------------------------------------------------- D1
def variance_configs():
    cfgs = []
    for sd in (0.1, 1.0):
        for L in (1, 2, 4, 6, 8, 10):
            cfgs.append(dict(N=3, L=L, cutoff=12, sd=sd))
        for N in (2, 4):
            cfgs.append(dict(N=N, L=6, cutoff=12, sd=sd))
        for cut in (8, 16):
            cfgs.append(dict(N=3, L=6, cutoff=cut, sd=sd))
    return cfgs


def mode_variance(idx, n_init, grid=32):
    cfgs = variance_configs()
    fam, T = target(grid)
    rows = []
    for i in idx:
        cf = cfgs[i]
        g = build(cf['N'], cf['L'], cf['cutoff'])
        grp, layer, _ = gate_groups(cf['N'], cf['L'])
        z = latents(1, g.latent_dim, seed=123)
        G, S, KN = [], [], []
        t0 = time.time()
        rng = np.random.default_rng(1000 + i)
        for _ in range(n_init):
            g.weights.assign(rng.normal(0.0, cf['sd'], g.total_params).astype(np.float32))
            s, k, gs, gk, K = grads(g, z, fam, T)
            G.append(gs)
            S.append(s)
            KN.append(K[0])
        G = np.array(G)
        var = G.var(0)
        row = dict(config=i, **cf, n_params=g.total_params, n_init=n_init, seconds=round(time.time() - t0, 1),
                   var_mean=float(var.mean()), var_median=float(np.median(var)), cost_mean=float(np.mean(S)),
                   cost_std=float(np.std(S)), ket_norm_median=float(np.median(KN)), ket_norm_min=float(np.min(KN)))
        for k, ix in grp.items():
            row[f'var_{k}'] = float(var[ix].mean())
        for l in range(cf['L']):
            row[f'var_layer{l}'] = float(var[layer == l].mean())
        rows.append(row)
        os.makedirs(OUT, exist_ok=True)
        np.savez(os.path.join(OUT, f'variance_cfg{i:02d}.npz'), grads=G, cost=np.array(S), ket=np.array(KN), **cf)
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items() if not k.startswith('var_layer')},
              flush=True)
    return rows


def parse_range(s):
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['cutoff', 'grad', 'slices', 'variance'])
    ap.add_argument('runs', nargs='*')
    ap.add_argument('--n-z', type=int, default=16)
    ap.add_argument('--configs', default='0-19')
    ap.add_argument('--n-init', type=int, default=100)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    runs = sorted({r for pat in a.runs for r in (glob.glob(pat) or [pat])
                   if os.path.exists(os.path.join(r, 'best_weights.npy'))})
    tag = f'_{a.tag}' if a.tag else ''
    if a.mode == 'cutoff':
        write(mode_cutoff(runs, a.n_z), f'cutoff{tag}.csv')
    elif a.mode == 'grad':
        write(mode_grad(runs, a.n_z), f'grad{tag}.csv')
    elif a.mode == 'slices':
        write(mode_slices(runs, a.n_z), f'slices{tag}.csv')
    else:
        write(mode_variance(parse_range(a.configs), a.n_init), f'variance{tag}.csv')


if __name__ == '__main__':
    main()
