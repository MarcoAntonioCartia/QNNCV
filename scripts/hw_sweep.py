"""
Hardware-realizability sweep for trained CV generators (eval-only; no training).

Takes the frozen best_weights.npy of trained CV runs and asks how the learned
family degrades under the imperfections of a photonic realization, using the
SAME 200 fixed-seed latent draws as scripts/rescore.py, so every condition is
a paired comparison against the ideal circuit of the same checkpoint. Fidelity
and coverage marginal W1 and MMD^2 reproduce rescore.py's values for the ideal
row; the diversity ratio uses ALL pairs (rescore.py samples 300 pairs), and the
validation split is rebuilt like rescore.py but without its hash check.

Conditions
  ideal          the trained circuit as is (fidelity/coverage/MMD^2 = rescore.py's)
  eta_raw        homodyne detection efficiency eta on both output modes, raw
                 outcomes: x_out = sqrt(eta) x + sqrt(1-eta) x_vac (exact for a
                 pure-loss channel right before an ideal homodyne detector;
                 vacuum quadrature variance 1/2 in the hbar = 1 units of the
                 readout), computed analytically from the ideal density
  eta_cal        the same, with outcomes rescaled by 1/sqrt(eta) (a calibrated
                 detector): a Gaussian blur of variance (1-eta)/(2 eta)
  loss_db        propagation loss: a LossChannel of the given dB on every mode
                 after every layer (mixed-state simulation, density matrix)
  rcap           finite squeezing: every trained squeeze magnitude clipped to
                 |r| <= rcap (the front encoding has no squeezing)
  kerr           Kerr strengths scaled by the factor (0 = Gaussian-only circuit,
                 i.e. what a platform without a deterministic Kerr gate runs)

Also written per checkpoint: gate statistics in hardware units (squeezing in
dB, displacement magnitudes, Kerr phases, mean photon numbers) and the
Clements decomposition (rectangular_symmetric: symmetric Mach-Zehnder meshes
with internal/external phases) of every trained interferometer, verified by
reconstruction, plus an illustrative 12-bit DAC code per phase.

    python scripts/hw_sweep.py logs/qgan_2d_ring_*_m3_ep300_dlr0.0005_gp10_gs32_seed*
    python scripts/hw_sweep.py <runs> --quick        # 40 samples, fewer levels
Outputs: analysis/hw_sweep/{<run>.json, hw_sweep.csv, gates_<run>.csv}
"""

import argparse
import csv
import glob
import json
import os
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
import strawberryfields as sf                         # noqa: E402
from strawberryfields import ops                      # noqa: E402
from strawberryfields import decompositions as sfdec  # noqa: E402

from src.quantum.circuit import CVQGANGenerator, EncodingSpec   # noqa: E402
from src.quantum import weights_io                              # noqa: E402
from src.quantum.hermite import compute_hermite_basis           # noqa: E402
from src.families.registry import get_family                    # noqa: E402
from src.data.dataset import generate_dataset                   # noqa: E402
from src.training.qgan_2d import seed_everything                # noqa: E402

DB_PER_NEPER_R = 20.0 / np.log(10.0)       # squeezing in dB = 8.686 * r


# ---------------------------------------------------------------- metrics ---
def marginals(P):
    """(n, G, G) densities -> clipped, renormalised marginals (n, G) for both
    axes, exactly as src/metrics/wasserstein.compute_wasserstein_2d."""
    P = P / P.sum(axis=(1, 2), keepdims=True)
    out = []
    for ax in (1, 2):
        m = np.clip(P.sum(axis=ax), 1e-10, None)
        out.append(m / m.sum(axis=1, keepdims=True))
    return out


def w1_matrix(A, B):
    """Marginal W1 (grid cells) between every density of A (na,G,G) and B
    (nb,G,G): mean over the two axes of sum_i |F_a(i) - F_b(i)|, i < G-1,
    which is scipy's wasserstein_distance on integer positions."""
    out = 0.0
    for ma, mb in zip(marginals(A), marginals(B)):
        Fa, Fb = np.cumsum(ma, axis=1)[:, :-1], np.cumsum(mb, axis=1)[:, :-1]
        out = out + np.abs(Fa[:, None, :] - Fb[None, :, :]).sum(-1)
    return out / 2.0


def mmd2_unbiased(A, B):
    """Unbiased MMD^2, RBF on flattened sum-normalised densities, median
    bandwidth over the pooled sample (rescore.py's statistic, no p-value)."""
    Z = np.concatenate([A.reshape(len(A), -1), B.reshape(len(B), -1)]).astype(np.float64)
    Z = Z / Z.sum(1, keepdims=True)
    sq = (Z ** 2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0)
    K = np.exp(-D2 / (np.median(D2[np.triu_indices(len(Z), 1)]) + 1e-12))
    n, m = len(A), len(B)
    Kaa, Kbb, Kab = K[:n, :n], K[n:, n:], K[:n, n:]
    return float((Kaa.sum() - np.trace(Kaa)) / (n * (n - 1))
                 + (Kbb.sum() - np.trace(Kbb)) / (m * (m - 1)) - 2 * Kab.mean())


def score(fakes, val):
    W = w1_matrix(fakes, val)
    Wff = w1_matrix(fakes, fakes)
    Wvv = w1_matrix(val, val)
    iu_f, iu_v = np.triu_indices(len(fakes), 1), np.triu_indices(len(val), 1)
    return {'fid_w1': float(W.min(1).mean()), 'cov_w1': float(W.min(0).mean()),
            'div_ratio': float(Wff[iu_f].mean() / Wvv[iu_v].mean()),
            'mmd2': mmd2_unbiased(fakes, val)}


# ---------------------------------------------------------------- circuits ---
class LossyCVGenerator(CVQGANGenerator):
    """CVQGANGenerator with a LossChannel(T) on every mode after every layer.
    Same gates in the same order (golden-equivalent at T = 1), so the trained
    flat weight vector binds unchanged. Read out from the reduced density
    matrix of the two output modes."""

    def __init__(self, *, loss_T, **kw):
        self.loss_T = float(loss_T)
        super().__init__(**kw)

    def _build_symbolic_circuit(self):
        import itertools  # noqa: F401  (parent's uid counter is module-level)
        from src.quantum import circuit as _c
        self.prog = sf.Program(self.n_modes)
        uid = next(_c._GENERATOR_UID)
        self.latent_params = [self.prog.params(f"g{uid}_z_{i}")
                              for i in range(self.encoding.n_front_latents(self.n_modes))]
        self.weight_params = [self.prog.params(f"g{uid}_w_{i}") for i in range(self.total_params)]
        with self.prog.context as q:
            self.encoding.apply(q, self.latent_params, self.latent_scale, self.n_modes)
            w = 0
            for _ in range(self.n_layers):
                w = self._apply_interferometer(q, w)
                for m in range(self.n_modes):
                    ops.Sgate(self.weight_params[w], self.weight_params[w + 1]) | q[m]
                    w += 2
                w = self._apply_interferometer(q, w)
                for m in range(self.n_modes):
                    ops.Dgate(self.weight_params[w], self.weight_params[w + 1]) | q[m]
                    w += 2
                if self.use_kerr:
                    for m in range(self.n_modes):
                        ops.Kgate(self.weight_params[w]) | q[m]
                        w += 1
                for m in range(self.n_modes):
                    ops.LossChannel(self.loss_T) | q[m]

    def generate_batch(self, z_batch, xvec, yvec, return_ket_norm=False):
        if self.batch_size:                       # batched engine: one run
            rho = tf.cast(self._run_circuit(z_batch).reduced_dm([0, 1]), tf.complex64)
        else:                                     # unbatched engine (batch 1): one run per latent
            rho = tf.stack([tf.cast(self._run_circuit(z_batch[b]).reduced_dm([0, 1]), tf.complex64)
                            for b in range(int(z_batch.shape[0]))])
        # rho: (B, n, n', m, m')
        H = tf.cast(compute_hermite_basis(xvec, self.cutoff_dim), tf.complex64)
        Hy = tf.cast(compute_hermite_basis(yvec, self.cutoff_dim), tf.complex64)
        P = tf.math.real(tf.einsum('bnpmq,xn,xp,ym,yq->bxy', rho, H, H, Hy, Hy))
        P = P / (tf.reduce_sum(P, axis=[1, 2], keepdims=True) + 1e-10)
        tr = tf.math.real(tf.reduce_sum(tf.linalg.diag_part(tf.reshape(
            tf.transpose(rho, [0, 1, 3, 2, 4]), [-1, self.cutoff_dim ** 2, self.cutoff_dim ** 2])), axis=-1))
        return (P, tr) if return_ket_norm else P


def layout(g):
    """Flat-weight indices per layer, in the circuit's consumption order."""
    M, N, L = g.interferometer_params, g.n_modes, g.n_layers
    idx = []
    for l in range(L):
        b = l * g.params_per_layer
        idx.append({'U1': list(range(b, b + M)),
                    'S_r': [b + M + 2 * m for m in range(N)],
                    'S_phi': [b + M + 2 * m + 1 for m in range(N)],
                    'U2': list(range(b + M + 2 * N, b + 2 * M + 2 * N)),
                    'D_r': [b + 2 * M + 2 * N + 2 * m for m in range(N)],
                    'D_phi': [b + 2 * M + 2 * N + 2 * m + 1 for m in range(N)],
                    'K': [b + 2 * M + 4 * N + m for m in range(N)] if g.use_kerr else []})
    return idx


def interferometer_unitary(params, N):
    """Unitary of _apply_interferometer(params) via SF's Gaussian compiler
    (passive symplectic S in xxpp order -> U = S_xx + i S_px)."""
    prog = sf.Program(N)
    n_bs = N * (N - 1) // 2
    th, ph, rp = params[:n_bs], params[n_bs:2 * n_bs], params[2 * n_bs:]
    with prog.context as q:
        n = 0
        for layer in range(N):
            for k, (a, b) in enumerate(zip(q[:-1], q[1:])):
                if (layer + k) % 2 != 1:
                    ops.BSgate(float(th[n]), float(ph[n])) | (a, b)
                    n += 1
        for i in range(N - 1):
            ops.Rgate(float(rp[i])) | q[i]
    S = None
    for cmd in prog.compile(compiler='gaussian_unitary').circuit:
        if cmd.op.__class__.__name__ == 'GaussianTransform':
            S = np.asarray(cmd.op.p[0], float)
    if S is None:
        return np.eye(N, dtype=complex)
    return S[:N, :N] + 1j * S[N:, :N]


def clements(U):
    tlist, diag, _ = sfdec.rectangular_symmetric(U)
    N = U.shape[0]
    R = np.eye(N, dtype=complex)
    for (m, n, ip, ep, _n) in tlist:
        R = sfdec.mach_zehnder(int(m), int(n), ip, ep, N) @ R
    R = np.diag(diag) @ R
    err = float(np.abs(R - U).max())
    mzis = [{'modes': [int(m), int(n)], 'internal': float(ip % (2 * np.pi)),
             'external': float(ep % (2 * np.pi))} for (m, n, ip, ep, _n) in tlist]
    outph = [float(np.angle(d) % (2 * np.pi)) for d in diag]
    return mzis, outph, err


def dac12(phase):
    """Illustrative linear map of a phase in [0, 2 pi) to a 12-bit code
    (Pmod DA2, 0-3.3 V, assuming a shifter with a 2 pi swing over the range)."""
    return int(round((phase % (2 * np.pi)) / (2 * np.pi) * 4095))


# ------------------------------------------------------------------ driver ---
def build(cfg, weights, loss_T=None, batch_size=8):
    kw = dict(n_modes=int(cfg['n_modes']), n_output_modes=2, n_layers=int(cfg['n_layers']),
              cutoff_dim=int(cfg['cutoff_dim']), use_kerr=not cfg.get('no_kerr', False),
              latent_scale=float(cfg['latent_scale']), batch_size=batch_size,
              encoding=EncodingSpec('input_displacement'))
    g = LossyCVGenerator(loss_T=loss_T, **kw) if loss_T is not None else CVQGANGenerator(**kw)
    g.weights.assign(weights)
    return g


def draw(g, xvec, n, seed=0):
    tf.random.set_seed(seed)
    zs, out = [], []
    while len(out) < n:
        z = tf.random.normal([g.batch_size, g.latent_dim])
        p = g.generate_batch(z, xvec, xvec).numpy()
        for b in range(g.batch_size):
            if len(out) < n and np.isfinite(p[b]).all():
                out.append(p[b]); zs.append(z.numpy()[b])
    return np.array(out), np.array(zs)


LOSS_BATCH = 1


def draw_like(g, xvec, zs):
    """Densities for given latent vectors, g.batch_size at a time (the same
    latents as the ideal draw, so the comparison stays paired)."""
    out, bs = [], g.batch_size or 1
    for s in range(0, len(zs), bs):
        zb = zs[s:s + bs]
        pad = bs - len(zb)
        if pad:
            zb = np.concatenate([zb, np.repeat(zb[-1:], pad, 0)])
        out.append(g.generate_batch(tf.constant(zb, tf.float32), xvec, xvec).numpy()[:bs - pad])
    return np.concatenate(out)


def detect(Pfine, xf, xvec, eta, calibrated):
    """Analytic homodyne inefficiency on both output modes."""
    if calibrated:
        s2 = (1 - eta) / (2 * eta); K = np.exp(-(xvec[:, None] - xf[None, :]) ** 2 / (2 * s2))
    else:
        s2 = (1 - eta) / 2; K = np.exp(-(xvec[:, None] - np.sqrt(eta) * xf[None, :]) ** 2 / (2 * s2))
    out = np.einsum('ia,bac,jc->bij', K, Pfine, K)
    return out / out.sum(axis=(1, 2), keepdims=True)


def run_one(run_dir, n_samples, levels, out_dir, n_loss=80):
    name = os.path.basename(os.path.normpath(run_dir))
    cfg = json.load(open(os.path.join(run_dir, 'run_config.json')))
    if cfg.get('generator', 'cv') != 'cv' or cfg.get('encoding', 'input_displacement') != 'input_displacement':
        raise ValueError(f"{name}: hw_sweep handles plain CV runs only")
    G = int(cfg['grid_size'])
    seed_everything(int(cfg['seed']))
    fam = get_family(cfg['family'], G, 3.0)
    data = generate_dataset(fam, cfg['n_train'] + cfg['n_val'])
    val = np.array([m for m, _ in data[cfg['n_train']:]])
    xvec = fam.xvec
    w0 = weights_io.load_weights(os.path.join(run_dir, 'best_weights.npy'))['weights'].astype(np.float32)
    rows, t0 = [], time.time()

    g = build(cfg, w0)
    ideal, zs = draw(g, xvec, n_samples)
    rows.append({'condition': 'ideal', 'level': 0.0, **score(ideal, val)})

    # detection efficiency (analytic, from the ideal density on a fine wide grid)
    xf = np.linspace(-4.5, 4.5, 361)
    fine = []
    for s in range(0, n_samples, g.batch_size):
        zb = tf.constant(zs[s:s + g.batch_size])
        if zb.shape[0] < g.batch_size:
            zb = tf.concat([zb, tf.repeat(zb[-1:], g.batch_size - zb.shape[0], 0)], 0)
        fine.append(g.generate_batch(zb, xf, xf).numpy())
    fine = np.concatenate(fine)[:n_samples]
    for eta in levels['eta']:
        rows.append({'condition': 'eta_raw', 'level': eta, **score(detect(fine, xf, xvec, eta, False), val)})
        rows.append({'condition': 'eta_cal', 'level': eta, **score(detect(fine, xf, xvec, eta, True), val)})

    lay = layout(g)
    # finite squeezing
    for rc in levels['rcap']:
        w = w0.copy()
        for L in lay:
            w[L['S_r']] = np.clip(w[L['S_r']], -rc, rc)
        g.weights.assign(w)
        rows.append({'condition': 'rcap', 'level': rc, **score(draw(g, xvec, n_samples)[0], val)})
    # Kerr scaling (0 = Gaussian-only circuit)
    for f in levels['kerr']:
        w = w0.copy()
        for L in lay:
            w[L['K']] = w[L['K']] * f
        g.weights.assign(w)
        rows.append({'condition': 'kerr', 'level': f, **score(draw(g, xvec, n_samples)[0], val)})
    g.weights.assign(w0)

    # propagation loss (mixed state, ~17 s per batch of 8): fewer samples, so
    # an ideal reference on the same first n_loss draws ('ideal_nloss') is the
    # paired baseline (coverage depends on the number of generated samples)
    rows.append({'condition': 'ideal_nloss', 'level': 0.0, **score(ideal[:n_loss], val)})
    # Memory: the mixed-state engine holds ~2 GB per batch element at
    # 3 modes / cutoff 12 (an 8-wide batch reached ~18 GB and crashed jobs on
    # 15 Nov), so the lossy engine runs 1 sample at a time (~5 GB peak; 2 at a time measured 10 GB) and is freed
    # before the next level. The draws use the same seed-0 latent stream.
    import gc
    for db in levels['loss_db']:
        gl = build(cfg, w0, loss_T=10 ** (-db / 10), batch_size=LOSS_BATCH)
        rows.append({'condition': 'loss_db', 'level': db, **score(draw_like(gl, xvec, zs[:n_loss]), val)})
        del gl
        gc.collect()

    # gate statistics and Clements export
    N = g.n_modes
    r = np.concatenate([np.abs(w0[L['S_r']]) for L in lay])
    d = np.concatenate([np.abs(w0[L['D_r']]) for L in lay])
    k = np.concatenate([np.abs(w0[L['K']]) for L in lay]) if g.use_kerr else np.zeros(1)
    nvec = np.arange(g.cutoff_dim)
    acc = np.zeros(N)
    for s0 in range(0, n_samples - n_samples % g.batch_size, g.batch_size):
        ket = g._run_circuit(tf.constant(zs[s0:s0 + g.batch_size])).ket()
        prob = tf.abs(tf.cast(ket, tf.complex64)).numpy() ** 2
        acc += [np.einsum(f"b{'ijk'[:N]},{'ijk'[m]}->", prob, nvec) for m in range(N)]
    # mean photon number per mode inside the cutoff, over all ideal draws
    mean_n = [float(a / (n_samples - n_samples % g.batch_size)) for a in acc]
    stats = {'squeeze_r_max': float(r.max()), 'squeeze_r_median': float(np.median(r)),
             'squeeze_db_max': float(r.max() * DB_PER_NEPER_R),
             'squeeze_db_median': float(np.median(r) * DB_PER_NEPER_R),
             'displacement_max': float(d.max()), 'displacement_median': float(np.median(d)),
             'front_alpha_p95': float(np.percentile(np.abs(zs[:, 0::2]) * float(cfg['latent_scale']), 95)),
             'kerr_max': float(k.max()), 'kerr_median': float(np.median(k)),
             'mean_photons_per_mode': mean_n}
    gates, max_err = [], 0.0
    for l, L in enumerate(lay):
        entry = {'layer': l}
        for u in ('U1', 'U2'):
            mz, outph, err = clements(interferometer_unitary(w0[L[u]], N))
            max_err = max(max_err, err)
            entry[u] = {'mzi': mz, 'output_phases': outph}
        entry['S'] = [{'mode': m, 'r': float(w0[L['S_r'][m]]), 'phi': float(w0[L['S_phi'][m]]),
                       'db': float(abs(w0[L['S_r'][m]]) * DB_PER_NEPER_R)} for m in range(N)]
        entry['D'] = [{'mode': m, 'r': float(w0[L['D_r'][m]]), 'phi': float(w0[L['D_phi'][m]])}
                      for m in range(N)]
        entry['K'] = [{'mode': m, 'kappa': float(w0[L['K'][m]])} for m in range(N)] if g.use_kerr else []
        gates.append(entry)
    stats['clements_max_reconstruction_error'] = max_err

    with open(os.path.join(out_dir, f'gates_{name}.csv'), 'w', newline='', encoding='utf-8') as fh:
        wr = csv.writer(fh)
        wr.writerow(['layer', 'element', 'modes', 'parameter', 'value_rad_or_r', 'dB', 'dac12_illustrative'])
        for e in gates:
            for u in ('U1', 'U2'):
                for i, mz in enumerate(e[u]['mzi']):
                    for kind in ('internal', 'external'):
                        wr.writerow([e['layer'], f'{u}.MZI{i}', '-'.join(map(str, mz['modes'])),
                                     f'phase_{kind}', round(mz[kind], 6), '', dac12(mz[kind])])
                for m, ph in enumerate(e[u]['output_phases']):
                    wr.writerow([e['layer'], f'{u}.out', m, 'phase', round(ph, 6), '', dac12(ph)])
            for s in e['S']:
                wr.writerow([e['layer'], 'S', s['mode'], 'r', round(s['r'], 6), round(s['db'], 3), ''])
            for s in e['D']:
                wr.writerow([e['layer'], 'D', s['mode'], 'alpha', round(s['r'], 6), '', ''])
            for s in e['K']:
                wr.writerow([e['layer'], 'K', s['mode'], 'kappa', round(s['kappa'], 6), '', ''])

    res = {'run': name, 'seed': cfg['seed'], 'n_samples': n_samples, 'rows': rows,
           'stats': stats, 'gates': gates, 'seconds': round(time.time() - t0, 1)}
    json.dump(res, open(os.path.join(out_dir, f'{name}.json'), 'w'), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--n-samples', type=int, default=200)
    ap.add_argument('--quick', action='store_true', help='40 samples, two levels per condition')
    ap.add_argument('--out', default=os.path.join('analysis', 'hw_sweep'))
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--n-loss', type=int, default=80, help='samples for the mixed-state loss conditions')
    a = ap.parse_args()
    levels = {'eta': [0.99, 0.95, 0.90, 0.80, 0.70, 0.50], 'rcap': [0.30, 0.20, 0.10, 0.05],
              'kerr': [0.5, 0.0], 'loss_db': [0.05, 0.1, 0.2, 0.5]}
    n = a.n_samples
    if a.quick:
        n = 40
        levels = {k: v[:2] for k, v in levels.items()}
    os.makedirs(a.out, exist_ok=True)
    runs = sorted({r for pat in a.runs for r in (glob.glob(pat) or [pat])})
    for rd in runs:
        name = os.path.basename(os.path.normpath(rd))
        if not a.force and os.path.exists(os.path.join(a.out, f'{name}.json')):
            continue
        res = run_one(rd, n, levels, a.out, n_loss=min(a.n_loss, n))
        ideal = res['rows'][0]
        print(f"{name}: ideal fid {ideal['fid_w1']:.3f} cov {ideal['cov_w1']:.3f} | "
              f"r_max {res['stats']['squeeze_db_max']:.2f} dB | kerr_max {res['stats']['kerr_max']:.3f} | "
              f"clements err {res['stats']['clements_max_reconstruction_error']:.1e} | {res['seconds']} s", flush=True)
    with open(os.path.join(a.out, 'hw_sweep.csv'), 'w', newline='', encoding='utf-8') as fh:
        wr = None
        for f in sorted(glob.glob(os.path.join(a.out, 'qgan_*.json'))):
            r = json.load(open(f))
            for row in r['rows']:
                out = {'run': r['run'], 'seed': r['seed'], 'n_samples': r['n_samples'], **row}
                if wr is None:
                    wr = csv.DictWriter(fh, fieldnames=list(out)); wr.writeheader()
                wr.writerow(out)
    print("written", os.path.join(a.out, 'hw_sweep.csv'))


if __name__ == '__main__':
    main()
