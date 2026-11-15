"""
E3: a Riofrío-style classical *sampler* GAN at small parameter counts (standalone).

Tests hypothesis H1 of reports/Por que gana el clasico (15 Nov 2026): Riofrío et
al. (2024) compare quantum models that hold an explicit distribution over a grid
with a classical GAN whose generator maps noise to points; our classical baseline
instead writes the density directly. If a sampler at ~100-200 parameters fails
on the ring where our explicit density models succeed, the 100x parameter gap
reported by Riofrío measures "sampler vs density model", not "quantum vs
classical".

Setup, following Riofrío et al.'s classical baseline: generator = MLP from
8-dimensional standard normal noise to a point (x, y), 4 hidden layers (their
choice for the O and X sets), leaky ReLU; discriminator = small MLP on points;
standard non-saturating GAN loss (binary cross-entropy); Adam. Training data =
points drawn exactly from the target density (cell drawn from a fine 120 x 120
grid, uniform jitter inside the cell). Targets: our canonical ring (radius 1.15,
width 0.25, the target of the E1 single-target fits) and a thin ring (width
0.05), closer to Riofrío's O set.

Score: KL(target || histogram of 20,000 generated points) on 16 x 16 and on
32 x 32 over [-3, 3] (cell floor 1e-10, as scripts/kl_eval.py), plus marginal W1
in cells (src/metrics/wasserstein). A learning-rate ladder is run for every
size so the baseline is tuned, not a straw man.

    python scripts/sampler_gan.py                    # full E3 grid
    python scripts/sampler_gan.py --quick            # 1 seed, 1 size, 500 steps
Writes analysis/e3_sampler/e3.csv and prints medians per target, size and lr.
"""

import argparse
import csv
import itertools
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from src.metrics.wasserstein import compute_wasserstein_2d  # noqa: E402

XR = 3.0
TARGETS = {'canonical': (1.15, 0.25), 'thin': (1.15, 0.05)}


def ring_density(G, radius, width):
    x = np.linspace(-XR, XR, G)
    X, Y = np.meshgrid(x, x)
    d = np.exp(-0.5 * ((np.sqrt(X ** 2 + Y ** 2) - radius) / width) ** 2)
    return d / d.sum()


def sample_points(n, radius, width, rng, G=120):
    """Exact draws from the gridded target: a cell by its mass, then uniform jitter."""
    P = ring_density(G, radius, width).ravel()
    idx = rng.choice(P.size, size=n, p=P)
    i, j = np.divmod(idx, G)
    step = 2 * XR / (G - 1)
    x = -XR + j * step + rng.uniform(-step / 2, step / 2, n)
    y = -XR + i * step + rng.uniform(-step / 2, step / 2, n)
    return np.stack([x, y], 1).astype(np.float32)


def histogram(points, G):
    """[y, x] histogram over [-XR, XR]^2 with G cells per axis, as a density."""
    edges = np.linspace(-XR - XR / (G - 1), XR + XR / (G - 1), G + 1)
    H, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=[edges, edges])
    return H / max(H.sum(), 1.0)


def kl(q, p, floor=1e-10):
    q = np.clip(q, floor, None); q = q / q.sum()
    p = np.clip(p, floor, None); p = p / p.sum()
    return float((q * np.log(q / p)).sum())


def mlp(sizes, out, act):
    layers = [tf.keras.Input(shape=(sizes[0],))]
    for h in sizes[1:]:
        layers.append(tf.keras.layers.Dense(h))
        layers.append(tf.keras.layers.LeakyReLU(0.2) if act == 'leaky' else tf.keras.layers.Activation(act))
    layers.append(tf.keras.layers.Dense(out))
    return tf.keras.Sequential(layers)


def run(target, width_h, lr, seed, steps, batch=256, noise=8):
    tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)
    radius, width = TARGETS[target]
    data = sample_points(20000, radius, width, rng)
    G = mlp([noise] + [width_h] * 4, 2, 'leaky')
    D = mlp([2, 16, 16], 1, 'leaky')
    n_params = int(sum(int(np.prod(v.shape)) for v in G.trainable_variables))
    og = tf.keras.optimizers.Adam(lr, beta_1=0.5)
    od = tf.keras.optimizers.Adam(lr, beta_1=0.5)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    data_tf = tf.constant(data)

    @tf.function
    def step():
        idx = tf.random.uniform([batch], 0, data.shape[0], dtype=tf.int32)
        real = tf.gather(data_tf, idx)
        z = tf.random.normal([batch, noise])
        with tf.GradientTape() as td:
            fake = G(z, training=True)
            ld = bce(tf.ones([batch, 1]), D(real, training=True)) + bce(tf.zeros([batch, 1]), D(fake, training=True))
        od.apply_gradients(zip(td.gradient(ld, D.trainable_variables), D.trainable_variables))
        z = tf.random.normal([batch, noise])
        with tf.GradientTape() as tg:
            lg = bce(tf.ones([batch, 1]), D(G(z, training=True), training=True))   # non-saturating
        og.apply_gradients(zip(tg.gradient(lg, G.trainable_variables), G.trainable_variables))
        return ld, lg

    t0 = time.time()
    for _ in range(steps):
        step()
    pts = G(tf.random.normal([20000, noise]), training=False).numpy()
    out = {'target': target, 'width_h': width_h, 'n_params': n_params, 'lr': lr, 'seed': seed, 'steps': steps,
           'seconds': round(time.time() - t0, 1)}
    for g in (16, 32):
        T, H = ring_density(g, radius, width), histogram(pts, g)
        out[f'kl{g}'] = kl(T, H)
        out[f'w1_{g}'] = float(compute_wasserstein_2d(T, H)) if H.sum() > 0 else float('inf')
    out['frac_inside'] = float(np.mean((np.abs(pts) <= XR).all(1)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--out', default=os.path.join('analysis', 'e3_sampler'))
    a = ap.parse_args()
    widths = [3, 4, 6, 32]           # 4 hidden layers: ~71, ~106, ~194 params, and a ~3k reference
    lrs = [3e-4, 1e-3, 3e-3]
    seeds = range(5)
    targets = list(TARGETS)
    steps = a.steps
    if a.quick:
        widths, lrs, seeds, targets, steps = [4], [1e-3], range(1), ['canonical'], 500
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for target, w, lr, s in itertools.product(targets, widths, lrs, seeds):
        r = run(target, w, lr, s, steps)
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    path = os.path.join(a.out, 'e3_quick.csv' if a.quick else 'e3.csv')
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print('\nmedians over seeds (target, params, lr): kl16 / kl32 / w1_32')
    for (t, w, lr), grp in itertools.groupby(sorted(rows, key=lambda r: (r['target'], r['width_h'], r['lr'])),
                                             key=lambda r: (r['target'], r['width_h'], r['lr'])):
        g = list(grp)
        med = lambda k: statistics.median(r[k] for r in g)  # noqa: E731
        print(f"  {t:9s} {g[0]['n_params']:5d} lr {lr:g}: {med('kl16'):.3f} / {med('kl32'):.3f} / {med('w1_32'):.3f}")
    print('written', path)


if __name__ == '__main__':
    main()
