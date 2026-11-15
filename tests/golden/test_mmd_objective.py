"""
Structural suite for the Born-machine (MMD) objective (no golden data):

- density kernel: MMDTrainer._adversarial_loss equals an independent NumPy
  computation of the unbiased MMD^2 with k = sum_b exp(-d^2 / (b^2 m)), m the
  median squared distance over the training set, on the same real batch;
- sample kernel: equals the brute-force unbiased U-statistic over densities,
  sum_{b!=b'} p_b^T K p_b' / (B(B-1)) + sum_{r!=r'} q_r^T K q_r' / (R(R-1))
  - 2 sum_{b,r} p_b^T K q_r / (B R), K the grid-point kernel matrix;
- unbiasedness: a generator drawing from the training family scores 0 in
  expectation for both kernels (the old V-statistic failed this);
- the loss is differentiable w.r.t. the generated densities (non-zero grad);
- finite shots (sample kernel): 1e7 shots within 5 % of exact, 20 shots
  differ, and the gradient passes the shot sampler;
- guards: batch_size < 2, mmd_real_batch < 2 and an unknown kernel raise;
  the CLI parser exposes --objective with default 'wgan_gp' (default path
  unchanged; the e2e goldens pin it numerically).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import load_t2q, main_dispatch

G = 8


def make(mod, kernel, batch=4, real=6):
    import numpy as np
    from src.training.qgan_2d import MMDTrainer
    rng = np.random.default_rng(3)
    train = []
    for _ in range(20):
        d = rng.random((G, G)) + 0.05
        train.append((d / d.sum(), {}))
    xv = np.linspace(-3.0, 3.0, G)
    return MMDTrainer(mmd_kernel=kernel, mmd_bandwidths=(0.5, 1.0, 2.0), mmd_real_batch=real,
                      generator=None, discriminator=None, g_optimizer=None, d_optimizer=None,
                      train_set=train, fixed_real_tf=None, blur_kernel=None, xvec=xv, yvec=xv,
                      batch_size=batch, n_critic=5, gp_weight=5.0, gp_warmup=50,
                      instance_noise=0.0, noise_anneal=1, noise_floor=0.0,
                      supervised_weight=0.0, supervised_warmup=1,
                      ket_penalty_weight=0.0, g_grad_clip=0.0)


def test():
    import numpy as np
    import tensorflow as tf
    mod = load_t2q()
    rng = np.random.default_rng(7)
    gen = rng.random((4, G, G)) + 0.01
    gen = gen / gen.sum(axis=(1, 2), keepdims=True)

    # --- density kernel vs NumPy reference ------------------------------
    tr = make(mod, 'density')
    np.random.seed(11)
    idx = np.random.randint(len(tr.train_set), size=tr.mmd_real_batch)
    np.random.seed(11)
    got = float(tr._adversarial_loss(tf.constant(gen, tf.float32)))
    T = np.stack([m.ravel() / m.sum() for m, _ in tr.train_set]).astype(np.float64)
    D2t = ((T[:, None, :] - T[None, :, :]) ** 2).sum(-1)
    m = np.median(D2t[np.triu_indices(len(T), 1)])
    k = lambda A, B: sum(np.exp(-((A[:, None] - B[None]) ** 2).sum(-1) / (b * b * m))  # noqa: E731
                         for b in (0.5, 1.0, 2.0))
    F, R = gen.reshape(4, -1), T[idx]
    Kff, Krr, Kfr = k(F, F), k(R, R), k(F, R)
    ref = ((Kff.sum() - np.trace(Kff)) / (4 * 3) + (Krr.sum() - np.trace(Krr)) / (6 * 5)
           - 2 * Kfr.mean())
    assert abs(got - ref) < 1e-4 * max(1.0, abs(ref)), f"density MMD {got} != reference {ref}"
    print(f"  ok: density MMD^2 matches NumPy reference ({got:.6f})")

    # --- sample kernel vs brute force -----------------------------------
    ts = make(mod, 'sample')
    np.random.seed(5)
    idx = np.random.randint(len(ts.train_set), size=ts.mmd_real_batch)
    np.random.seed(5)
    got = float(ts._adversarial_loss(tf.constant(gen, tf.float32)))
    X, Y = np.meshgrid(ts.xvec, ts.yvec)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    d2 = ((P[:, None] - P[None]) ** 2).sum(-1)
    K = sum(np.exp(-d2 / (2 * (b * 0.5) ** 2)) for b in (0.5, 1.0, 2.0))
    Fg, Rg = gen.reshape(4, -1), T[idx]
    ks = lambda A, B: A @ K @ B.T  # noqa: E731
    Gff, Grr, Gfr = ks(Fg, Fg), ks(Rg, Rg), ks(Fg, Rg)
    ref = ((Gff.sum() - np.trace(Gff)) / (4 * 3) + (Grr.sum() - np.trace(Grr)) / (6 * 5)
           - 2 * Gfr.mean())
    assert abs(got - ref) < 1e-4 * max(1.0, abs(ref)), f"sample MMD {got} != reference {ref}"
    print(f"  ok: sample MMD^2 matches the brute-force U-statistic ({got:.3e})")

    # --- unbiasedness: a 'perfect' generator (fakes drawn from the same
    # family as the reals) must score 0 in expectation for BOTH kernels; the
    # earlier V-statistic had a positive (1/B) tr(K Cov) offset that rewarded
    # collapse, and this check catches it.
    for kern, trk in (('density', tr), ('sample', ts)):
        vals = []
        for _ in range(300):
            gi = np.random.randint(len(trk.train_set), size=4)
            fake = np.stack([trk.train_set[i][0] for i in gi]).astype(np.float32)
            vals.append(float(trk._adversarial_loss(tf.constant(fake))))
        m, se = float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals)))
        assert abs(m) < 4 * se + 1e-9, f"{kern}: perfect generator scores {m:.3e} +- {se:.1e} (biased)"
        print(f"  ok: {kern} kernel unbiased (perfect generator {m:.2e} +- {se:.1e})")

    # --- differentiable ----------------------------------------------------
    v = tf.Variable(gen, dtype=tf.float32)
    with tf.GradientTape() as tape:
        loss = tr._adversarial_loss(v)
    gnorm = float(tf.norm(tape.gradient(loss, v)))
    assert np.isfinite(gnorm) and gnorm > 0, f"gradient norm {gnorm}"
    print(f"  ok: gradient reaches the densities (|grad| = {gnorm:.3e})")

    # --- finite shots: noisy value, exact-density gradient path ---------
    tshot = make(mod, 'sample')
    tshot.mmd_shots = 10 ** 7
    np.random.seed(5)
    big = float(tshot._adversarial_loss(tf.constant(gen, tf.float32)))
    np.random.seed(5)
    exact = float(ts._adversarial_loss(tf.constant(gen, tf.float32)))
    assert abs(big - exact) < 0.05 * abs(exact), f"1e7 shots {big} vs exact {exact} (>5% apart)"
    tshot.mmd_shots = 20
    np.random.seed(5)
    small = float(tshot._adversarial_loss(tf.constant(gen, tf.float32)))
    assert abs(small - exact) > 1e-3 * abs(exact), "20 shots identical to exact: shot path inactive"
    tshot.mmd_shots = 50
    with tf.GradientTape() as tape:
        loss = tshot._adversarial_loss(v)
    g50 = tape.gradient(loss, v)
    assert g50 is not None and np.isfinite(float(tf.norm(g50))), "no gradient through shots"
    print(f"  ok: 1e7 shots within 5% of exact ({big:.4e} vs {exact:.4e}); 20 shots differ; "
          f"gradient passes the shot sampler")

    # --- guards and CLI default ----------------------------------------
    for kw, what in ((dict(batch=1), 'batch_size 1'), (dict(real=1), 'real batch 1')):
        try:
            make(mod, 'density', **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{what} did not raise")
    try:
        make(mod, 'bogus')
    except ValueError:
        pass
    else:
        raise AssertionError("unknown kernel did not raise")
    from src.training.qgan_2d import MMDTrainer
    for bad in ((), (0.0, 1.0), (-1.0,)):
        try:
            tbad = make(mod, 'density')
            MMDTrainer.__init__(tbad, mmd_bandwidths=bad, **{k: getattr(tbad, k) for k in (
                'generator', 'discriminator', 'g_optimizer', 'd_optimizer', 'train_set',
                'fixed_real_tf', 'blur_kernel', 'xvec', 'yvec', 'batch_size', 'n_critic',
                'gp_weight', 'gp_warmup', 'instance_noise', 'noise_anneal', 'noise_floor',
                'supervised_weight', 'supervised_warmup', 'ket_penalty_weight', 'g_grad_clip')})
        except ValueError:
            continue
        raise AssertionError(f"bandwidths {bad} did not raise")
    parser = mod.build_parser()
    assert parser.get_default('objective') == 'wgan_gp', parser.get_default('objective')
    print("  ok: guards raise; --objective defaults to wgan_gp")


def generate(force=False):
    # Structural test — no golden data to generate; run the checks instead.
    test()


if __name__ == '__main__':
    main_dispatch(generate, test, 'test_mmd_objective')
