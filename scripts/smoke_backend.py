"""
Smoke-fit any registered generator backend to a family's canonical target
WITHOUT the GAN trainer: a few Adam steps on KL(target || generated) plus
the backend's own regularizer. This is the standalone check every piece
of docs/benchmark_plan.md ships with; it never touches
src/training/qgan_2d.py and writes nothing under logs/.

    python scripts/smoke_backend.py --generator classical --steps 200
    python scripts/smoke_backend.py --generator cv --steps 20 --n-modes 2 --cutoff-dim 6

Prints the KL at the start and the end, the nearest-member W1 of a few
samples against the family's canonical member, and describe(). Exit code
1 if the loss did not decrease.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402  (repo root + simps alias, before src/SF)

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from src.backends import registry  # noqa: E402
from src.families.registry import get_family  # noqa: E402
from src.metrics.wasserstein import compute_wasserstein_2d  # noqa: E402


def build(args):
    if args.generator == 'cv':
        from src.backends.cv import CVGeneratorBackend
        return CVGeneratorBackend(n_modes=args.n_modes, n_layers=args.n_layers,
                                  cutoff_dim=args.cutoff_dim, latent_scale=0.3,
                                  batch_size=args.batch)
    if args.generator == 'classical':
        from src.backends.classical import ClassicalCoordGenerator
        return (ClassicalCoordGenerator.large() if args.large
                else ClassicalCoordGenerator.parameter_matched(features=args.features))
    raise SystemExit(f"unknown generator {args.generator}; registered: "
                     f"{sorted(registry.GENERATOR_REGISTRY)}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--generator', required=True, choices=sorted(registry.GENERATOR_REGISTRY))
    p.add_argument('--family', default='ring')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--reg-weight', type=float, default=20.0, help='weight on regularizer(aux)')
    p.add_argument('--n-modes', type=int, default=2)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--cutoff-dim', type=int, default=6)
    p.add_argument('--features', default='xy', choices=('xy', 'xyr'))
    p.add_argument('--large', action='store_true')
    args = p.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    fam = get_family(args.family, 40, 3.0)
    target, _ = fam.get_canonical()
    target = tf.constant(target / target.sum(), tf.float32)
    g = build(args)
    print("describe:", g.describe())
    opt = tf.keras.optimizers.Adam(args.lr)
    eps = 1e-10

    def kl(probs):
        return tf.reduce_mean(tf.reduce_sum(
            target * (tf.math.log(target + eps) - tf.math.log(probs + eps)), axis=[-2, -1]))

    z = tf.random.normal([args.batch, g.latent_dim])
    first = float(kl(g.generate_batch(z, fam.xvec, fam.yvec)[0]))
    t0 = time.time()
    for step in range(1, args.steps + 1):
        z = tf.random.normal([args.batch, g.latent_dim])
        with tf.GradientTape() as tape:
            probs, aux = g.generate_batch(z, fam.xvec, fam.yvec)
            loss = kl(probs) + args.reg_weight * g.regularizer(aux)
        grads = tape.gradient(loss, g.trainable_variables)
        opt.apply_gradients(zip(grads, g.trainable_variables))
        if step % max(1, args.steps // 5) == 0 or step == 1:
            print(f"  step {step:4d}  loss {float(loss):.4f}")
    last = float(kl(g.generate_batch(z, fam.xvec, fam.yvec)[0]))
    probs = g.generate_batch(tf.random.normal([args.batch, g.latent_dim]), fam.xvec, fam.yvec)[0].numpy()
    w1 = [compute_wasserstein_2d(target.numpy(), probs[b]) for b in range(args.batch)]
    print(f"KL start {first:.4f} -> end {last:.4f}  ({time.time() - t0:.1f}s, "
          f"{(time.time() - t0) / args.steps:.3f} s/step); SUPERVISED/CANONICAL W1 "
          f"of {args.batch} samples: {np.round(w1, 3).tolist()}")
    print("NOTE: this is the Path-A objective (one fixed target, supervised KL). "
          "It is NOT comparable to the GAN sweep's nearest-member W1 over the "
          "family; never put the two in one table without naming the axis.")
    ok = last < first
    print("SMOKE", "OK" if ok else "FAILED (loss did not decrease)")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
