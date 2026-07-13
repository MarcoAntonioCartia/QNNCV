"""
Child runner for the Path A end-to-end golden (NOT a test itself).

Loads the root train_2d_qgan.py via importlib (golden tests deliberately
stay on the shim surface), runs the tiny pure-supervised config
in-process so it can hold the returned generator, then saves the
fixed-z density next to the run outputs.

Usage: python e2e_patha_child.py <out_dir>
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

out_dir = sys.argv[1]

if REPO not in sys.path:
    sys.path.insert(0, REPO)
spec = importlib.util.spec_from_file_location(
    't2q', os.path.join(REPO, 'train_2d_qgan.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

generator, history = mod.train_2d_qgan(
    seed=0, deterministic=True,
    family_name='gaussian', n_train=8, n_val=6, n_total_modes=2,
    n_layers=2, cutoff_dim=6, epochs=4, val_every=2, plot_every=999,
    supervised_weight=1.0, supervised_warmup=4,
    # former signature defaults, passed explicitly since the default
    # unification; the e2e_patha goldens were generated under these values
    d_lr=0.0002, n_critic=1, batch_size=1, d_dropout=0.3, latent_scale=1.0,
    log_dir=out_dir,
)

xvec = np.linspace(-3.0, 3.0, 40)
density = generator.generate_distribution_2d(
    tf.zeros([generator.latent_dim]), xvec, xvec).numpy()
np.save(os.path.join(out_dir, 'fixed_z_density.npy'), density)
print('CHILD DONE')
