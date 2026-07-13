#!/usr/bin/env python
"""
Unified 2D CV-QGAN with Pre-Generated Dataset
==============================================

TRUE QGAN Implementation with these key improvements:
1. Pre-generate 500 distributions (400 train, 100 validation)
2. Generator receives RANDOM latent vectors z as input
3. z is encoded into the circuit via displacement gates
4. Optional true batching via the SF TF backend (--batch-size; 1 = sequential)
5. Visualizations show ACTUAL training samples, not just canonical
6. Validation on held-out set for generalization metrics

Architecture (per Killoran et al.):
    ENCODING: Dgate(z0, z1)|q[0], Dgate(z2, z3)|q[1], ...  # Latent -> quantum state
    Then for each layer:
        U1(interferometer) -> S(squeeze) -> U2(interferometer) -> D(displacement) -> K(Kerr)

Supports N >= 2 total qumodes. Only the first 2 modes produce the output P(x,y).
Extra modes (ancilla) provide additional entanglement and expressivity via
beamsplitters, then are traced out (partial trace) at measurement time.

Author: QNNCV Project
Date: January 2026
"""

# =============================================================================
# Imports & Compatibility
# =============================================================================

# Compatibility patches
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

import os
import sys
import argparse
import json

# Refactor strangler: the training code is being extracted into src/ (see
# tests/golden/). Hard `from src...` imports below need the repo root on
# sys.path even when this file is importlib-loaded from elsewhere (e.g.
# scripts/verify_batching.py runs with scripts/ as sys.path[0]).
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Determinism env vars are read at TensorFlow *import* time, but argparse runs
# in main() long after the import below. Scan sys.argv here so that opt-in
# --deterministic runs take effect. Default (flag absent) leaves env untouched.
if '--deterministic' in sys.argv:
    os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# numpy/tensorflow/strawberryfields stay imported here: their __version__s
# are recorded into run_config.json by main() (golden-pinned provenance).
import numpy as np
import tensorflow as tf
import strawberryfields as sf
from datetime import datetime

# Warning suppression (if available)
try:
    from src.utils.warning_suppression import enable_clean_training
    enable_clean_training()
except ImportError:
    pass

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Re-exports (shim surface)
# =============================================================================
# The golden tests (tests/golden/) load this file via importlib and reach the
# extracted code through these names — keep each until nothing loads it via
# the shim (grep tests/ and scripts/ before removing). main() itself uses
# train_2d_qgan and resolve_seed.

from src.quantum.hermite import compute_hermite_basis, recommend_cutoff
from src.quantum.circuit import CVQGANGenerator
from src.models.discriminators.qgan2d_discriminator import Discriminator2D
from src.families.registry import get_family
from src.metrics.wasserstein import compute_wasserstein_2d
from src.metrics.energy import build_energy_distance_context, energy_distances
from src.data.dataset import generate_dataset
from src.metrics.validation import validate
from src.critic_input.pipeline import (to_critic_input, build_blur_kernel,
                                        critic_blur, compute_gradient_penalty)
from src.training.qgan_2d import (train_2d_qgan, resolve_seed,
                                   seed_everything)


# =============================================================================
# Main
# =============================================================================


def _format_val(v):
    """Format a value for directory naming (strip trailing zeros from floats)."""
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def build_hp_suffix(args, parser):
    """Build a suffix string from non-default hyperparameters.

    Only parameters that differ from argparse defaults are included,
    using short abbreviations. Cosmetic params are excluded.
    """
    # (argparse dest, abbreviation) in display order
    HP_MAP = [
        ('n_modes',           'm'),
        ('n_layers',          'nl'),
        ('cutoff_dim',        'c'),
        ('no_kerr',           'nokerr'),
        ('epochs',            'ep'),
        ('g_lr',              'glr'),
        ('d_lr',              'dlr'),
        ('n_critic',          'nc'),
        ('batch_size',        'bs'),
        ('supervised_weight', 'sw'),
        ('supervised_warmup', 'swup'),
        ('gp_weight',         'gp'),
        ('gp_warmup',         'gpwup'),
        ('instance_noise',    'noise'),
        ('noise_anneal',      'nann'),
        ('noise_floor',       'nfloor'),
        ('critic_blur',       'blur'),
        ('d_dropout',         'drop'),
        ('latent_scale',      'ls'),
        ('ket_penalty_weight', 'kpen'),
        ('g_grad_clip',       'gclip'),
        ('grid_size',         'gs'),
        ('seed',              'seed'),
    ]

    parts = []
    for dest, abbrev in HP_MAP:
        default = parser.get_default(dest)
        actual = getattr(args, dest, default)
        if actual != default:
            if isinstance(actual, bool):
                if actual:          # flag was set (e.g. --no-kerr)
                    parts.append(abbrev)
            else:
                parts.append(f"{abbrev}{_format_val(actual)}")
    return ('_' + '_'.join(parts)) if parts else ''


def build_parser():
    """Build the CLI argument parser.

    Exposed as a function (CLI-shim seam) so tooling can introspect the flag
    surface — e.g. tests/golden/test_sweep_sync.py keeps run_sweep.py's
    SWEEP_PARAMS from drifting out of sync with these flags.
    """
    parser = argparse.ArgumentParser(description='Train 2D CV-QGAN with Pre-Generated Dataset')
    parser.add_argument('--family', type=str, required=True,
                       choices=['gaussian', 'ring', 'correlated', 'four_gaussians', 'vibronic'],
                       help='Distribution family to learn (REQUIRED)')
    parser.add_argument('--n-train', type=int, default=400,
                       help='Number of training samples')
    parser.add_argument('--n-val', type=int, default=100,
                       help='Number of validation samples')
    parser.add_argument('--n-modes', type=int, default=2,
                       help='Total qumodes (2=no ancilla, 3=1 ancilla, etc.)')
    parser.add_argument('--n-ancilla', type=int, default=None,
                       help='Number of ancilla modes (alternative to --n-modes)')
    parser.add_argument('--n-layers', type=int, default=6,
                       help='Number of CV-QNN layers')
    parser.add_argument('--cutoff-dim', type=int, default=12,
                       help='Fock space cutoff dimension')
    parser.add_argument('--no-kerr', action='store_true',
                       help='Disable Kerr gates (not recommended)')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs')
    parser.add_argument('--g-lr', type=float, default=0.005,
                       help='Generator learning rate')
    parser.add_argument('--d-lr', type=float, default=0.005,
                       help='Critic learning rate (WGAN-GP wants a well-trained '
                            'critic; consider raising toward --g-lr)')
    parser.add_argument('--n-critic', type=int, default=5,
                       help='Critic steps per generator step (WGAN-GP standard: 5)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Samples per training step via SF TF backend batching '
                            '(1 = legacy sequential; try 8)')
    parser.add_argument('--supervised-weight', type=float, default=0.0,
                       help='Initial supervised loss weight (0=pure GAN, 1=pure supervised)')
    parser.add_argument('--supervised-warmup', type=int, default=200,
                       help='Epochs over which supervised weight decays to 0')
    parser.add_argument('--gp-weight', type=float, default=5.0,
                       help='Gradient penalty weight (lambda)')
    parser.add_argument('--gp-warmup', type=int, default=50,
                       help='Epochs to warm up gradient penalty')
    parser.add_argument('--instance-noise', type=float, default=0.1,
                       help='Initial instance noise std added to D inputs')
    parser.add_argument('--noise-anneal', type=int, default=200,
                       help='Epochs to anneal instance noise to the floor')
    parser.add_argument('--noise-floor', type=float, default=0.0,
                       help='Instance-noise floor after anneal (0 = anneal to zero, legacy)')
    parser.add_argument('--critic-blur', type=float, default=0.0,
                       help='Gaussian blur sigma (grid cells) applied identically to real '
                            'and fake critic inputs; 0 = off (legacy). Intended range 0.5-1.0')
    parser.add_argument('--d-dropout', type=float, default=0.0,
                       help='Dropout rate in discriminator')
    parser.add_argument('--latent-scale', type=float, default=0.3,
                       help='Scale for latent vector encoding')
    parser.add_argument('--ket-penalty-weight', type=float, default=20.0,
                       help='Weight on (1 - ket_norm)^2 penalty in G loss to prevent Fock truncation')
    parser.add_argument('--g-grad-clip', type=float, default=5.0,
                       help='Max norm for G gradient clipping (<=0 disables)')
    parser.add_argument('--grid-size', type=int, default=40,
                       help='Grid resolution')
    parser.add_argument('--plot-every', type=int, default=20,
                       help='Plot frequency')
    parser.add_argument('--val-every', type=int, default=20,
                       help='Validation frequency')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility. Default: an '
                            'entropy-based seed is drawn and recorded, so '
                            'unseeded runs still vary but stay reproducible.')
    parser.add_argument('--deterministic', action='store_true',
                       help='Opt into stronger (slower) same-machine determinism: '
                            'TF op determinism + single-threaded intra/inter-op + '
                            'TF_DETERMINISTIC_OPS/oneDNN env (off by default).')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Resolve total modes from --n-modes or --n-ancilla
    if args.n_ancilla is not None:
        n_total_modes = 2 + args.n_ancilla
    else:
        n_total_modes = args.n_modes

    if n_total_modes < 2:
        parser.error("--n-modes must be at least 2 (need 2 output modes)")

    # Resolve the seed up front so it is recorded everywhere (folder tag,
    # run_config.json, console, history.npz) — never leave it unknown.
    args.seed = resolve_seed(args.seed)

    # Build log directory name with non-default hyperparameters
    # If --n-ancilla was used, reflect it in args.n_modes for suffix building
    args.n_modes = n_total_modes
    hp_suffix = build_hp_suffix(args, parser)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/qgan_2d_{args.family}_{timestamp}{hp_suffix}"

    # Dump a durable run config (args + environment + seed) so any result can be
    # reproduced and each A/B table row can cite its seed.
    os.makedirs(log_dir, exist_ok=True)
    run_config = dict(vars(args))
    run_config.update({
        'timestamp': timestamp,
        'tf_version': tf.__version__,
        'numpy_version': np.__version__,
        'sf_version': sf.__version__,
        'deterministic': args.deterministic,
        'TF_DETERMINISTIC_OPS': os.environ.get('TF_DETERMINISTIC_OPS'),
        'TF_ENABLE_ONEDNN_OPTS': os.environ.get('TF_ENABLE_ONEDNN_OPTS'),
    })
    with open(f"{log_dir}/run_config.json", 'w') as f:
        json.dump(run_config, f, indent=2, sort_keys=True)

    train_2d_qgan(
        family_name=args.family,
        n_train=args.n_train,
        n_val=args.n_val,
        n_total_modes=n_total_modes,
        n_layers=args.n_layers,
        cutoff_dim=args.cutoff_dim,
        use_kerr=not args.no_kerr,
        epochs=args.epochs,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        n_critic=args.n_critic,
        batch_size=args.batch_size,
        supervised_weight=args.supervised_weight,
        supervised_warmup=args.supervised_warmup,
        gp_weight=args.gp_weight,
        gp_warmup=args.gp_warmup,
        instance_noise=args.instance_noise,
        noise_anneal=args.noise_anneal,
        noise_floor=args.noise_floor,
        critic_blur_sigma=args.critic_blur,
        d_dropout=args.d_dropout,
        latent_scale=args.latent_scale,
        ket_penalty_weight=args.ket_penalty_weight,
        g_grad_clip=args.g_grad_clip,
        grid_size=args.grid_size,
        log_dir=log_dir,
        plot_every=args.plot_every,
        val_every=args.val_every,
        seed=args.seed,
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()
