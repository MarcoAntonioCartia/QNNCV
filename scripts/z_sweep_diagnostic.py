"""
z-sweep diagnostic: how does the generated density respond to the latent z?

Loads a trained CVQGANGenerator (either encoding, either best_weights.npy
format), sweeps z over a deterministic grid, and extracts per-density shape
descriptors. Purpose: run on FROZEN baseline weights to produce the
"front encoding exploits only rigid motions" before-figure, and later on
interleaved weights for the after-figure.

Sweeps (all deterministic, no RNG):
  - 1D rays: z(t) = t * e_i for every latent component i,
    t in linspace(-t_max, t_max, n_steps)
  - 2D grid: z spanned by two chosen components (--grid-components)

Descriptors per density (probability P on the [-x_range, x_range] grid):
  centroid_x/y, orientation theta = 0.5*atan2(2Cxy, Cxx - Cyy), principal
  widths sigma_maj/sigma_min (sqrt of covariance eigenvalues), mean_radius
  and radial_width about the centroid, raw second moments m2x/m2y, and the
  truncated-ket norm.

Usage:
  python scripts/z_sweep_diagnostic.py --run-dir logs/<run>          # config
      and best_weights.npy read from the run folder
  python scripts/z_sweep_diagnostic.py --weights PATH --encoding ... \
      --n-modes 2 --n-layers 6 --cutoff-dim 12 --latent-scale 0.3
  python scripts/z_sweep_diagnostic.py --smoke [--encoding ...]      # seeded
      random weights, small sweep (the only mode to run before real
      checkpoints exist)

Outputs <out>.npz (descriptor tensors + config) and <out>.png (summary
figure: each descriptor vs t per ray; heatmaps over the 2D grid).
"""

import _bootstrap  # noqa: F401  (repo root + scipy simps alias, BEFORE src.*)

import argparse
import json
import os

import numpy as np

DESCRIPTORS = ['centroid_x', 'centroid_y', 'orientation', 'sigma_maj',
               'sigma_min', 'mean_radius', 'radial_width', 'm2x', 'm2y',
               'ket_norm']


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    src = p.add_argument_group('generator source (pick one)')
    src.add_argument('--run-dir', type=str, default=None,
                     help='Run folder: reads run_config.json + best_weights.npy')
    src.add_argument('--weights', type=str, default=None,
                     help='Path to a best_weights.npy (v1 or v2)')
    src.add_argument('--smoke', action='store_true',
                     help='Seeded random weights, small sweep (no file needed)')
    cfg = p.add_argument_group('generator config (with --weights/--smoke)')
    cfg.add_argument('--encoding', type=str, default='input_displacement',
                     choices=['input_displacement', 'interleaved_squeeze'])
    cfg.add_argument('--inj-layers', type=str, default=None)
    cfg.add_argument('--r-max', type=float, default=None)
    cfg.add_argument('--squeeze-scale', type=float, default=0.0,
                     help='interleaved only: value of the gain w_s (fixed '
                          'unless --train-squeeze-scale); must match the '
                          'checkpoint when loading a fixed-scale v2 file')
    cfg.add_argument('--train-squeeze-scale', action='store_true')
    cfg.add_argument('--inj-latent', type=str, default='shared',
                     choices=['shared', 'dedicated'])
    cfg.add_argument('--n-modes', type=int, default=2)
    cfg.add_argument('--n-layers', type=int, default=6)
    cfg.add_argument('--cutoff-dim', type=int, default=12)
    cfg.add_argument('--no-kerr', action='store_true')
    cfg.add_argument('--latent-scale', type=float, default=0.3)
    swp = p.add_argument_group('sweep')
    swp.add_argument('--grid-size', type=int, default=40)
    swp.add_argument('--x-range', type=float, default=3.0)
    swp.add_argument('--t-max', type=float, default=2.5)
    swp.add_argument('--n-steps', type=int, default=21)
    swp.add_argument('--grid-components', type=int, nargs=2, default=[0, 2],
                     help='Two latent components spanning the 2D grid')
    p.add_argument('--out', type=str, default=None,
                   help='Output prefix (default: <run-dir>/z_sweep or '
                        './z_sweep_smoke)')
    return p


def load_generator(args):
    """Build the generator + assign weights from the chosen source."""
    import tensorflow as tf
    from src.quantum.circuit import CVQGANGenerator, EncodingSpec
    from src.quantum import weights_io

    if args.run_dir:
        with open(os.path.join(args.run_dir, 'run_config.json')) as f:
            cfg = json.load(f)
        encoding = cfg.get('encoding', 'input_displacement')
        inj = cfg.get('inj_layers')
        if isinstance(inj, str):
            inj = tuple(int(s) for s in inj.split(',') if s.strip())
        # Gain mode: runs from before the fixed-scale design have neither key
        # and were the trainable (learned-gain) design by definition.
        legacy_trainable = (encoding == 'interleaved_squeeze'
                            and 'squeeze_scale' not in cfg)
        gen_kwargs = dict(
            n_modes=cfg['n_modes'], n_layers=cfg['n_layers'],
            cutoff_dim=cfg['cutoff_dim'], use_kerr=not cfg.get('no_kerr'),
            latent_scale=cfg['latent_scale'],
            r_max=cfg.get('r_max'), inj_layers=inj,
            squeeze_scale=cfg.get('squeeze_scale', 0.0),
            train_squeeze_scale=cfg.get('train_squeeze_scale',
                                        legacy_trainable),
            inj_latent=cfg.get('inj_latent', 'shared'))
        weights_path = os.path.join(args.run_dir, 'best_weights.npy')
    else:
        inj = args.inj_layers
        if isinstance(inj, str):
            inj = tuple(int(s) for s in inj.split(',') if s.strip())
        encoding = args.encoding
        gen_kwargs = dict(
            n_modes=args.n_modes, n_layers=args.n_layers,
            cutoff_dim=args.cutoff_dim, use_kerr=not args.no_kerr,
            latent_scale=args.latent_scale,
            r_max=args.r_max, inj_layers=inj,
            squeeze_scale=args.squeeze_scale,
            train_squeeze_scale=args.train_squeeze_scale,
            inj_latent=args.inj_latent)
        weights_path = args.weights

    if encoding == 'input_displacement':
        enc = EncodingSpec(encoding)
    else:
        enc = EncodingSpec(encoding, inj_layers=gen_kwargs['inj_layers'],
                           r_max=gen_kwargs['r_max'],
                           squeeze_scale=gen_kwargs['squeeze_scale'],
                           train_squeeze_scale=gen_kwargs['train_squeeze_scale'],
                           inj_latent=gen_kwargs['inj_latent'])
    for k in ('inj_layers', 'r_max', 'squeeze_scale', 'train_squeeze_scale',
              'inj_latent'):
        gen_kwargs.pop(k)

    tf.random.set_seed(0)  # deterministic init; overwritten unless --smoke
    generator = CVQGANGenerator(n_output_modes=2, encoding=enc, **gen_kwargs)

    if args.smoke:
        print("SMOKE mode: seeded random weights (no checkpoint loaded)")
        if generator.w_s is not None:
            # give the modulation something to show in the smoke figure
            generator.w_s.assign(0.3 * np.ones(generator.w_s.shape,
                                               dtype=np.float32))
    else:
        weights_io.apply_weights(generator, weights_io.load_weights(
            weights_path), source=weights_path)
        print(f"Loaded weights: {weights_path}")
    return generator, encoding


def describe(p, xvec):
    """Shape descriptors of one normalized density P (x-major grid)."""
    X = xvec[:, None]
    Y = xvec[None, :]
    mass = p.sum()
    mux = float((p * X).sum() / mass)
    muy = float((p * Y).sum() / mass)
    cxx = float((p * (X - mux) ** 2).sum() / mass)
    cyy = float((p * (Y - muy) ** 2).sum() / mass)
    cxy = float((p * (X - mux) * (Y - muy)).sum() / mass)
    theta = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
    half_tr = 0.5 * (cxx + cyy)
    half_diff = np.sqrt((0.5 * (cxx - cyy)) ** 2 + cxy ** 2)
    sigma_maj = float(np.sqrt(max(half_tr + half_diff, 0.0)))
    sigma_min = float(np.sqrt(max(half_tr - half_diff, 0.0)))
    R = np.sqrt((X - mux) ** 2 + (Y - muy) ** 2)
    mean_radius = float((p * R).sum() / mass)
    radial_width = float(np.sqrt((p * (R - mean_radius) ** 2).sum() / mass))
    m2x = float((p * X ** 2).sum() / mass)
    m2y = float((p * Y ** 2).sum() / mass)
    return [mux, muy, float(theta), sigma_maj, sigma_min,
            mean_radius, radial_width, m2x, m2y]


def sweep(generator, xvec, t_values, grid_components):
    """Run all rays + the 2D grid; return descriptor tensors."""
    import tensorflow as tf
    D = generator.latent_dim
    n_steps = len(t_values)

    def density_descr(zv):
        p, kn = generator.generate_distribution_2d(
            tf.constant(zv, dtype=tf.float32), xvec, xvec,
            return_ket_norm=True)
        return describe(p.numpy(), xvec) + [float(kn)]

    rays = np.zeros((D, n_steps, len(DESCRIPTORS)))
    for i in range(D):
        for j, t in enumerate(t_values):
            zv = np.zeros(D, dtype=np.float32)
            zv[i] = t
            rays[i, j] = density_descr(zv)
        print(f"  ray e_{i} done")

    ca, cb = grid_components
    grid = np.zeros((n_steps, n_steps, len(DESCRIPTORS)))
    for j, ta in enumerate(t_values):
        for k, tb in enumerate(t_values):
            zv = np.zeros(D, dtype=np.float32)
            zv[ca] = ta
            zv[cb] = tb
            grid[j, k] = density_descr(zv)
    print(f"  2D grid over (e_{ca}, e_{cb}) done")
    return rays, grid


def plot_summary(out_png, t_values, rays, grid, grid_components, encoding):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    D = rays.shape[0]
    n_desc = len(DESCRIPTORS)
    fig, axes = plt.subplots(2, (n_desc + 1) // 2, figsize=(4 * ((n_desc + 1) // 2), 8))
    axes = axes.ravel()
    for d in range(n_desc):
        ax = axes[d]
        for i in range(D):
            ax.plot(t_values, rays[i, :, d], label=f'$e_{{{i}}}$', lw=1.2)
        ax.set_title(DESCRIPTORS[d])
        ax.set_xlabel('t')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    for d in range(n_desc, len(axes)):
        axes[d].axis('off')
    fig.suptitle(f'z-sweep rays — encoding: {encoding}')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    ca, cb = grid_components
    keys = ['sigma_maj', 'mean_radius', 'radial_width', 'orientation']
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.6))
    for ax, key in zip(axes, keys):
        d = DESCRIPTORS.index(key)
        im = ax.imshow(grid[:, :, d].T, origin='lower',
                       extent=[t_values[0], t_values[-1],
                               t_values[0], t_values[-1]], aspect='auto')
        ax.set_title(key)
        ax.set_xlabel(f'$z_{{{ca}}}$')
        ax.set_ylabel(f'$z_{{{cb}}}$')
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(f'z-sweep 2D grid — encoding: {encoding}')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    grid_png = out_png.replace('.png', '_grid.png')
    fig.savefig(grid_png, dpi=140)
    plt.close(fig)
    return grid_png


def main():
    args = build_parser().parse_args()
    if not (args.run_dir or args.weights or args.smoke):
        raise SystemExit("pick a source: --run-dir, --weights, or --smoke")
    if args.smoke:
        args.n_steps = min(args.n_steps, 7)

    generator, encoding = load_generator(args)
    if args.grid_components[0] >= generator.latent_dim or \
            args.grid_components[1] >= generator.latent_dim:
        raise SystemExit(f"--grid-components out of range for latent_dim "
                         f"{generator.latent_dim}")

    xvec = np.linspace(-args.x_range, args.x_range, args.grid_size)
    t_values = np.linspace(-args.t_max, args.t_max, args.n_steps)

    print(f"Sweeping {generator.latent_dim} rays x {args.n_steps} steps + "
          f"{args.n_steps}x{args.n_steps} grid ...")
    rays, grid = sweep(generator, xvec, t_values, args.grid_components)

    out = args.out or (os.path.join(args.run_dir, 'z_sweep') if args.run_dir
                       else './z_sweep_smoke')
    npz = out + '.npz'
    np.savez(npz,
             t_values=t_values,
             ray_descriptors=rays,
             grid_descriptors=grid,
             descriptor_names=np.array(DESCRIPTORS),
             grid_components=np.array(args.grid_components),
             encoding=encoding,
             config=json.dumps(vars(args), default=str))
    png = out + '.png'
    grid_png = plot_summary(png, t_values, rays, grid,
                            args.grid_components, encoding)
    print(f"Saved: {npz}\n       {png}\n       {grid_png}")
    assert np.all(np.isfinite(rays)) and np.all(np.isfinite(grid)), \
        "non-finite descriptors in sweep"
    print("All descriptors finite.")


if __name__ == '__main__':
    main()
