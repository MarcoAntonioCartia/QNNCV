"""
Figure D: Wigner functions and Wigner negativity of the trained CV generators
(eval-only; reads analysis/wigner/ written by scripts/wigner_eval.py and the
fidelity ranking from analysis/rescore/rescore.csv).

D1 fig_wigner_cv_best_median_worst
   Rows: the best, median and worst of the 20 CV runs trained with the dense
   critic, ranked by fidelity W1 (rescore). Columns: the generated joint density
   P(x1, x2) for z = 0, then the Wigner function of output mode 1 and of output
   mode 2 (single-mode reduced states, hbar = 1 as the readout). Diverging map
   centred on 0, one symmetric scale for every Wigner panel; negative regions
   are the non-classical part. delta = negative volume of that panel.
D2 fig_wigner_negativity
   Negative volume per run (mean over 16 latents and both output modes), one
   column per CV cell: GAN with each critic, and the supervised fits of T1
   (600 epochs, by learning rate). Reference lines: Fock state |1> (delta 0.213)
   and a random initialisation (median 0.003): training creates the negativity.
   Within a cell more negativity does NOT mean a better fit (Spearman with
   fidelity W1, lower = better: dense +0.30 p 0.20, coord +0.13, cvq +0.63
   p 0.003, dvq -0.09); the supervised lr 0.05 fits (lowest KL) have less
   negativity than the lr 0.005 ones.

Caption caveats: these are reduced single-mode states of an entangled 3-mode
state, so they are mixed and their negativity is a lower-level witness (zero
single-mode negativity does not make the joint state Wigner-positive). z = 0 is
the centre of the latent family, one member only.
"""

import collections
import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import style as S

GRID_DIR = os.path.join(S.REPO, 'analysis', 'wigner', 'grids')
NEG_COLOR, POS_COLOR, MID_COLOR = '#b5246b', '#3a3935', '#f4f3ee'
WCMAP = LinearSegmentedColormap.from_list('wig', [NEG_COLOR, MID_COLOR, POS_COLOR])
PCMAP = LinearSegmentedColormap.from_list('dens', [S.SURFACE, '#f6c9ae', S.PALETTE[1], '#7a2e0b'])
FOCK1_DELTA = 2 * np.exp(-0.5) - 1          # (integral |W| - 1)/2 for |1>: 0.2131
INIT_DELTA = 0.0032     # measured 15 Nov: median over 10 random inits (std 0.1, as training), 16 latents
CONTOUR_FRAC = 0.02    # outline W < -2 % of the panel scale; >80 % of delta lies there (checked)


def neg(W, x):
    return float(np.sum(np.clip(-W, 0, None)) * (x[1] - x[0]) ** 2)


def ranked_dense():
    """[(label, seed, fid, npz path)] for best / median / worst dense-critic CV run."""
    group = S.cell_group('cv', 'dense')
    fid = S.by_seed(S.read_csv('analysis/rescore/rescore.csv'), group, 'fid_w1')
    dirs = S.run_dirs(group)
    have = [(f, s) for s, f in fid.items()
            if s in dirs and os.path.exists(os.path.join(GRID_DIR, os.path.basename(dirs[s]) + '.npz'))]
    have.sort()
    if len(have) < 3:
        return []
    picks = [('mejor', have[0]), ('mediana', have[(len(have) - 1) // 2]), ('peor', have[-1])]
    return [(lab, s, f, os.path.join(GRID_DIR, os.path.basename(dirs[s]) + '.npz')) for lab, (f, s) in picks], len(have)


def best_median_worst():
    out = ranked_dense()
    if not out:
        print('fig_wigner: no grids yet, D1 skipped')
        return
    picks, n = out
    data = [(lab, s, f, np.load(p)) for lab, s, f, p in picks]
    vmax = max(max(np.abs(d['W0']).max(), np.abs(d['W1']).max()) for *_, d in data)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(3, 3, figsize=(S.TEXTWIDTH, 0.9 * S.TEXTWIDTH), layout='constrained')
    for i, (lab, seed, fid, d) in enumerate(data):
        xp, P, xw = d['xp'], d['P'], d['xw']
        ax = axes[i, 0]
        ax.imshow(P.T / P.max(), origin='lower', extent=[xp[0], xp[-1], xp[0], xp[-1]], cmap=PCMAP,
                  vmin=0, vmax=1, interpolation='bilinear')
        fid_s = f'{fid:.2f}'.replace('.', ',')
        ax.set_ylabel(f'{lab} (semilla {seed})\nW$_1$ de fidelidad {fid_s}\n\n$x_2$')
        for j, key in enumerate(('W0', 'W1')):
            ax = axes[i, j + 1]
            W = d[key]
            imW = ax.imshow(W, origin='lower', extent=[xw[0], xw[-1], xw[0], xw[-1]], cmap=WCMAP,
                            norm=norm, interpolation='bilinear')
            if (W < -CONTOUR_FRAC * vmax).any():      # outline clear negative regions, not tail ripples
                ax.contour(xw, xw, W, levels=[-CONTOUR_FRAC * vmax], colors=[NEG_COLOR], linewidths=0.6)
            ax.text(0.04, 0.95, f'$\\delta$ = {neg(W, xw):.3f}'.replace('.', ','), transform=ax.transAxes,
                    va='top', fontsize=7, color=S.INK,
                    bbox=dict(boxstyle='round,pad=0.2', fc=S.SURFACE, ec='none', alpha=0.85))
            ax.set_ylabel(f'$p_{j + 1}$')
        for ax in axes[i]:
            ax.grid(False)
            ax.set_aspect('equal')
    for ax in axes[:-1].ravel():
        ax.tick_params(labelbottom=False)
    axes[-1, 0].set_xlabel('$x_1$')
    axes[-1, 1].set_xlabel('$x_1$')
    axes[-1, 2].set_xlabel('$x_2$')
    axes[0, 0].set_title('densidad generada\n$P(x_1, x_2)$, $z = 0$', loc='left')
    axes[0, 1].set_title('Wigner, modo de salida 1\n$W(x_1, p_1)$', loc='left')
    axes[0, 2].set_title('Wigner, modo de salida 2\n$W(x_2, p_2)$', loc='left')
    cb = fig.colorbar(imW, ax=axes[:, 1:], shrink=0.6, aspect=30, pad=0.02)
    cb.set_label('$W(x, p)$  (negativo = no clásico)')
    cb.outline.set_visible(False)
    fig.suptitle(f'VC con crítico denso: mejor, mediana y peor de {n} semillas por W$_1$ de fidelidad',
                 fontsize=8, color=S.INK2)
    S.save(fig, 'fig_wigner_cv_best_median_worst')


GROUPS = [   # (label, group)
    ('GAN\ndenso', S.cell_group('cv', 'dense')),
    ('GAN\npuntual', S.cell_group('cv', 'coord')),
    ('GAN\ncuántico VC', S.cell_group('cv', 'cvq')),
    ('GAN\ncuántico VD', S.cell_group('cv', 'dvq')),
    ('superv.\nlr 0,001', 'qgan_2d_ring_m3_ep600_glr0.001_dlr0.0005_sw1_swup600_gp10_gs16'),
    ('superv.\nlr 0,005', 'qgan_2d_ring_m3_ep600_dlr0.0005_sw1_swup600_gp10_gs16'),
    ('superv.\nlr 0,02', 'qgan_2d_ring_m3_ep600_glr0.02_dlr0.0005_sw1_swup600_gp10_gs16'),
    ('superv.\nlr 0,05', 'qgan_2d_ring_m3_ep600_glr0.05_dlr0.0005_sw1_swup600_gp10_gs16'),
]


def negativity_strip():
    rows = S.read_csv('analysis/wigner/wigner.csv')
    if not rows:
        print('fig_wigner: no wigner.csv yet, D2 skipped')
        return
    latest = {}
    for r in sorted(rows, key=lambda r: r['run']):          # timestamped names: latest per seed wins
        latest[(r['group'], int(float(r['seed'])))] = r
    vals = collections.defaultdict(list)
    for (g, s), r in latest.items():
        vals[g].append(0.5 * (float(r['neg0_mean']) + float(r['neg1_mean'])))
    fig, ax = plt.subplots(figsize=(S.TEXTWIDTH, 0.42 * S.TEXTWIDTH), layout='constrained')
    rng = np.random.default_rng(0)
    xt, xl = [], []
    for i, (lab, g) in enumerate(GROUPS):
        v = np.array(vals.get(g, []))
        if not len(v):
            continue
        x = i + (2 if i >= 4 else 0) * 0.15
        ax.scatter(x + rng.uniform(-0.13, 0.13, len(v)), v, s=14, color=S.PALETTE[1], alpha=0.75,
                   edgecolor=S.SURFACE, linewidth=0.5, zorder=3, marker='s')
        ax.hlines(np.median(v), x - 0.25, x + 0.25, color=S.INK, lw=1.4, zorder=4)
        xt.append(x); xl.append(f'{lab}\n(n={len(v)})')
    ax.axhline(FOCK1_DELTA, color=S.MUTED, ls='--', lw=0.9)
    ax.text(len(GROUPS) - 0.5, FOCK1_DELTA, 'estado de Fock $|1\\rangle$', ha='right', va='bottom',
            fontsize=7, color=S.INK2)
    ax.axhline(INIT_DELTA, color=S.MUTED, ls=':', lw=0.9)
    ax.text(len(GROUPS) - 0.5, INIT_DELTA, 'inicialización aleatoria (mediana)', ha='right', va='bottom',
            fontsize=7, color=S.INK2)
    ax.axvline(3.65, color=S.GRID, lw=1.0)
    ax.text(1.5, 1.02, 'GAN, un crítico por columna (cuadrícula 32, 300 épocas)', transform=ax.get_xaxis_transform(),
            ha='center', fontsize=7, color=S.INK2)
    ax.text(5.8, 1.02, 'ajuste supervisado T1 (cuadrícula 16, 600 épocas)', transform=ax.get_xaxis_transform(),
            ha='center', fontsize=7, color=S.INK2)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=6.8)
    ax.set_ylabel('volumen negativo $\\delta$ de Wigner\n(media de 16 latentes y 2 modos)')
    ax.set_ylim(bottom=0)
    ax.grid(axis='x', visible=False)
    S.save(fig, 'fig_wigner_negativity')


def main():
    best_median_worst()
    negativity_strip()


if __name__ == '__main__':
    S.apply_style()
    main()
