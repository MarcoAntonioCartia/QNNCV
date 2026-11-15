"""
Figure A: paired comparison classical vs CV generator across the four critics.

A1 (fig_forest): forest plot. One panel per metric, one row per critic.
    Point = median over seeds of the paired delta d = classical - CV, bar =
    95 % percentile-bootstrap interval of that median (10,000 resamples,
    scripts/analyze_sweep.paired_bootstrap), right-hand text = seeds on which
    the CV generator is better. All metrics are distances (lower is better),
    so d > 0 means the CV generator is better.
A2 (fig_paired_seeds): one line per seed from the classical to the CV value,
    coloured by the winner, for fidelity and coverage W1.

Data: analysis/rescore/rescore.csv (W1, MMD^2; 200 fixed-seed samples) and
analysis/kl_eval/kl_family.csv (Riofrío-style mixture KL on 16 x 16).

    python scripts/figures/fig_forest.py
"""

import numpy as np
import matplotlib.pyplot as plt

import style as S

METRICS = [('fid_w1', 'rescore'), ('cov_w1', 'rescore'), ('mmd2', 'rescore'), ('kl16_mixture', 'kl')]
SRC = {'rescore': 'analysis/rescore/rescore.csv', 'kl': 'analysis/kl_eval/kl_family.csv'}


def paired(rows, critic, col):
    a = S.by_seed(rows, S.cell_group('classical', critic), col)
    b = S.by_seed(rows, S.cell_group('cv', critic), col)
    seeds = sorted(set(a) & set(b))
    return seeds, np.array([a[s] for s in seeds]), np.array([b[s] for s in seeds])


def forest():
    data = {k: S.read_csv(v) for k, v in SRC.items()}
    fig, axes = plt.subplots(1, len(METRICS), figsize=(S.TEXTWIDTH, 2.5), sharey=True)
    y = np.arange(len(S.CRITICS))[::-1]
    summary = []
    for ax, (col, src) in zip(axes, METRICS):
        ax.axvline(0, color=S.INK2, lw=0.8, zorder=1)
        for yi, crit in zip(y, S.CRITICS):
            seeds, cl, cv = paired(data[src], crit, col)
            if len(seeds) < 3:
                continue
            d = cl - cv
            (lo, hi), _, _, _ = S.paired_bootstrap(d, 'lower')
            med = float(np.median(d))
            n_cv = int((d > 0).sum())
            sig = lo > 0 or hi < 0
            winner = 'cv' if med > 0 else 'classical'
            c = S.ARM[winner]['color'] if sig else S.MUTED
            ax.plot([lo, hi], [yi, yi], color=c, lw=2, solid_capstyle='round', zorder=2)
            if sig:
                ax.plot(med, yi, marker=S.ARM[winner]['marker'], ms=6, color=c, mec='white', mew=1.2, zorder=3)
            else:   # hollow: "interval includes 0" is not told apart by colour alone
                ax.plot(med, yi, marker='o', ms=6, mfc=S.SURFACE, mec=c, mew=1.4, zorder=3)
            ax.annotate(f'{n_cv}/{len(seeds)}', (med, yi), xytext=(0, 7),
                        textcoords='offset points', va='bottom', ha='center', fontsize=6.5, color=S.INK2,
                        bbox=dict(boxstyle='round,pad=0.1', fc=S.SURFACE, ec='none'), zorder=4)
            summary.append((col, crit, med, lo, hi, n_cv, len(seeds)))
        ax.set_title(S.LABEL[col])
        ax.set_ylim(-0.5, len(S.CRITICS) - 0.3)
        ax.grid(axis='y', visible=False)
        lim = max(abs(v) for v in ax.get_xlim())
        ax.set_xlim(-lim, lim)
    axes[0].set_yticks(y, [S.LABEL[c] for c in S.CRITICS])
    axes[0].set_ylabel('crítico')
    fig.supxlabel('mediana de la diferencia pareada clásico − VC (IC 95 %):  < 0 clásico mejor,  > 0 VC mejor\n'
                  'hueco gris = el intervalo incluye 0 (sin corrección por comparaciones múltiples);'
                  '  n/20 = semillas con VC mejor', fontsize=8, y=-0.09)
    fig.subplots_adjust(wspace=0.35)
    S.save(fig, 'fig_forest_classical_vs_cv')
    print('\nmetric, critic, median delta (classical - CV), 95% CI, CV better on')
    for col, crit, med, lo, hi, n, N in summary:
        print(f'  {col:13s} {crit:6s} {med:+.3f} [{lo:+.3f}, {hi:+.3f}]  {n}/{N}')


def paired_seeds():
    rows = S.read_csv(SRC['rescore'])
    cols = ['fid_w1', 'cov_w1']
    fig, axes = plt.subplots(len(cols), len(S.CRITICS), figsize=(S.TEXTWIDTH, 3.6), sharey='row')
    for i, col in enumerate(cols):
        for j, crit in enumerate(S.CRITICS):
            ax = axes[i, j]
            seeds, cl, cv = paired(rows, crit, col)
            for a, b in zip(cl, cv):
                w = 'cv' if b < a else 'classical'
                ax.plot([0, 1], [a, b], color=S.ARM[w]['color'], lw=0.8, alpha=0.55, zorder=2)
            for x, v, arm in ((0, cl, 'classical'), (1, cv, 'cv')):
                ax.plot(x, np.median(v), marker=S.ARM[arm]['marker'], ms=7, color=S.ARM[arm]['color'],
                        mec='white', mew=1.2, zorder=4)
            ax.set_xlim(-0.35, 1.35)
            ax.set_xticks([0, 1], [S.LABEL['classical_short'], S.LABEL['cv_short']])
            ax.grid(axis='x', visible=False)
            n_cv = int((cv < cl).sum())
            head = S.LABEL[crit].replace('\n', ' ') + '\n' if i == 0 else ''
            ax.set_title(f'{head}VC mejor: {n_cv}/{len(seeds)}', fontsize=7.5 if i == 0 else 6.5)
            if j == 0:
                ax.set_ylabel(S.LABEL[col] + '\n(menor es mejor)')
    fig.subplots_adjust(wspace=0.15, hspace=0.3)
    S.save(fig, 'fig_paired_seeds_classical_vs_cv')


if __name__ == '__main__':
    S.apply_style()
    forest()
    paired_seeds()
