"""
Figure B: training dynamics as bands over seeds, one colour per generator.

Line = median over seeds, dark band = interquartile range (25-75 %), faint
band = min-max. The median and IQR are used instead of mean and min-max so
that a single diverging seed cannot stretch the band.

B1 (fig_bands_val_<critic>): validation metrics recorded during training
    (history.npz, every val_every = 20 epochs on the 100-member validation
    set): nearest-member W1 (fidelity) and diversity (mean pairwise W1 of the
    fakes), with the data's own self-diversity as reference. Arms share axes:
    these are the same metric for every generator.
B2 (fig_bands_critic_<critic>): the critic's own Wasserstein estimate per
    epoch, d_real_score - d_fake_score (= -d_loss; gradient penalty not
    included), ONE PANEL PER ARM with its own y-axis: each arm trains its own
    critic (CV with gp 10, classical and DV with gp 5), so the scales are not
    comparable across arms. Curves smoothed per seed with a 5-epoch moving
    average. (history['wasserstein'] is NOT this: it is the marginal W1 of one
    generated sample to the canonical ring member, once per epoch.)
    The generator gradient norm is NOT plotted: history.npz logs it after
    per-tensor clipping at g_grad_clip (5), so it shows clip saturation, not
    the gradient (pending fix: log the norm before clipping).

    python scripts/figures/fig_bands.py [--critic dense|coord|cvq|dvq|all]
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

import style as S

VAL_EVERY = 20


def load_histories(group, keys):
    runs = S.run_dirs(group)
    out = {k: [] for k in keys}
    for seed, d in sorted(runs.items()):
        h = np.load(f'{d}/history.npz', allow_pickle=True)
        for k in keys:
            out[k].append(np.asarray(h[k], dtype=float))
    for k in keys:   # keep only complete runs of the modal length
        if out[k]:
            L = max(set(len(v) for v in out[k]), key=[len(v) for v in out[k]].count)
            out[k] = np.array([v for v in out[k] if len(v) == L])
    return out, len(runs)


def draw_band(ax, x, curves, arm, label=None, solid=False):
    med, q25, q75, lo, hi = S.band(curves)
    c = S.ARM[arm]['color']
    ax.fill_between(x, lo, hi, color=c, alpha=0.08, lw=0)
    ax.fill_between(x, q25, q75, color=c, alpha=0.28, lw=0)
    ax.plot(x, med, color=c, ls='-' if solid else S.ARM[arm]['ls'], lw=1.2 if solid else 1.6, label=label)


def smooth(C, w=5):
    k = np.ones(w) / w
    return np.array([np.convolve(c, k, mode='valid') for c in C])


def val_figure(critic):
    rows = S.read_csv('analysis/rescore/rescore.csv')
    fig, axes = plt.subplots(1, 2, figsize=(S.TEXTWIDTH, 2.4))
    ref = [float(r['val_self_div_w1']) for r in rows
           if r['group'] in [S.cell_group(a, critic) for a in S.ARMS] and r.get('val_self_div_w1')]
    drawn = []
    for arm in S.ARMS:
        H, n = load_histories(S.cell_group(arm, critic), ['val_nearest_w1', 'val_diversity'])
        if len(H['val_nearest_w1']) < 3:
            continue
        x = VAL_EVERY * np.arange(1, H['val_nearest_w1'].shape[1] + 1)
        lab = f"{S.LABEL[arm]}, n = {len(H['val_nearest_w1'])}"
        draw_band(axes[0], x, H['val_nearest_w1'], arm, lab)
        draw_band(axes[1], x, H['val_diversity'], arm)
        drawn.append(arm)
    if ref:
        axes[1].axhline(np.median(ref), color=S.INK2, lw=0.9, ls=(0, (4, 2)))
        axes[1].annotate('datos', (1.0, np.median(ref)), xycoords=('axes fraction', 'data'),
                         xytext=(2, 0), textcoords='offset points', va='center', fontsize=7, color=S.INK2)
    axes[0].set_ylabel(S.LABEL['fid_w1'] + ' (val.)\nmenor es mejor')
    axes[1].set_ylabel('diversidad (W$_1$ medio\nentre muestras)')
    for ax in axes:
        ax.set_xlabel(S.LABEL['epoch'])
        ax.set_xlim(0, 300)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 1.08))
    fig.subplots_adjust(wspace=0.35)
    if drawn:
        S.save(fig, f'fig_bands_val_{critic}')
    return drawn


def critic_figure(critic):
    arms = [a for a in S.ARMS if len(S.run_dirs(S.cell_group(a, critic))) >= 3]
    if not arms:
        return
    fig, axes = plt.subplots(1, len(arms), figsize=(S.TEXTWIDTH, 2.0), squeeze=False)
    for j, arm in enumerate(arms):
        H, n = load_histories(S.cell_group(arm, critic), ['d_real_score', 'd_fake_score'])
        C = smooth(H['d_real_score'] - H['d_fake_score'])   # critic's own W estimate (= -d_loss, GP excluded)
        x = np.arange(C.shape[1]) + 3      # centre of the 5-epoch window, epochs 1-based
        draw_band(axes[0, j], x, C, arm, solid=True)
        axes[0, j].axhline(0, color=S.INK2, lw=0.6)
        axes[0, j].set_xlim(0, 300)
        axes[0, j].set_title(f"{S.LABEL[arm]}, n = {len(C)}", fontsize=8)
        axes[0, j].set_xlabel(S.LABEL['epoch'])
    axes[0, 0].set_ylabel('estimación de W\ndel crítico')
    fig.subplots_adjust(wspace=0.3)
    S.save(fig, f'fig_bands_critic_{critic}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--critic', default='all', choices=S.CRITICS + ['all'])
    a = ap.parse_args()
    S.apply_style()
    for crit in (S.CRITICS if a.critic == 'all' else [a.critic]):
        print(crit, 'arms drawn:', val_figure(crit))
        critic_figure(crit)
