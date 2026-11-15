"""
Figure F: how the trained CV generator degrades under photonic imperfections
(eval-only; reads analysis/hw_sweep/hw_sweep.csv written by scripts/hw_sweep.py).

Runs: the 20 CV generators trained against the dense critic (ring, 32x32, 300
epochs). Each condition is paired with the ideal circuit of the same checkpoint
and the same latent draws, so every point is a per-seed ratio
metric(condition) / metric(ideal), then median + IQR band over the seeds
(1 = no change; above 1 = worse, all three metrics are lower-is-better).
Propagation loss is a mixed-state simulation on fewer draws (80), so it is
paired with its own ideal reference on those draws ('ideal_nloss').

Rows = metric (fidelity W1, coverage W1, MMD^2); columns = imperfection:
  (a) homodyne efficiency eta: raw outcomes vs a calibrated detector (1/sqrt eta)
  (b) propagation loss per layer and mode, dB (applied after each of 6 layers)
  (c) squeezing cap |r| <= r_max, evenly spaced categories (dB in the axis label);
      'sin límite' = the trained circuit
  (d) Kerr strength scaled by a factor (0 = Gaussian-only circuit)

Caption caveats: fidelity alone can improve under blur (a,b) because a blurred
sample sits closer to *some* validation member, while coverage and MMD^2 get
worse; read the three rows together. Squeeze cap only bites above the trained
magnitudes (median max |r| over runs printed by main()).
"""

import collections

import numpy as np
import matplotlib.pyplot as plt

import style as S

METRICS = ['fid_w1', 'cov_w1', 'mmd2']
DB_PER_R = 20.0 / np.log(10.0)
N_LAYERS = 6
CAL = dict(color=S.INK2, marker='D', ls=':')


def load():
    rows = S.read_csv('analysis/hw_sweep/hw_sweep.csv')
    by = collections.defaultdict(dict)          # (cond, level) -> {seed: row}
    for r in rows:
        by[(r['condition'], float(r['level']))][int(r['seed'])] = r
    return by


def ratios(by, cond, levels, metric, ref='ideal'):
    """(n_seeds, n_levels) per-seed ratio to the matched ideal; level None = ideal itself."""
    refs = by[(ref, 0.0)]
    seeds = sorted(refs)
    out = []
    for s in seeds:
        base = float(refs[s][metric])
        row = []
        for lv in levels:
            if lv is None:
                row.append(1.0)
            else:
                r = by.get((cond, lv), {}).get(s)
                row.append(float(r[metric]) / base if r else np.nan)
        out.append(row)
    return np.array(out), len(seeds)


def draw(ax, x, R, st, label=None):
    med, q25, q75, _, _ = S.band(R)
    ax.fill_between(x, q25, q75, color=st['color'], alpha=0.18, lw=0)
    ax.plot(x, med, color=st['color'], ls=st['ls'], marker=st['marker'], ms=4.5,
            mfc=S.SURFACE if st.get('hollow') else st['color'], mew=1.1, label=label)


def main():
    by = load()
    if not by:
        print('fig_hw: no analysis/hw_sweep/hw_sweep.csv, skipped')
        return
    cv = dict(S.ARM['cv'])
    fig, axes = plt.subplots(3, 4, figsize=(S.TEXTWIDTH, 0.78 * S.TEXTWIDTH), sharey='row',
                             layout='constrained')
    n_seen = set()

    eta = [0.99, 0.95, 0.9, 0.8, 0.7, 0.5]
    loss = [0.05, 0.1, 0.2, 0.5]
    rcap = [0.3, 0.2, 0.1, 0.05]
    kerr = [0.5, 0.0]
    for i, m in enumerate(METRICS):
        ax = axes[i, 0]
        R, n = ratios(by, 'eta_raw', [None] + eta, m); n_seen.add(n)
        draw(ax, [1.0] + eta, R, cv, 'salida cruda')
        R, n = ratios(by, 'eta_cal', [None] + eta, m)
        draw(ax, [1.0] + eta, R, CAL, 'detector calibrado')
        ax.set_xlim(1.02, 0.47)

        ax = axes[i, 1]
        R, n = ratios(by, 'loss_db', [None] + loss, m, ref='ideal_nloss'); n_seen.add(n)
        draw(ax, [0.0] + loss, R, cv)
        ax.set_xlim(-0.02, 0.52)

        ax = axes[i, 2]
        R, n = ratios(by, 'rcap', [None] + rcap, m); n_seen.add(n)
        draw(ax, np.arange(len(rcap) + 1), R, cv)      # evenly spaced categories
        ax.set_xlim(-0.3, len(rcap) + 0.3)
        ax.set_xticks(np.arange(len(rcap) + 1))
        ax.set_xticklabels(['sin\nlímite'] + [f'{r:g}'.replace('.', ',') for r in rcap])

        ax = axes[i, 3]
        R, n = ratios(by, 'kerr', [None] + kerr, m); n_seen.add(n)
        draw(ax, [1.0] + kerr, R, cv)
        ax.set_xlim(1.05, -0.05)
        ax.set_xticks([1.0, 0.5, 0.0])

        for ax in axes[i]:
            ax.axhline(1.0, color=S.MUTED, lw=0.8, zorder=1)
        axes[i, 0].set_ylabel(f'{S.LABEL[m]}\n(relativo al ideal)')

    for ax in axes[:-1].ravel():
        ax.tick_params(labelbottom=False)
    axes[-1, 0].set_xlabel('eficiencia $\\eta$')
    axes[-1, 1].set_xlabel(f'dB por capa y modo\n({N_LAYERS} capas)')
    axes[-1, 2].set_xlabel('$|r|$ máximo\n(0,1 ≈ 0,9 dB; 0,3 ≈ 2,6 dB)')
    axes[-1, 3].set_xlabel('factor de Kerr')
    titles = ['(a) detección\nhomodina', '(b) pérdida de\npropagación',
              '(c) compresión\nfinita', '(d) intensidad\nde Kerr']
    for ax, t in zip(axes[0], titles):
        ax.set_title(t, loc='left')
    axes[0, 0].legend(loc='upper left', handlelength=2.2, fontsize=6.5)
    n = max(n_seen)
    fig.supxlabel(f'VC entrenado con el crítico denso; mediana y rango intercuartílico sobre {n} semillas,\n'
                  'cada una comparada con su propio circuito ideal (1 = sin cambio; mayor = peor).',
                  fontsize=7, color=S.INK2)

    # caption facts
    import glob, json, os
    rmax = [json.load(open(p))['stats']['squeeze_r_max']
            for p in glob.glob(os.path.join(S.REPO, 'analysis', 'hw_sweep', 'qgan_*.json'))]
    if rmax:
        print(f'fig_hw: trained max |r| per run: median {np.median(rmax):.2f}, '
              f'range {min(rmax):.2f}-{max(rmax):.2f} ({np.median(rmax) * DB_PER_R:.1f} dB median)')
    S.save(fig, 'fig_hw_degradation_cv')


if __name__ == '__main__':
    S.apply_style()
    main()
