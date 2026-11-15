"""
Figure E: single-target KL on 16 x 16 against parameter count, in the style
of Riofrío et al. (2024), for hypothesis H1 of reports/Por que gana el
clasico (sampler vs explicit density model).

- Classical *sampler* GAN (E3, scripts/sampler_gan.py): noise(8) -> 4 hidden
  layers -> (x, y); per size, the learning rate with the best median is
  shown; marker = median over 5 seeds, bar = interquartile range.
- Explicit density models (E1, Path A supervised fit of the canonical ring,
  no critic; scripts/kl_eval.py --mode single): classical coordinate net,
  CV circuit (frozen lr 0.005; the lr 0.05 runs give the same KL within
  0.02 and are left out), DV circuit. Filled = 16 x 16
  grid, hollow = 32 x 32 sum-pooled to 16 x 16. Median and IQR over 5 seeds.
- Dashed line: KL of a 20,000-point histogram of the target itself (the
  sampler's floor; an explicit density has no sampling floor).

Same ring parameters for all (radius 1.15, width 0.25, [-3, 3]^2), but each
model is scored against its own grid's target: filled points against the
ring point-sampled on 16 x 16, hollow points against the 32 x 32 target
sum-pooled to 16 (the two targets differ by KL 0.023). Filled vs hollow is
therefore "different grid and target", not "same target".

Caption caveats (from the 15 Nov independent check):
- Sampler and explicit points differ in training as well as representation:
  adversarial BCE (5,000 steps) vs supervised KL (300 epochs, no critic).
- DV points may have fewer than 5 seeds while E1 is running (n printed).
- At ~71 params the best lr wins by 0.004 in median over lr 1e-3, whose IQR
  is much narrower; the long bar there reflects that near-tie.

    python scripts/figures/fig_params.py
"""

import numpy as np
import matplotlib.pyplot as plt

import style as S

SAMPLER = PALETTE_SAMPLER = S.PALETTE[3]
FLOOR = 0.012   # measured: KL(T16 || 20k-point histogram of the target), median of 20 draws (0.0116)
# Generator parameter counts (IMPLEMENTATION.md; DV = 2 params per qubit per layer).
EXPLICIT = [  # (label key, arm, group suffix pattern, n_params, filled?)
    ('classical', 'classical', 'glr0.05_dlr0.0005_sw1_swup300_gs16_gcl', 181, True),
    ('classical', 'classical', 'glr0.05_dlr0.0005_sw1_swup300_gs32_gcl', 181, False),
    ('cv', 'cv', 'dlr0.0005_sw1_swup300_gp10_gs16', 186, True),
    ('cv', 'cv', 'dlr0.0005_sw1_swup300_gp10_gs32', 186, False),
    ('dv', 'dv', 'glr0.05_dlr0.0005_sw1_swup300_ls1_gs16_gdv_nq8_dvl11', 176, True),
    ('dv', 'dv', 'glr0.05_dlr0.0005_sw1_swup300_ls1_gs32_gdv', 180, False),
]


def q(v):
    return np.median(v), np.percentile(v, 25), np.percentile(v, 75)


def main():
    S.apply_style()
    e3 = [r for r in S.read_csv('analysis/e3_sampler/e3.csv') if r['target'] == 'canonical']
    e1 = S.read_csv('analysis/kl_eval/kl_single.csv')
    fig, ax = plt.subplots(figsize=(S.TEXTWIDTH * 0.62, 2.9))

    # sampler: best lr per size by median KL16
    pts = []
    for w in sorted({int(r['width_h']) for r in e3}):
        rs = [r for r in e3 if int(r['width_h']) == w]
        best = min({float(r['lr']) for r in rs},
                   key=lambda lr: np.median([float(r['kl16']) for r in rs if float(r['lr']) == lr]))
        v = [float(r['kl16']) for r in rs if float(r['lr']) == best]
        pts.append((int(rs[0]['n_params']), *q(v), best))
    x = np.array([p[0] for p in pts])
    ok = np.array([p[1] < 5 for p in pts])          # the ~3.5k sampler collapsed (KL ~ 15): not part of the trend
    lrs = sorted({p[4] for p in pts})
    ax.plot(x[ok], [p[1] for p, k in zip(pts, ok) if k], color=SAMPLER, marker='D', ms=5, lw=1.2,
            zorder=6,
            label='muestreador clásico\n(GAN ruido → punto, lr ' + ', '.join(f'{v:g}' for v in lrs) + ')')
    ax.vlines(x, [p[2] for p in pts], [p[3] for p in pts], color=SAMPLER, lw=2, zorder=5)
    for p, k in zip(pts, ok):
        if not k:
            ax.plot(p[0], p[1], marker='D', ms=5, color=SAMPLER, ls='none')
            ax.annotate('colapsa\n(5 000 pasos)', (p[0], p[1]), xytext=(0, -8), textcoords='offset points',
                        fontsize=6.5, color=S.INK2, ha='center', va='top')

    # explicit models
    printed = []
    for key, arm, suffix, n, filled in EXPLICIT:
        v = [float(r['kl16']) for r in e1 if r['group'].endswith(suffix) and r['group'].startswith('qgan_2d_ring_m3_ep300_')
             and r['group'][len('qgan_2d_ring_m3_ep300_'):] == suffix]
        if not v:
            continue
        m, lo, hi = q(v)
        xo = n * (1.03 if not filled else 0.97)
        st = S.ARM[arm]
        ax.vlines(xo, lo, hi, color=st['color'], lw=2)
        ax.plot(xo, m, marker=st['marker'], ms=6, color=st['color'], ls='none',
                mfc=st['color'] if filled else S.SURFACE, mec=st['color'], mew=1.4,
                label=f"{S.LABEL[arm.split('_')[0] + '_short']} explícito" if filled else None)
        printed.append((suffix, n, m, lo, hi, len(v)))

    ax.axhline(FLOOR, color=S.INK2, lw=0.8, ls=(0, (4, 2)))
    ax.annotate('suelo del histograma', (x.max() * 1.1, FLOOR), xytext=(0, 3), textcoords='offset points',
                fontsize=6.5, color=S.INK2, ha='right')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(S.LABEL['params'])
    ax.set_ylabel('KL$_{16}$ (anillo canónico)\nmenor es mejor')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
    ax.text(1.03, 0.25, 'relleno: rejilla 16 × 16\nhueco: 32 × 32 agrupada a 16',
            transform=ax.transAxes, fontsize=6.5, color=S.INK2, va='top')
    S.save(fig, 'fig_params_kl16_sampler_vs_explicit')
    print('\nsampler (params, median, q25, q75, best lr):')
    for p in pts:
        print(f'  {p[0]:5d} {p[1]:.3f} [{p[2]:.3f}, {p[3]:.3f}] lr {p[4]:g}')
    print('explicit (group, params, median, q25, q75, n):')
    for row in printed:
        print('  {} {} {:.3f} [{:.3f}, {:.3f}] n={}'.format(*row))


if __name__ == '__main__':
    main()
