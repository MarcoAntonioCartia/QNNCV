"""
Shared style, cell map and loaders for the thesis figures (eval-only; never
trains, never imports TensorFlow).

Every figure script in this folder imports this module so that the three
generator arms keep one colour, one marker and one line style across the
whole book, and so that every figure is sized to the book's text width
(tesis-usb.cls: letter paper, 3 cm + 2.5 cm margins -> 16.09 cm).

Labels are in Spanish with the book's Spanish acronyms (VC, VD); edit LABEL
in one place to change them everywhere.
"""

import csv
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))           # scripts/ (analyze_sweep: no TF import)
from analyze_sweep import paired_bootstrap, iqm      # noqa: E402,F401

OUT_DIR = os.path.join(REPO, 'analysis', 'figures')
TEXTWIDTH = 16.09 / 2.54                             # inches

# Fixed-order categorical palette (same validated reference palette as
# scripts/analyze_sweep.py). Colour follows the arm, never its rank.
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURFACE = '#e1e0d9', '#c3c2b7', '#ffffff'

ARMS = ['classical', 'cv', 'dv']
ARM = {   # colour, marker, line style (identity is never carried by colour alone)
    'classical': dict(color=PALETTE[0], marker='o', ls='-'),
    'cv':        dict(color=PALETTE[1], marker='s', ls='--'),
    'dv':        dict(color=PALETTE[2], marker='^', ls=':'),
}
LABEL = {
    'classical': 'Clásico (181 p.)', 'cv': 'VC (186 p.)', 'dv': 'VD (180 p.)',
    'classical_short': 'Clásico', 'cv_short': 'VC', 'dv_short': 'VD',
    'dense': 'denso\n(16 593 p.)', 'coord': 'puntual clásico\n(105 p.)',
    'cvq': 'cuántico VC\n(96 p.)', 'dvq': 'cuántico VD\n(96 p.)',
    'fid_w1': 'W$_1$ de fidelidad', 'cov_w1': 'W$_1$ de cobertura', 'mmd2': 'MMD$^2$',
    'kl16_mixture': 'KL$_{16}$ de la mezcla', 'diversity': 'diversidad / validación',
    'epoch': 'época', 'params': 'parámetros entrenables',
}
CRITICS = ['dense', 'coord', 'cvq', 'dvq']

# Ring matrix cells (32x32 grid, 300 epochs, d_lr 0.0005, seeds 0-19).
# Group names are rescore.group_of(run folder). DV = tuned settings (ls1, m3).
_GEN_PREFIX = {
    'cv':        'qgan_2d_ring_m3_ep300_dlr0.0005_gp10_gs32',
    'classical': 'qgan_2d_ring_m3_ep300_glr0.05_dlr0.0005_gs32_gcl',
    'dv':        'qgan_2d_ring_m3_ep300_glr0.05_dlr0.0005_ls1_gs32_gdv',
}
_CRIT_SUFFIX = {'dense': '', 'coord': '_critcoord', 'cvq': '_critcvq', 'dvq': '_critdvq_clay12'}


def cell_group(gen, critic):
    return _GEN_PREFIX[gen] + _CRIT_SUFFIX[critic]


def apply_style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
        'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK, 'axes.titlecolor': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.axisbelow': True, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'lines.linewidth': 1.6, 'legend.frameon': False, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02, 'pdf.fonttype': 42,
    })


def save(fig, name):
    """Vector PDF for Overleaf plus a PNG preview, both in analysis/figures/."""
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = []
    for ext, kw in (('pdf', {}), ('png', {'dpi': 200})):
        p = os.path.join(OUT_DIR, f'{name}.{ext}')
        fig.savefig(p, **kw)
        paths.append(p)
    plt.close(fig)
    print('written', ' '.join(os.path.relpath(p, REPO) for p in paths))
    return paths


# ---------------------------------------------------------------- loaders
def read_csv(path):
    path = os.path.join(REPO, path)
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def by_seed(rows, group, col):
    """{seed: float(col)} for one group; later rows win (re-scores overwrite)."""
    out = {}
    for r in rows:
        if r.get('group') == group and r.get(col) not in (None, '', 'nan'):
            out[int(float(r['seed']))] = float(r[col])
    return out


def run_dirs(group, epochs=300):
    """Completed run folders of a group (best_weights + history present)."""
    import re
    ts = re.compile(r'^(qgan_2d_ring)_\d{8}_\d{6}_(.*)_seed(\d+)$')
    out = []
    for d in glob.glob(os.path.join(REPO, 'logs', 'qgan_2d_ring_*')):
        m = ts.match(os.path.basename(d))
        if m and f'{m.group(1)}_{m.group(2)}' == group and os.path.exists(os.path.join(d, 'history.npz')):
            out.append((int(m.group(3)), d))
    seen = {}
    for s, d in sorted(out):          # latest timestamp per seed wins
        seen[s] = d
    return seen


def band(curves):
    """(n_runs, T) -> median, q25, q75, min, max along runs (NaN-safe)."""
    C = np.asarray(curves, dtype=float)
    return (np.nanmedian(C, 0), np.nanpercentile(C, 25, 0), np.nanpercentile(C, 75, 0),
            np.nanmin(C, 0), np.nanmax(C, 0))
