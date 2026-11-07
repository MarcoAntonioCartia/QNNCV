"""
Baseline-sweep analysis over logs/ (read-only)
==============================================

Groups the run directories under --logs by their hyper-parameter tag -- the
folder name minus the timestamp and the ``_seedN`` suffix, i.e. exactly what
``train_2d_qgan.build_hp_suffix`` produced -- plus a ``_warm`` marker when
``run_config.json`` records ``init_weights`` (a warm start is a design factor
that is deliberately absent from folder names). Per run it reads the metrics
at the checkpoint the trainer saved as ``best_weights.npy`` (the periodic
validation with the lowest nearest-member W1) and writes:

    <out>/runs.csv                   one row per run directory
    <out>/groups.csv                 one row per group (n, median, IQR, min, max, mean)
    <out>/summary.md                 the same tables in Markdown + data-quality notes
    <out>/<family>_best_metrics.png  per-seed distributions per group (box + dots)
    <out>/<family>_curves_nearW1.png val nearest-W1 vs epoch, median + IQR band

``--compare A B`` adds a paired-by-seed comparison of two groups (per-seed
deltas, wins, median delta, Wilcoxon signed-rank p) on the console and in
<out>/compare_<A>_vs_<B>.{csv,md}. A and B may be unique substrings of the
group names.

Reads only history.npz, run_config.json and norm_monitor.npz. Never writes
into a run directory, never trains, never imports TensorFlow or ``src``.

Usage (qnncv python, from the repo root):
    python scripts/analyze_sweep.py
    python scripts/analyze_sweep.py --family ring --n-seeds 30
    python scripts/analyze_sweep.py --family ring --compare ep100_dlr0.0005_gp10 ep100_dlr0.0005_gp10_encIS

Units: the W1 metrics are marginal W1 in grid cells; the energy distances are
in x/y units. ``diversity`` is the mean pairwise W1 between the generated
samples -- read it together with nearest-W1. The validation set's own
self-diversity (the reference for ``diversity``) is printed by the trainer at
startup and is NOT stored in history.npz, so it is not reported here.
Wall-clock figures come from file mtimes (run_config.json is written at start,
history.npz at the end) and are approximate.
"""
import argparse
import csv
import datetime as _dt
import glob
import json
import math
import os
import re
import sys
from collections import OrderedDict, defaultdict

import numpy as np

# ---------------------------------------------------------------- constants
RUN_PREFIX = 'qgan_2d_'
FAMILIES = ['four_gaussians', 'correlated', 'gaussian', 'ring', 'vibronic']
TS_RE = re.compile(r'^(?P<prefix>.*?)_(?P<ts>\d{8}_\d{6})(?P<suffix>.*)$')
SEED_RE = re.compile(r'_seed(?P<seed>\d+)$')

# short name -> history.npz key (order = report order)
VAL_KEYS = OrderedDict([
    ('nearW1',  'val_nearest_w1'),
    ('canonW1', 'val_wasserstein'),
    ('div',     'val_diversity'),
    ('nearED',  'val_nearest_ed'),
    ('canonED', 'val_canonical_ed'),
])
SELECT_KEY = {'nearest_w1': 'val_nearest_w1', 'canonical_w1': 'val_wasserstein'}

# run_config.json keys that legitimately differ between the seeds of one group
CFG_IGNORE = {'seed', 'timestamp', 'init_weights', 'deterministic',
              'TF_DETERMINISTIC_OPS', 'TF_ENABLE_ONEDNN_OPTS',
              'numpy_version', 'sf_version', 'tf_version', 'plot_every'}

# metric column -> description (order = report order)
STAT_METRICS = OrderedDict([
    ('best_nearW1',      'nearest-member W1 at the best checkpoint (grid cells)'),
    ('best_canonW1',     'canonical W1 at the best checkpoint (grid cells)'),
    ('best_div',         'diversity at the best checkpoint (mean pairwise W1, grid cells)'),
    ('best_nearED',      'nearest-member energy distance at the best checkpoint (x/y units)'),
    ('best_canonED',     'canonical energy distance at the best checkpoint (x/y units)'),
    ('best_epoch',       'epoch of the best checkpoint'),
    ('last_nearW1',      'nearest-member W1 at the last periodic validation'),
    ('ket_min',          'min batch-mean ket norm over training (history.npz)'),
    ('nm_min',           'min per-sample ket norm (norm_monitor.npz, interleaved runs only)'),
    ('wall_s_per_epoch', 'approx. wall-clock seconds per epoch (file mtimes)'),
])
STATS = ('n', 'median', 'q25', 'q75', 'min', 'max', 'mean')

# Re-scored metrics from scripts/rescore.py (analysis/rescore/rescore.csv,
# one row per run folder, 200 fixed-seed z draws): attached as rs_<col>.
RESCORE_COLS = OrderedDict([
    ('fid_w1',          're-scored fidelity: mean nearest-member W1 of the fakes (grid cells)'),
    ('cov_w1',          're-scored coverage: mean over val members of the nearest fake W1 (grid cells)'),
    ('chamfer_w1',      're-scored fidelity + coverage, W1'),
    ('fid_ed',          're-scored fidelity in energy distance (x/y units)'),
    ('cov_ed',          're-scored coverage in energy distance (x/y units)'),
    ('chamfer_ed',      're-scored fidelity + coverage, energy distance'),
    ('canon_w1_mean',   're-scored canonical W1'),
    ('diversity_w1',    're-scored diversity (mean pairwise W1 of the fakes)'),
    ('val_self_div_w1', 'validation-set self-diversity reference (mean pairwise W1)'),
    ('mmd2',            're-scored MMD^2 between fake and val sets'),
    ('mmd_p',           'permutation p-value of the two-sample MMD test'),
    ('ket_norm_min',    're-scored per-sample ket-norm minimum'),
    ('ket_norm_mean',   're-scored per-sample ket-norm mean'),
])
RESCORE_DEFAULT = os.path.join('analysis', 'rescore', 'rescore.csv')
for _col, _desc in RESCORE_COLS.items():
    STAT_METRICS['rs_' + _col] = _desc

# Fixed-order categorical palette and chart chrome (validated reference
# palette, light surface). Colour follows the group, never its rank.
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
           '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'

NAN = float('nan')


# ---------------------------------------------------------------- helpers
def is_nan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def fmt(v, nd=3):
    """Format a cell: '' for missing, fixed decimals for floats."""
    if is_nan(v):
        return ''
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{nd}f}"
    return str(v)


def compress_ranges(ints):
    """[0,1,2,5,7,8] -> '0-2,5,7-8'."""
    ints = sorted(set(int(i) for i in ints))
    if not ints:
        return ''
    parts, start, prev = [], ints[0], ints[0]
    for i in ints[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = i
    parts.append(f"{start}-{prev}" if start != prev else str(start))
    return ','.join(parts)


def parse_name(name):
    """Folder name -> (tag, timestamp, seed). Tag = name minus timestamp and _seedN."""
    seed = None
    base = name
    m = SEED_RE.search(name)
    if m:
        seed = int(m.group('seed'))
        base = name[:m.start()]
    m2 = TS_RE.match(base)
    if m2:
        return m2.group('prefix') + m2.group('suffix'), m2.group('ts'), seed
    return base, None, seed


def guess_family(tag):
    rest = tag[len(RUN_PREFIX):] if tag.startswith(RUN_PREFIX) else tag
    for fam in sorted(FAMILIES, key=len, reverse=True):
        if rest == fam or rest.startswith(fam + '_'):
            return fam
    return rest.split('_')[0] or 'unknown'


def short_label(group, family):
    pref = RUN_PREFIX + family
    lab = group[len(pref):] if group.startswith(pref) else group
    return lab.lstrip('_') or '(defaults)'


def hist_equal(h1, h2):
    if set(h1) != set(h2):
        return False
    for k in h1:
        a, b = h1[k], h2[k]
        try:
            if not np.array_equal(a, b, equal_nan=True):
                return False
        except TypeError:
            if not np.array_equal(a, b):
                return False
    return True


def nanstats(values):
    v = np.asarray([x for x in values if not is_nan(x)], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return OrderedDict((s, (0 if s == 'n' else NAN)) for s in STATS)
    return OrderedDict([
        ('n', int(v.size)), ('median', float(np.median(v))),
        ('q25', float(np.percentile(v, 25))), ('q75', float(np.percentile(v, 75))),
        ('min', float(v.min())), ('max', float(v.max())), ('mean', float(v.mean())),
    ])


# ---------------------------------------------------------------- loading
def load_run(path, select_key):
    name = os.path.basename(path)
    tag, ts, seed_from_name = parse_name(name)
    r = OrderedDict(name=name, tag=tag, timestamp=ts, dir=path, note='')

    cfg = None
    cfg_path = os.path.join(path, 'run_config.json')
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception as e:  # noqa: BLE001 - report, don't crash the sweep
            r['note'] = f'unreadable run_config.json ({e.__class__.__name__})'
    c = cfg or {}
    r['cfg'] = cfg
    r['family'] = c.get('family') or guess_family(tag)
    r['seed'] = c.get('seed', seed_from_name)
    if r['seed'] is None:
        r['seed'] = seed_from_name
    r['warm'] = bool(c.get('init_weights'))
    r['encoding'] = c.get('encoding') or 'input_displacement'
    r['group'] = tag + ('_warm' if r['warm'] else '')
    r['epochs_planned'] = c.get('epochs')
    r['val_every'] = int(c.get('val_every') or 20)

    hist = None
    hpath = os.path.join(path, 'history.npz')
    if not os.path.exists(hpath):
        r['status'] = 'no-history'
    else:
        try:
            with np.load(hpath, allow_pickle=True) as z:
                hist = {k: z[k] for k in z.files}
        except Exception as e:  # noqa: BLE001
            r['status'] = 'bad-history'
            r['note'] = f'unreadable history.npz ({e.__class__.__name__})'
    r['hist'] = hist

    for short in VAL_KEYS:
        r['best_' + short] = NAN
        r['last_' + short] = NAN
    r.update(epochs_done=None, n_val=0, best_idx=None, best_epoch=NAN,
             ket_last=NAN, ket_min=NAN, nm_min=NAN, nm_warn=None,
             wall_s_per_epoch=NAN, duplicate_of=None, identical=None)
    for col in RESCORE_COLS:
        r['rs_' + col] = NAN

    if hist is not None:
        per_epoch = hist.get('g_loss', hist.get('wasserstein'))
        done = int(len(per_epoch)) if per_epoch is not None else 0
        r['epochs_done'] = done
        if cfg is None:
            r['status'] = 'no-config'
        elif r['epochs_planned'] is not None and done < int(r['epochs_planned']):
            r['status'] = 'incomplete'
        else:
            r['status'] = 'complete'

        sel = hist.get(select_key)
        if sel is None and select_key == 'val_nearest_w1' and 'val_wasserstein' in hist:
            sel = hist['val_wasserstein']
            r['note'] = (r['note'] + '; ' if r['note'] else '') + \
                'no val_nearest_w1 in history: best checkpoint selected on canonical W1'
        if sel is not None and len(sel):
            sel = np.asarray(sel, dtype=float)
            r['n_val'] = int(sel.size)
            finite = np.isfinite(sel)
            if finite.any():
                idx = int(np.argmin(np.where(finite, sel, np.inf)))
                r['best_idx'] = idx
                r['best_epoch'] = float((idx + 1) * r['val_every'])
                for short, key in VAL_KEYS.items():
                    arr = hist.get(key)
                    if arr is None or len(arr) == 0:
                        continue
                    arr = np.asarray(arr, dtype=float)
                    if idx < arr.size:
                        r['best_' + short] = float(arr[idx])
                    r['last_' + short] = float(arr[-1])
        kn = hist.get('ket_norm')
        if kn is not None and len(kn):
            kn = np.asarray(kn, dtype=float)
            r['ket_last'] = float(kn[-1])
            r['ket_min'] = float(np.nanmin(kn)) if np.isfinite(kn).any() else NAN
        if cfg is not None and done > 0:
            dt = os.path.getmtime(hpath) - os.path.getmtime(cfg_path)
            r['wall_s_per_epoch'] = dt / done if dt > 0 else NAN

    nmp = os.path.join(path, 'norm_monitor.npz')
    if os.path.exists(nmp):
        try:
            with np.load(nmp, allow_pickle=True) as z:
                if 'ket_norm_min' in z.files and len(z['ket_norm_min']):
                    r['nm_min'] = float(np.nanmin(np.asarray(z['ket_norm_min'], dtype=float)))
                if 'warn_count' in z.files:
                    r['nm_warn'] = int(np.sum(z['warn_count']))
        except Exception as e:  # noqa: BLE001
            r['note'] = (r['note'] + '; ' if r['note'] else '') + \
                f'unreadable norm_monitor.npz ({e.__class__.__name__})'
    return r


def attach_rescore(runs, path):
    """Join scripts/rescore.py's csv (one row per run folder, column `run`)
    onto the runs as rs_<col>; returns (n_matched, n_rows, missing_cols)."""
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    by_run = {row.get('run', ''): row for row in rows}
    present = set(rows[0].keys()) if rows else set()
    missing = [c for c in RESCORE_COLS if c not in present]
    matched = 0
    for r in runs:
        row = by_run.get(r['name'])
        if row is None:
            continue
        matched += 1
        for col in RESCORE_COLS:
            v = row.get(col, '')
            try:
                r['rs_' + col] = float(v)
            except (TypeError, ValueError):
                r['rs_' + col] = NAN
    return matched, len(rows), missing


def mark_duplicates(runs):
    """Same group + same seed in several folders: keep the earliest as primary."""
    notes = []
    by_key = defaultdict(list)
    for r in runs:
        if r['seed'] is not None and r['hist'] is not None:
            by_key[(r['group'], r['seed'])].append(r)
    for (group, seed), rs in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if len(rs) < 2:
            continue
        rs.sort(key=lambda r: (r['timestamp'] or '99999999_999999', r['name']))
        primary = rs[0]
        for dup in rs[1:]:
            same = hist_equal(primary['hist'], dup['hist'])
            dup['duplicate_of'] = primary['name']
            dup['identical'] = same
            kind = 'identical history (re-run of the same seed)' if same else \
                   'DIFFERENT history (seed collision: code or config changed between runs?)'
            notes.append(f"{group} seed {seed}: `{dup['name']}` duplicates `{primary['name']}` -- {kind}; "
                         f"only the earlier run enters the statistics.")
    return notes


def usable(r, include_incomplete):
    if r['duplicate_of'] is not None:
        return False
    return r['status'] == 'complete' or (include_incomplete and r['status'] == 'incomplete')


# ---------------------------------------------------------------- grouping
def summarize_groups(runs, n_seeds, include_incomplete):
    """-> list of (summary dict, all runs of the group, usable runs), family-sorted."""
    groups = OrderedDict()
    key = lambda r: (r['family'], r['group'],  # noqa: E731
                     r['seed'] if r['seed'] is not None else -1, r['timestamp'] or '')
    for r in sorted(runs, key=key):
        groups.setdefault(r['group'], []).append(r)

    entries = []
    for g, rs in groups.items():
        us = [r for r in rs if usable(r, include_incomplete)]
        s = OrderedDict()
        s['group'] = g
        s['family'] = rs[0]['family']
        s['label'] = short_label(g, s['family'])
        s['encoding'] = '/'.join(sorted(set(r['encoding'] for r in rs)))
        s['warm'] = rs[0]['warm']
        s['epochs'] = '/'.join(str(e) for e in sorted(set(r['epochs_planned'] for r in rs
                                                           if r['epochs_planned'] is not None)))
        s['n_dirs'] = len(rs)
        s['n_usable'] = len(us)
        for st in ('complete', 'incomplete', 'no-history', 'bad-history', 'no-config'):
            s['n_' + st.replace('-', '_')] = sum(1 for r in rs if r['status'] == st)
        s['n_duplicate'] = sum(1 for r in rs if r['duplicate_of'] is not None)
        seeds_us = sorted(r['seed'] for r in us if r['seed'] is not None)
        seeds_all = [r['seed'] for r in rs if r['seed'] is not None]
        n_exp = n_seeds if n_seeds else ((max(seeds_all) + 1) if seeds_all else 0)
        s['seeds_usable'] = compress_ranges(seeds_us)
        s['seeds_missing'] = compress_ranges(set(range(n_exp)) - set(seeds_us))
        s['incomplete_runs'] = ', '.join(f"seed{r['seed']}@{r['epochs_done']}/{r['epochs_planned']}"
                                         for r in rs if r['status'] == 'incomplete')
        for m in STAT_METRICS:
            for st, v in nanstats([r[m] for r in us]).items():
                s[f'{m}_{st}'] = v
        # config consistency inside the group
        cfgs = [r['cfg'] for r in rs if r['cfg'] is not None]
        diff = set()
        for cfg in cfgs[1:]:
            for k in set(cfgs[0]) | set(cfg):
                if k in CFG_IGNORE:
                    continue
                if cfgs[0].get(k) != cfg.get(k):
                    diff.add(k)
        s['cfg_mismatch'] = ','.join(sorted(diff))
        entries.append((s, rs, us))
    return entries


# ---------------------------------------------------------------- output: console / csv / md
def visible(entries, show_empty):
    """Groups shown in the tables; empty groups (no usable run) are hidden by default."""
    shown = [e for e in entries if show_empty or e[0]['n_usable'] > 0]
    hidden = [e[0]['group'] for e in entries if e not in shown]
    return shown, hidden


def console_table(entries, show_empty=False):
    shown, hidden = visible(entries, show_empty)
    hdr = f"{'group':<58s} {'n':>3s} {'nearW1 med [q25,q75]':>22s} {'div med':>8s} {'nearED':>7s}  missing"
    print(hdr)
    print('-' * len(hdr))
    for s, _, _ in shown:
        if s['n_usable'] == 0:
            cell = '-'
        else:
            cell = f"{s['best_nearW1_median']:.3f} [{s['best_nearW1_q25']:.3f},{s['best_nearW1_q75']:.3f}]"
        miss = s['seeds_missing'] or '-'
        if s['incomplete_runs']:
            miss += f"  (incomplete: {s['incomplete_runs']})"
        print(f"{s['group'][:58]:<58s} {s['n_usable']:>3d} {cell:>22s} "
              f"{fmt(s['best_div_median']):>8s} {fmt(s['best_nearED_median']):>7s}  {miss}")
    if hidden:
        print(f"({len(hidden)} groups without a usable run hidden; --show-empty lists them: "
              + ', '.join(hidden) + ')')


RUN_FIELDS = (['name', 'group', 'family', 'encoding', 'seed', 'warm', 'timestamp', 'status',
               'epochs_planned', 'epochs_done', 'val_every', 'n_val', 'best_idx', 'best_epoch']
              + ['best_' + k for k in VAL_KEYS] + ['last_' + k for k in VAL_KEYS]
              + ['ket_last', 'ket_min', 'nm_min', 'nm_warn', 'wall_s_per_epoch']
              + ['rs_' + c for c in RESCORE_COLS]
              + ['duplicate_of', 'identical', 'note', 'dir'])


def write_runs_csv(path, runs):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        w.writeheader()
        for r in sorted(runs, key=lambda r: (r['family'], r['group'],
                                             r['seed'] if r['seed'] is not None else -1,
                                             r['timestamp'] or '')):
            w.writerow({k: fmt(r.get(k), 6) for k in RUN_FIELDS})


def write_groups_csv(path, entries):
    fields = list(entries[0][0].keys()) if entries else ['group']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s, _, _ in entries:
            w.writerow({k: fmt(v, 6) for k, v in s.items()})


def md_table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|']
    for row in rows:
        out.append('| ' + ' | '.join(str(c) for c in row) + ' |')
    return '\n'.join(out)


def write_summary_md(path, entries, runs, dup_notes, args):
    L = []
    L.append('# Sweep analysis')
    L.append('')
    L.append(f"Generated {_dt.datetime.now():%Y-%m-%d %H:%M} by `scripts/analyze_sweep.py` "
             f"over `{args.logs}` ({len(runs)} run directories"
             + (f", families {', '.join(args.family)}" if args.family else '') + ').')
    L.append('')
    L.append('Best checkpoint = the periodic validation with the lowest '
             + ('nearest-member W1' if args.select == 'nearest_w1' else 'canonical W1')
             + ' (the weights saved as `best_weights.npy`). W1 metrics are marginal W1 in grid '
             'cells; energy distances are in x/y units; `div` is the mean pairwise W1 between '
             'generated samples (the validation set\'s own self-diversity is not stored in '
             'history.npz and is not shown). Statistics use complete runs only'
             + (' plus incomplete runs (--include-incomplete)' if args.include_incomplete else '')
             + ', one per (group, seed). Wall-clock per epoch is approximate (file mtimes).')
    L.append('')

    L.append('## Groups')
    L.append('')
    shown, hidden = visible(entries, args.show_empty)
    hdr = ['group', 'family', 'epochs', 'n', 'nearW1 median [IQR]', 'canonW1 median',
           'div median [IQR]', 'nearED median [IQR]', 'best epoch median', 's/epoch', 'missing seeds']
    rows = []
    for s, _, _ in shown:
        if s['n_usable'] == 0:
            rows.append([f"`{s['group']}`", s['family'], s['epochs'], 0, '-', '-', '-', '-', '-', '-',
                         s['seeds_missing'] or '-'])
            continue
        rows.append([
            f"`{s['group']}`", s['family'], s['epochs'], s['n_usable'],
            f"{s['best_nearW1_median']:.3f} [{s['best_nearW1_q25']:.3f}, {s['best_nearW1_q75']:.3f}]",
            fmt(s['best_canonW1_median']),
            f"{fmt(s['best_div_median'])} [{fmt(s['best_div_q25'])}, {fmt(s['best_div_q75'])}]",
            f"{fmt(s['best_nearED_median'])} [{fmt(s['best_nearED_q25'])}, {fmt(s['best_nearED_q75'])}]",
            fmt(s['best_epoch_median'], 0), fmt(s['wall_s_per_epoch_median'], 1),
            (s['seeds_missing'] or '-') + (f" (incomplete: {s['incomplete_runs']})" if s['incomplete_runs'] else ''),
        ])
    L.append(md_table(hdr, rows))
    L.append('')
    if hidden:
        L.append(f"{len(hidden)} groups without a usable run are not listed (`--show-empty` lists them): "
                 + ', '.join(f'`{g}`' for g in hidden) + '. They still appear in the data-quality notes.')
        L.append('')

    L.append('## Full statistics per group')
    L.append('')
    for s, _, us in entries:
        if s['n_usable'] == 0:
            continue
        L.append(f"### `{s['group']}`  (family {s['family']}, encoding {s['encoding']}, "
                 f"{s['n_usable']} usable of {s['n_dirs']} dirs, seeds {s['seeds_usable'] or '-'})")
        L.append('')
        rows = []
        for m, desc in STAT_METRICS.items():
            if s[f'{m}_n'] == 0:
                continue
            nd = 0 if m == 'best_epoch' else (1 if m == 'wall_s_per_epoch' else 3)
            rows.append([m, s[f'{m}_n']] + [fmt(s[f'{m}_{st}'], nd) for st in STATS[1:]] + [desc])
        L.append(md_table(['metric', 'n', 'median', 'q25', 'q75', 'min', 'max', 'mean', 'description'], rows))
        L.append('')

    L.append('## Data-quality notes')
    L.append('')
    notes = []
    for s, rs, _ in entries:
        for r in rs:
            if r['status'] == 'incomplete':
                notes.append(f"incomplete: `{r['name']}` stopped at epoch {r['epochs_done']}/{r['epochs_planned']}")
            elif r['status'] == 'no-history':
                notes.append(f"no history.npz (never finished or crashed early): `{r['name']}`")
            elif r['status'] == 'bad-history':
                notes.append(f"unreadable history.npz: `{r['name']}` ({r['note']})")
            elif r['status'] == 'no-config':
                notes.append(f"no run_config.json (pre-`--seed` era run, excluded from statistics): `{r['name']}`")
            if r['note'] and r['status'] not in ('bad-history',):
                notes.append(f"`{r['name']}`: {r['note']}")
        if s['cfg_mismatch']:
            notes.append(f"config mismatch inside `{s['group']}`: keys {s['cfg_mismatch']} differ between seeds")
    notes.extend(dup_notes)
    if not notes:
        notes.append('none')
    L.extend(f'- {n}' for n in notes)
    L.append('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


# ---------------------------------------------------------------- plots
def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _slot_color(slot):
    return PALETTE[slot] if slot < len(PALETTE) else MUTED


def family_order(entries):
    """Groups of one family ordered by usable-run count (desc) then name: the
    biggest groups take the first palette slots, consistently across figures."""
    return sorted(entries, key=lambda e: (-e[0]['n_usable'], e[0]['group']))


def plot_family_best(family, entries, out_dir, min_runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ordered = family_order(entries)
    shown = [(i, s, us) for i, (s, _, us) in enumerate(ordered) if len(us) >= min_runs]
    if not shown:
        return None
    panels = [('best_nearW1', 'nearest-member W1\n(grid cells, lower is better)'),
              ('best_div', 'diversity\n(mean pairwise W1, grid cells)'),
              ('best_nearED', 'nearest-member energy distance\n(x/y units, lower is better)')]
    n = len(shown)
    fig, axes = plt.subplots(len(panels), 1, figsize=(max(7.0, 1.25 * n + 2.5), 9.5),
                             sharex=True, facecolor=SURFACE)
    rng = np.random.default_rng(0)
    for ax, (m, ylabel) in zip(axes, panels):
        _style_axes(ax)
        for pos, (slot, s, us) in enumerate(shown):
            vals = np.asarray([r[m] for r in us], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            if vals.size >= 3:
                ax.boxplot(vals, positions=[pos], widths=0.5, showfliers=False,
                           boxprops=dict(color=AXIS, linewidth=1),
                           whiskerprops=dict(color=AXIS, linewidth=1),
                           capprops=dict(color=AXIS, linewidth=1),
                           medianprops=dict(color=INK, linewidth=1.6))
            x = pos + rng.uniform(-0.16, 0.16, size=vals.size)
            ax.scatter(x, vals, s=20, color=_slot_color(slot), alpha=0.9,
                       edgecolors=SURFACE, linewidths=0.6, zorder=3)
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    axes[-1].set_xticks(range(n))
    axes[-1].set_xticklabels([f"{s['label'].replace('_', chr(10))}\nn={len(us)}" for _, s, us in shown],
                             fontsize=8, color=INK2)
    axes[-1].set_xlim(-0.6, n - 0.4)
    axes[0].set_title(f"{family}: metrics at the best checkpoint, one dot per seed",
                      color=INK, fontsize=11, loc='left')
    fig.tight_layout()
    path = os.path.join(out_dir, f'{family}_best_metrics.png')
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_family_curves(family, entries, out_dir, min_runs, max_groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ordered = family_order(entries)
    shown = [(i, s, us) for i, (s, _, us) in enumerate(ordered) if len(us) >= min_runs][:max_groups]
    if not shown:
        return None
    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor=SURFACE)
    _style_axes(ax)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    for slot, s, us in shown:
        runs_ok = [r for r in us if r['hist'] is not None and 'val_nearest_w1' in r['hist']]
        if not runs_ok:
            continue
        n_val = min(len(r['hist']['val_nearest_w1']) for r in runs_ok)
        if n_val == 0:
            continue
        mat = np.asarray([np.asarray(r['hist']['val_nearest_w1'][:n_val], dtype=float) for r in runs_ok])
        mat = np.where(np.isfinite(mat), mat, np.nan)
        x = (np.arange(n_val) + 1) * runs_ok[0]['val_every']
        med = np.nanmedian(mat, axis=0)
        color = _slot_color(slot)
        ax.plot(x, med, color=color, linewidth=2, label=f"{s['label']} (n={len(runs_ok)})", zorder=3)
        if len(runs_ok) >= 3:
            ax.fill_between(x, np.nanpercentile(mat, 25, axis=0), np.nanpercentile(mat, 75, axis=0),
                            color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel('epoch', color=INK2, fontsize=9)
    ax.set_ylabel('validation nearest-member W1 (grid cells)', color=INK2, fontsize=9)
    ax.set_title(f"{family}: validation nearest-member W1 vs epoch (median, IQR band for n >= 3)",
                 color=INK, fontsize=11, loc='left')
    ax.legend(fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1.0), labelcolor=INK2)
    fig.tight_layout()
    path = os.path.join(out_dir, f'{family}_curves_nearW1.png')
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    return path


# ---------------------------------------------------------------- compare
def resolve_group(query, group_names):
    if query in group_names:
        return query
    cands = [g for g in group_names if query in g]
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise SystemExit(f"--compare: no group matches '{query}'. Groups:\n  " + '\n  '.join(group_names))
    raise SystemExit(f"--compare: '{query}' is ambiguous, matches:\n  " + '\n  '.join(cands))


COMPARE_METRICS = [('best_nearW1', 'lower'), ('best_div', None), ('best_nearED', 'lower'),
                   ('best_canonW1', 'lower'), ('best_epoch', None), ('nm_min', None),
                   # re-scored (200 samples) — present only when rescore.csv was joined
                   ('rs_fid_w1', 'lower'), ('rs_cov_w1', 'lower'), ('rs_chamfer_w1', 'lower'),
                   ('rs_fid_ed', 'lower'), ('rs_cov_ed', 'lower'), ('rs_chamfer_ed', 'lower'),
                   ('rs_diversity_w1', None), ('rs_mmd2', 'lower'), ('rs_mmd_p', None)]


def compare_groups(ga, gb, entries, out_dir):
    by_group = {s['group']: (s, us) for s, _, us in entries}
    sa, ua = by_group[ga]
    sb, ub = by_group[gb]
    A = {r['seed']: r for r in ua if r['seed'] is not None}
    B = {r['seed']: r for r in ub if r['seed'] is not None}
    seeds = sorted(set(A) & set(B))
    L = [f"# Paired comparison: A = `{ga}`  vs  B = `{gb}`", '']
    if sa['epochs'] != sb['epochs']:
        L.append(f"**Warning: training budgets differ (A {sa['epochs']} vs B {sb['epochs']} epochs); "
                 'differences are confounded by training length.**')
        L.append('')
    L.append(f"Paired on {len(seeds)} seeds present and complete in both groups "
             f"({compress_ranges(seeds) or 'none'}); A has {len(A)} usable seeds, B has {len(B)}. "
             'delta = B - A.')
    L.append('')
    if not seeds:
        text = '\n'.join(L)
        print(text)
        return text
    hdr = ['seed', 'nearW1 A', 'nearW1 B', 'delta', 'div A', 'div B', 'delta', 'nearED A', 'nearED B', 'delta']
    rows = []
    for sd in seeds:
        ra, rb = A[sd], B[sd]
        row = [sd]
        for m in ('best_nearW1', 'best_div', 'best_nearED'):
            row += [fmt(ra[m]), fmt(rb[m]), fmt(rb[m] - ra[m]) if not (is_nan(ra[m]) or is_nan(rb[m])) else '']
        rows.append(row)
    L.append(md_table(hdr, rows))
    L.append('')
    try:
        from scipy.stats import wilcoxon
    except Exception:  # noqa: BLE001
        wilcoxon = None
    hdr = ['metric', 'n', 'median A', 'median B', 'median delta', 'mean delta', 'B lower / ties / B higher',
           'Wilcoxon p (two-sided)', 'direction']
    rows = []
    for m, better in COMPARE_METRICS:
        a = np.asarray([A[s][m] for s in seeds], dtype=float)
        b = np.asarray([B[s][m] for s in seeds], dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        n = int(ok.sum())
        if n == 0:
            continue
        d = b - a
        p = ''
        if wilcoxon is not None and n >= 5 and np.any(d != 0):
            try:
                p = f"{float(wilcoxon(b, a).pvalue):.3f}"
            except Exception:  # noqa: BLE001
                p = ''
        elif n < 5:
            p = 'n<5'
        nd = 0 if m == 'best_epoch' else 3
        rows.append([m, n, fmt(np.median(a), nd), fmt(np.median(b), nd), fmt(np.median(d), nd), fmt(d.mean(), nd),
                     f"{int((d < 0).sum())} / {int((d == 0).sum())} / {int((d > 0).sum())}", p,
                     'lower is better' if better == 'lower' else 'no direction'])
    L.append(md_table(hdr, rows))
    L.append('')
    L.append('With n < 10 pairs the Wilcoxon test has little power; read the per-seed deltas, not the p-value alone.')
    text = '\n'.join(L)
    print(text)
    if out_dir:
        base = os.path.join(out_dir, f"compare_{short_label(ga, sa['family'])}_vs_{short_label(gb, sb['family'])}")
        base = re.sub(r'[^A-Za-z0-9_.\-]+', '-', os.path.basename(base))
        base = os.path.join(out_dir, base)
        with open(base + '.md', 'w', encoding='utf-8') as f:
            f.write(text + '\n')
        with open(base + '.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['seed'] + [f'{m}_{side}' for m, _ in COMPARE_METRICS for side in ('A', 'B', 'delta')])
            for sd in seeds:
                row = [sd]
                for m, _ in COMPARE_METRICS:
                    va, vb = A[sd][m], B[sd][m]
                    row += [fmt(va, 6), fmt(vb, 6), fmt(vb - va, 6) if not (is_nan(va) or is_nan(vb)) else '']
                w.writerow(row)
        print(f"\nwritten: {base}.md, {base}.csv")
    return text


# ---------------------------------------------------------------- main
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument('--logs', default='logs', help='directory holding the qgan_2d_* run folders (default: logs)')
    ap.add_argument('--out', default=os.path.join('analysis', 'sweep'),
                    help='output directory for csv/md/png (default: analysis/sweep)')
    ap.add_argument('--no-write', action='store_true', help='console output only, write nothing')
    ap.add_argument('--no-plots', action='store_true', help='skip the png figures')
    ap.add_argument('--family', nargs='*', default=None, help='only these families (default: all)')
    ap.add_argument('--rescore', default=None, metavar='CSV',
                    help=f'join scripts/rescore.py output (default: {RESCORE_DEFAULT} when it exists; '
                         f'"none" to skip); adds rs_<col> metrics to the tables and --compare')
    ap.add_argument('--since', default=None, metavar='YYYYMMDD',
                    help='only runs whose folder timestamp is on/after this date (folders without a '
                         'timestamp are dropped); separates batches that share a tag, e.g. a 500-epoch '
                         'arm rerun months after the first one')
    ap.add_argument('--n-seeds', type=int, default=None,
                    help='expected seeds per group for the missing-seed report (default: max seed seen + 1)')
    ap.add_argument('--select', choices=sorted(SELECT_KEY), default='nearest_w1',
                    help='validation metric that defines the best checkpoint (default: nearest_w1, as in GAN '
                         'mode; use canonical_w1 for Path A runs)')
    ap.add_argument('--include-incomplete', action='store_true',
                    help='let runs that stopped before their planned epochs enter the statistics')
    ap.add_argument('--show-empty', action='store_true',
                    help='also list groups that have no usable run (default: hidden, noted in the summary)')
    ap.add_argument('--min-runs-plot', type=int, default=1, help='hide groups with fewer usable runs from figures')
    ap.add_argument('--max-curve-groups', type=int, default=len(PALETTE),
                    help='largest groups to draw in the learning-curve figure (default: 8)')
    ap.add_argument('--compare', nargs=2, metavar=('GROUP_A', 'GROUP_B'),
                    help='paired-by-seed comparison of two groups (names or unique substrings)')
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not os.path.isdir(args.logs):
        print(f"ERROR: --logs directory not found: {args.logs}")
        return 2
    dirs = sorted(d for d in glob.glob(os.path.join(args.logs, RUN_PREFIX + '*')) if os.path.isdir(d))
    if not dirs:
        print(f"ERROR: no {RUN_PREFIX}* run directories under {args.logs}")
        return 2
    runs = [load_run(d, SELECT_KEY[args.select]) for d in dirs]
    if args.family:
        runs = [r for r in runs if r['family'] in set(args.family)]
        if not runs:
            print(f"ERROR: no runs for families {args.family}")
            return 2
    if args.since:
        runs = [r for r in runs if (r['timestamp'] or '') >= args.since]
        if not runs:
            print(f"ERROR: no runs with a folder timestamp on/after {args.since}")
            return 2
    rescore_path = args.rescore
    if rescore_path is None and os.path.exists(RESCORE_DEFAULT):
        rescore_path = RESCORE_DEFAULT
    if rescore_path and rescore_path.lower() != 'none':
        if not os.path.exists(rescore_path):
            print(f"ERROR: --rescore file not found: {rescore_path}")
            return 2
        matched, n_rows, missing = attach_rescore(runs, rescore_path)
        print(f"re-score joined from {rescore_path}: {matched} of {len(runs)} runs matched "
              f"({n_rows} rows in the file)" + (f"; columns missing: {missing}" if missing else ''))
    dup_notes = mark_duplicates(runs)
    entries = summarize_groups(runs, args.n_seeds, args.include_incomplete)

    print(f"{len(runs)} run directories -> {len(entries)} groups "
          f"({sum(1 for r in runs if r['status'] == 'complete')} complete, "
          f"{sum(1 for r in runs if r['status'] == 'incomplete')} incomplete, "
          f"{sum(1 for r in runs if r['status'] in ('no-history', 'bad-history'))} without history, "
          f"{sum(1 for r in runs if r['status'] == 'no-config')} without run_config, "
          f"{sum(1 for r in runs if r['duplicate_of'] is not None)} duplicate seeds)\n")
    console_table(entries, args.show_empty)

    out_dir = None
    if not args.no_write:
        out_dir = args.out
        os.makedirs(out_dir, exist_ok=True)
        write_runs_csv(os.path.join(out_dir, 'runs.csv'), runs)
        write_groups_csv(os.path.join(out_dir, 'groups.csv'), entries)
        write_summary_md(os.path.join(out_dir, 'summary.md'), entries, runs, dup_notes, args)
        written = ['runs.csv', 'groups.csv', 'summary.md']
        if not args.no_plots:
            fams = OrderedDict()
            for e in entries:
                fams.setdefault(e[0]['family'], []).append(e)
            for fam, es in fams.items():
                for fn in (plot_family_best(fam, es, out_dir, args.min_runs_plot),
                           plot_family_curves(fam, es, out_dir, args.min_runs_plot, args.max_curve_groups)):
                    if fn:
                        written.append(os.path.basename(fn))
        print(f"\nwritten to {out_dir}: " + ', '.join(written))

    if args.compare:
        names = [s['group'] for s, _, _ in entries]
        ga, gb = (resolve_group(q, names) for q in args.compare)
        print()
        compare_groups(ga, gb, entries, out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
