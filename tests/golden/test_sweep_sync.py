"""
Sync test: run_sweep.py's SWEEP_PARAMS can never silently drift from the
train_2d_qgan.py CLI again.

Checks (no golden data needed — the contract is structural):
1. Every SWEEP_PARAMS entry names a real CLI flag with the same type.
2. Every abbreviation matches what build_hp_suffix actually puts in the run
   folder name (behavioral check — this is the contract run_sweep's
   find_log_dir relies on to locate finished runs).
3. Reverse guard: every CLI flag absent from SWEEP_PARAMS is in the known
   non-sweepable set (catches a new training flag forgotten in run_sweep).
4. Bounds sanity: vmin <= CLI default <= vmax where a default exists.
5. find_log_dir resolves a sweep label to the right run folder by whole-tag
   matching: default-valued overrides (m2, ss0) have no folder tag, and
   'ss0' must not match 'ss0.4' (regression: the summary table once showed
   one folder for every experiment).
"""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO, load_t2q, main_dispatch

NON_SWEEPABLE = {
    '--family', '--n-train', '--n-val', '--n-ancilla', '--no-kerr',
    '--plot-every', '--val-every', '--deterministic', '--help', '-h',
    # interleaved_squeeze encoding flags: --encoding is categorical (string)
    # and SWEEP_PARAMS only supports numeric ranges — the A/B sweep passes it
    # as a fixed passthrough. The rest are per-run config, not sweep axes.
    '--encoding', '--inj-layers', '--r-max', '--init-weights', '--norm-floor',
    # boolean switch (store_true): the trainable-gain ablation is a fixed
    # passthrough per arm, like --encoding; --squeeze-scale itself IS a
    # sweep axis (numeric) and lives in SWEEP_PARAMS.
    '--train-squeeze-scale',
    # categorical (shared | dedicated): passthrough per arm
    '--inj-latent',
}


def load_run_sweep():
    spec = importlib.util.spec_from_file_location(
        'run_sweep', os.path.join(REPO, 'run_sweep.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test():
    t2q = load_t2q()
    sweep = load_run_sweep()
    parser = t2q.build_parser()

    actions = {}
    for action in parser._actions:
        for opt in action.option_strings:
            actions[opt] = action

    # (1) + (2) + (4)
    for name, (flag, ptype, abbrev, vmin, vmax) in sweep.SWEEP_PARAMS.items():
        assert flag in actions, (
            f"SWEEP_PARAMS[{name!r}] names {flag!r}, which is not a "
            f"train_2d_qgan.py CLI flag")
        action = actions[flag]
        assert action.type is ptype, (
            f"SWEEP_PARAMS[{name!r}] type {ptype} != CLI type {action.type} "
            f"for {flag}")

        default = parser.get_default(action.dest)
        if default is not None:
            assert vmin <= default <= vmax, (
                f"SWEEP_PARAMS[{name!r}] bounds [{vmin}, {vmax}] exclude the "
                f"CLI default {default} for {flag}")

        # behavioral abbreviation check: a non-default value must show up in
        # the folder-name suffix as <abbrev><value>
        sample = ptype(vmin) if ptype(vmin) != default else ptype(vmax)
        args = parser.parse_args(['--family', 'ring', flag, str(sample)])
        # replicate main()'s pre-suffix mutations
        args.seed = t2q.resolve_seed(args.seed)
        if args.n_ancilla is not None:
            args.n_modes = 2 + args.n_ancilla
        suffix = t2q.build_hp_suffix(args, parser)
        expected = f"{abbrev}{t2q._format_val(sample)}"
        assert expected in suffix.split('_'), (
            f"SWEEP_PARAMS[{name!r}] abbrev {abbrev!r}: expected {expected!r} "
            f"in folder suffix, got {suffix!r} — run_sweep.find_log_dir would "
            f"not locate this run")
        print(f"  ok: {name:20s} {flag:22s} -> {expected}")

    # (3) reverse guard
    sweep_flags = {flag for flag, *_ in sweep.SWEEP_PARAMS.values()}
    for opt, action in actions.items():
        if opt not in sweep_flags:
            assert opt in NON_SWEEPABLE, (
                f"CLI flag {opt!r} is neither in SWEEP_PARAMS nor in the "
                f"known non-sweepable set — new training flag forgotten in "
                f"run_sweep.py?")
    print(f"  ok: reverse guard ({len(actions)} CLI options accounted for)")

    # (5) find_log_dir: whole-token matching with default-valued overrides
    import tempfile
    from datetime import datetime
    tmp = tempfile.mkdtemp(prefix='qnncv_sweep_sync_')
    names = ['qgan_2d_ring_20261107_175636_nl2_c6_ep50_encIS_seed0',
             'qgan_2d_ring_20261107_175637_nl2_c6_ep50_encIS_ss0.2_seed0',
             'qgan_2d_ring_20261107_175637_nl2_c6_ep50_encIS_ss0.4_seed0',
             'qgan_2d_ring_20261107_175638_nl2_c6_ep50_encIS_ss0.2_seed1']
    for n in names:
        os.makedirs(os.path.join(tmp, 'logs', n))
    start = datetime(2026, 11, 7, 17, 56, 0)
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        cases = [('c6_ep50_nl2_m2_seed0_ss0', names[0]),
                 ('c6_ep50_nl2_m2_seed0_ss0.2', names[1]),
                 ('c6_ep50_nl2_m2_seed0_ss0.4', names[2]),
                 ('c6_ep50_nl2_m2_seed1_ss0.2', names[3])]
        for label, want in cases:
            got = sweep.find_log_dir(label, 'ring', start)
            assert got is not None and got.name == want, (
                f"find_log_dir({label!r}) -> {got}, expected {want}")
        assert sweep.find_log_dir('c6_ep50_nl2_m2_seed0_ss0', 'ring',
                                  datetime(2026, 11, 8)) is None, (
            "runs older than start_time must not match")
    finally:
        os.chdir(cwd)
    print("  ok: find_log_dir whole-token matching "
          "(m2/ss0 default-valued, ss0 vs ss0.4, seed0 vs seed1)")


def generate(force=False):
    # Structural test — no golden data to generate; run the checks instead.
    test()


if __name__ == '__main__':
    main_dispatch(generate, test, 'test_sweep_sync')
