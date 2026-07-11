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
"""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import REPO, load_t2q, main_dispatch

NON_SWEEPABLE = {
    '--family', '--n-train', '--n-val', '--n-ancilla', '--no-kerr',
    '--plot-every', '--val-every', '--deterministic', '--help', '-h',
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


def generate(force=False):
    # Structural test — no golden data to generate; run the checks instead.
    test()


if __name__ == '__main__':
    main_dispatch(generate, test, 'test_sweep_sync')
