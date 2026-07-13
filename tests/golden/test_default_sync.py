"""
Sync test: internal signature defaults can never silently diverge from the
train_2d_qgan.py CLI defaults again (the default-unification cleanup).

For every parameter shared between build_parser() and an internal signature
(train_2d_qgan, CVQGANGenerator.__init__, Discriminator2D.__init__), assert
the signature has NO default (required keyword-only — the argparse default
in build_parser() is the single source of truth) OR its default equals the
CLI default. Checks (no golden data needed — the contract is structural):

1. Every mapped pair agrees (or the signature side is required).
2. Completeness, both directions: every CLI dest is mapped or explicitly
   excluded, and every signature parameter is mapped, excluded, or in the
   known signature-only set — so a future flag or parameter cannot escape
   the check.
"""

import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import load_t2q, main_dispatch

INVERTED = 'inverted'  # sig default must equal (not CLI default)

# argparse dest -> train_2d_qgan parameter
TRAIN_MAP = {
    'family': 'family_name',
    'n_train': 'n_train',
    'n_val': 'n_val',
    'n_modes': 'n_total_modes',
    'n_layers': 'n_layers',
    'cutoff_dim': 'cutoff_dim',
    'no_kerr': ('use_kerr', INVERTED),   # --no-kerr sets use_kerr=False
    'epochs': 'epochs',
    'g_lr': 'g_lr',
    'd_lr': 'd_lr',
    'n_critic': 'n_critic',
    'batch_size': 'batch_size',
    'supervised_weight': 'supervised_weight',
    'supervised_warmup': 'supervised_warmup',
    'gp_weight': 'gp_weight',
    'gp_warmup': 'gp_warmup',
    'instance_noise': 'instance_noise',
    'noise_anneal': 'noise_anneal',
    'noise_floor': 'noise_floor',
    'critic_blur': 'critic_blur_sigma',
    'd_dropout': 'd_dropout',
    'latent_scale': 'latent_scale',
    'ket_penalty_weight': 'ket_penalty_weight',
    'g_grad_clip': 'g_grad_clip',
    'grid_size': 'grid_size',
    'plot_every': 'plot_every',
    'val_every': 'val_every',
    'seed': 'seed',
    'deterministic': 'deterministic',
}
# CLI dests with no train_2d_qgan counterpart
TRAIN_EXCLUDED_DESTS = {
    'n_ancilla',   # main() resolves it into n_modes before dispatch
}
# train_2d_qgan params with deliberately no CLI flag
TRAIN_SIG_ONLY = {'x_range', 'log_dir'}

# argparse dest -> CVQGANGenerator.__init__ parameter
GEN_MAP = {
    'n_modes': 'n_modes',                # generator receives total modes
    'n_layers': 'n_layers',
    'cutoff_dim': 'cutoff_dim',
    'no_kerr': ('use_kerr', INVERTED),
    'latent_scale': 'latent_scale',
}
GEN_EXCLUDED_PARAMS = {
    'batch_size',      # None = 'no batched engine', NOT the CLI training
                       # batch size — deliberately excluded from the sync
    'n_output_modes',  # always 2 for 2D output, no CLI flag
    'active_sd',       # init std, no CLI flag
    'passive_sd',      # init std, no CLI flag
    'encoding',        # SEAM 1, no CLI flag
}

# argparse dest -> Discriminator2D.__init__ parameter
DISC_MAP = {
    'd_dropout': 'dropout_rate',
}
DISC_EXCLUDED_PARAMS = {'hidden_dims', 'init_scale'}


def sig_params(fn):
    return {p.name: p for p in inspect.signature(fn).parameters.values()
            if p.name != 'self'}


def check_mapped(owner, params, mapping, parser):
    for dest, target in mapping.items():
        inverted = isinstance(target, tuple)
        pname = target[0] if inverted else target
        assert pname in params, (
            f"{owner}: mapped param {pname!r} (CLI dest {dest!r}) not in "
            f"signature — update the mapping tables")
        p = params[pname]
        if p.default is inspect.Parameter.empty:
            print(f"  ok: {owner}.{pname:20s} required "
                  f"(CLI default for --{dest.replace('_', '-')} is the "
                  f"single source of truth)")
            continue
        cli = parser.get_default(dest)
        want = (not cli) if inverted else cli
        assert p.default == want, (
            f"{owner}.{pname}: signature default {p.default!r} != CLI "
            f"default {want!r} (dest {dest!r}"
            f"{', inverted' if inverted else ''}) — build_parser() is the "
            f"single source of truth")
        print(f"  ok: {owner}.{pname:20s} = {p.default!r} (matches CLI)")


def mapped_names(mapping):
    return {t[0] if isinstance(t, tuple) else t for t in mapping.values()}


def test():
    mod = load_t2q()
    parser = mod.build_parser()
    dests = {a.dest for a in parser._actions if a.dest != 'help'}

    train_params = sig_params(mod.train_2d_qgan)
    gen_params = sig_params(mod.CVQGANGenerator.__init__)
    disc_params = sig_params(mod.Discriminator2D.__init__)

    # (1) every mapped pair agrees
    check_mapped('train_2d_qgan', train_params, TRAIN_MAP, parser)
    check_mapped('CVQGANGenerator', gen_params, GEN_MAP, parser)
    check_mapped('Discriminator2D', disc_params, DISC_MAP, parser)

    # (2) completeness: CLI side (against train_2d_qgan, the dispatch target)
    unaccounted = dests - set(TRAIN_MAP) - TRAIN_EXCLUDED_DESTS
    assert not unaccounted, (
        f"CLI dests neither mapped nor excluded: {sorted(unaccounted)} — "
        f"new flag forgotten in TRAIN_MAP?")

    # (2) completeness: signature side
    stray = set(train_params) - mapped_names(TRAIN_MAP) - TRAIN_SIG_ONLY
    assert not stray, (
        f"train_2d_qgan params neither mapped nor sig-only: {sorted(stray)}")
    stray = set(gen_params) - mapped_names(GEN_MAP) - GEN_EXCLUDED_PARAMS
    assert not stray, (
        f"CVQGANGenerator params neither mapped nor excluded: {sorted(stray)}")
    stray = set(disc_params) - mapped_names(DISC_MAP) - DISC_EXCLUDED_PARAMS
    assert not stray, (
        f"Discriminator2D params neither mapped nor excluded: {sorted(stray)}")
    print(f"  ok: completeness ({len(dests)} CLI dests, "
          f"{len(train_params)}/{len(gen_params)}/{len(disc_params)} params "
          f"accounted for)")


def generate(force=False):
    # Structural test — no golden data to generate; run the checks instead.
    test()


if __name__ == '__main__':
    main_dispatch(generate, test, 'test_default_sync')
