"""
Signature-freeze golden: public callables must keep their exact parameter
lists and defaults through every extraction phase.

Defaults are unified with the CLI: params whose old signature default
diverged from the CLI default (train_2d_qgan cutoff_dim/d_lr/n_critic/
batch_size/d_dropout/latent_scale, CVQGANGenerator cutoff_dim/latent_scale,
Discriminator2D dropout_rate) are now REQUIRED keyword-only parameters, so
the argparse defaults in build_parser() are the single source of truth.
test_default_sync.py enforces that any remaining shared default agrees
with the CLI.

Allowlist: parameters that a seam phase is permitted to APPEND (must have a
default so all existing call sites are unaffected).
"""

import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (load_t2q, save_golden_json, load_golden_json,
                      main_dispatch)

GOLDEN = 'signatures.json'

TARGETS = [
    ('train_2d_qgan', lambda m: m.train_2d_qgan),
    ('CVQGANGenerator.__init__', lambda m: m.CVQGANGenerator.__init__),
    ('Discriminator2D.__init__', lambda m: m.Discriminator2D.__init__),
    ('validate', lambda m: m.validate),
    ('get_family', lambda m: m.get_family),
    ('build_blur_kernel', lambda m: m.build_blur_kernel),
    ('generate_dataset', lambda m: m.generate_dataset),
    ('compute_hermite_basis', lambda m: m.compute_hermite_basis),
    ('to_critic_input', lambda m: m.to_critic_input),
    ('critic_blur', lambda m: m.critic_blur),
    ('compute_gradient_penalty', lambda m: m.compute_gradient_penalty),
    ('resolve_seed', lambda m: m.resolve_seed),
    ('seed_everything', lambda m: m.seed_everything),
]

# Params a seam phase may APPEND (name -> owner). Anything else extra fails.
APPEND_ALLOWLIST = {
    'CVQGANGenerator.__init__': ['encoding'],   # SEAM 1 (Phase 6)
}


def describe(fn):
    sig = inspect.signature(fn)
    out = []
    for p in sig.parameters.values():
        out.append({
            'name': p.name,
            'kind': str(p.kind),
            'default': (repr(p.default)
                        if p.default is not inspect.Parameter.empty
                        else '<required>'),
        })
    return out


def compute():
    mod = load_t2q()
    return {name: describe(get(mod)) for name, get in TARGETS}


def generate(force=False):
    save_golden_json(GOLDEN, compute(), force=force)


def test():
    golden = load_golden_json(GOLDEN)
    current = compute()
    for name, gold_params in golden.items():
        cur_params = current[name]
        assert len(cur_params) >= len(gold_params), (
            f"{name}: parameters were REMOVED "
            f"({len(cur_params)} < {len(gold_params)})")
        for g, c in zip(gold_params, cur_params):
            assert g == c, (
                f"{name}: param drift at '{g['name']}': golden={g} current={c}")
        extras = cur_params[len(gold_params):]
        allowed = APPEND_ALLOWLIST.get(name, [])
        for e in extras:
            assert e['name'] in allowed, (
                f"{name}: unexpected appended param '{e['name']}'")
            assert e['default'] != '<required>', (
                f"{name}: appended param '{e['name']}' must have a default")
        print(f"  ok: {name} ({len(cur_params)} params)")


if __name__ == '__main__':
    main_dispatch(generate, test, 'test_signatures')
