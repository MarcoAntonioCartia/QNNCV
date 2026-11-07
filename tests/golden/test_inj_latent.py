"""
Structural suite for the 'dedicated' latent variant of interleaved_squeeze
(no golden data):

1. Latent dims: shared = 2*n_modes, dedicated = 3*n_modes; the symbolic
   front params stay 2*n_modes in both; z_index_for_mode maps m -> 2m
   (shared) and 2*n_modes + m (dedicated).
2. Decoupling at the source: a dedicated generator at fixed scale s with its
   dedicated components at ZERO reproduces the input_displacement baseline
   exactly (same seed, same weights, same front components); a nonzero
   dedicated component changes the output.
3. The front components are untouched: with the dedicated components at
   zero, moving a front component moves the dedicated generator's output
   exactly as it moves the baseline's.
4. A z of the wrong width raises a ValueError naming latent_dim (both
   directions: a 2n vector into a dedicated generator, a 3n vector into
   the baseline).
5. EncodingSpec validation, get_config and the CLI surface:
   input_displacement rejects inj_latent='dedicated'; unknown values are
   rejected; --inj-latent defaults to 'shared'; the folder tag is 'zlded'
   and absent at the default.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import load_t2q, seed_all, assert_exact, main_dispatch

MODES = 2
LAYERS = 2
CUTOFF = 6
BATCH = 4
SCALE = 0.3


def build(mod, encoding=None):
    seed_all(0)
    return mod.CVQGANGenerator(n_modes=MODES, n_layers=LAYERS,
                               cutoff_dim=CUTOFF, latent_scale=1.0,
                               batch_size=BATCH, encoding=encoding)


def test():
    import numpy as np
    import tensorflow as tf
    mod = load_t2q()
    EncodingSpec = mod.EncodingSpec
    xvec = np.linspace(-3.0, 3.0, 40)

    g_base = build(mod)
    weights = g_base.weights.numpy()
    rs = np.random.RandomState(7)
    z_front = rs.standard_normal((BATCH, 2 * MODES)).astype(np.float32) * 0.5

    def probs(g, z):
        g.weights.assign(weights)
        p, _ = g.generate_batch(tf.constant(z), xvec, xvec, return_ket_norm=True)
        return p.numpy()

    # --- (1) dims and index mapping ---------------------------------------
    sh = EncodingSpec('interleaved_squeeze', squeeze_scale=SCALE)
    ded = EncodingSpec('interleaved_squeeze', squeeze_scale=SCALE,
                       inj_latent='dedicated')
    assert sh.latent_dim(MODES) == 2 * MODES and ded.latent_dim(MODES) == 3 * MODES
    assert sh.n_front_latents(MODES) == ded.n_front_latents(MODES) == 2 * MODES
    assert sh.z_index_for_mode(1) == 2 and ded.z_index_for_mode(1, MODES) == 2 * MODES + 1
    try:
        ded.z_index_for_mode(1)
        raise AssertionError("dedicated z_index_for_mode without n_modes must raise")
    except ValueError:
        pass
    g_ded = build(mod, encoding=ded)
    assert g_ded.latent_dim == 3 * MODES, g_ded.latent_dim
    assert len(g_ded.latent_params) == 2 * MODES, (
        f"dedicated components must not become symbolic params "
        f"({len(g_ded.latent_params)} symbolic z params)")
    assert len(g_base.latent_params) == 2 * MODES
    print(f"  ok: latent_dim shared={2 * MODES} dedicated={3 * MODES}, "
          f"symbolic front params {2 * MODES} in both, index map m -> 2n+m")

    # --- (2) dedicated components at zero == baseline, nonzero != ---------
    zeros = np.zeros((BATCH, MODES), dtype=np.float32)
    z_ded0 = np.concatenate([z_front, zeros], axis=1)
    p_base = probs(g_base, z_front)
    assert_exact(probs(g_ded, z_ded0), p_base,
                 label='dedicated (components at 0) == baseline')
    z_ded1 = np.concatenate([z_front, np.full((BATCH, MODES), 0.8, np.float32)], axis=1)
    d_on = float(np.max(np.abs(probs(g_ded, z_ded1) - p_base)))
    assert d_on > 1e-6, "a nonzero dedicated component left the output unchanged"
    print(f"  ok: dedicated z=0 reproduces the baseline exactly; "
          f"z=0.8 changes it (max |dP| = {d_on:.2e})")

    # --- (3) front components untouched ------------------------------------
    z_front_moved = z_front.copy()
    z_front_moved[:, 0] += 1.0
    z_ded_moved = np.concatenate([z_front_moved, zeros], axis=1)
    assert_exact(probs(g_ded, z_ded_moved), probs(g_base, z_front_moved),
                 label='front component moves dedicated == baseline')
    print("  ok: front components act identically in the dedicated variant")

    # --- (4) wrong latent width raises ------------------------------------
    for g, z, what in ((g_ded, z_front, '2n vector into dedicated'),
                       (g_base, z_ded0, '3n vector into baseline')):
        try:
            probs(g, z)
            raise AssertionError(f"{what}: no error for a wrong latent width")
        except ValueError as e:
            assert 'latent_dim' in str(e), str(e)
    print("  ok: wrong latent width raises ValueError naming latent_dim")

    # --- (5) validation, get_config, CLI ----------------------------------
    for kw in ({'kind': 'input_displacement', 'inj_latent': 'dedicated'},
               {'kind': 'interleaved_squeeze', 'inj_latent': 'bogus'}):
        try:
            EncodingSpec(**kw)
            raise AssertionError(f"EncodingSpec accepted {kw}")
        except ValueError:
            pass
    assert g_ded.get_config()['inj_latent'] == 'dedicated'
    assert build(mod).get_config()['inj_latent'] == 'shared'
    parser = mod.build_parser()
    assert parser.get_default('inj_latent') == 'shared'
    args = parser.parse_args(['--family', 'ring', '--encoding',
                              'interleaved_squeeze', '--inj-latent', 'dedicated'])
    args.seed = mod.resolve_seed(args.seed)
    parts = mod.build_hp_suffix(args, parser).split('_')
    assert 'encIS' in parts and 'zlded' in parts, parts
    args = parser.parse_args(['--family', 'ring', '--encoding', 'interleaved_squeeze'])
    args.seed = mod.resolve_seed(args.seed)
    parts = mod.build_hp_suffix(args, parser).split('_')
    assert not any(p.startswith('zl') for p in parts), parts
    print("  ok: EncodingSpec validation, get_config, CLI default 'shared', tag zlded")


def generate(force=False):
    # Structural test — no golden data to generate; run the checks instead.
    test()


if __name__ == '__main__':
    main_dispatch(generate, test, 'test_inj_latent')
