"""
Golden: distribution-family dataset generation.

For each family (gaussian, ring, correlated, four_gaussians, vibronic):
np.random.seed(123) is set immediately before construction+sampling, then
the first 4 sample() densities, their params (as JSON), and the canonical
member are frozen. All numpy / SF-gaussian-backend math -> exact compare.

The per-family re-seed makes each family's golden independent of the
iteration order and of any RNG the family constructor may consume.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (load_t2q, save_golden_npz, load_golden_npz,
                      save_golden_json, load_golden_json,
                      assert_exact, main_dispatch)

GOLDEN = 'families.npz'
GOLDEN_PARAMS = 'families_params.json'

FAMILIES = ['gaussian', 'ring', 'correlated', 'four_gaussians', 'vibronic']
N_SAMPLES = 4


def compute():
    import numpy as np
    mod = load_t2q()
    arrays, params_out = {}, {}
    for name in FAMILIES:
        np.random.seed(123)
        fam = mod.get_family(name)
        dists, params = [], []
        for _ in range(N_SAMPLES):
            d, p = fam.sample()
            dists.append(np.asarray(d))
            params.append(json.dumps(p, sort_keys=True, default=repr))
        canon, canon_p = fam.get_canonical()
        arrays[f'{name}_samples'] = np.stack(dists)
        arrays[f'{name}_canonical'] = np.asarray(canon)
        params_out[name] = {
            'samples': params,
            'canonical': json.dumps(canon_p, sort_keys=True, default=repr),
        }
    return arrays, params_out


def generate(force=False):
    arrays, params = compute()
    save_golden_npz(GOLDEN, arrays, force=force)
    save_golden_json(GOLDEN_PARAMS, params, force=force)


def test():
    golden = load_golden_npz(GOLDEN)
    golden_params = load_golden_json(GOLDEN_PARAMS)
    arrays, params = compute()
    for key in golden.files:
        assert_exact(arrays[key], golden[key], label=key)
        print(f"  ok: {key}")
    assert params == golden_params, "family sample params drifted"
    print("  ok: params")


if __name__ == '__main__':
    main_dispatch(generate, test, 'test_families')
