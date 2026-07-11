"""
Golden: Hermite-function position basis and cutoff recommendation.

Pure numpy/scipy computations -> exact comparison.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import (load_t2q, save_golden_npz, load_golden_npz,
                      assert_exact, main_dispatch)

GOLDEN = 'hermite.npz'

CUTOFFS = [4, 6, 10, 12]
REC_CASES = [(2, 8), (3, 8), (4, 10), (6, 8)]  # (n_modes, base_cutoff)


def compute():
    import numpy as np
    mod = load_t2q()
    xvec = np.linspace(-3.0, 3.0, 40)
    out = {'xvec': xvec}
    for c in CUTOFFS:
        out[f'basis_c{c}'] = mod.compute_hermite_basis(xvec, c).numpy()
    out['rec_cases'] = np.array(REC_CASES, dtype=np.int64)
    out['rec_results'] = np.array(
        [mod.recommend_cutoff(m, base) for m, base in REC_CASES],
        dtype=np.int64)
    return out


def generate(force=False):
    save_golden_npz(GOLDEN, compute(), force=force)


def test():
    golden = load_golden_npz(GOLDEN)
    current = compute()
    for key in golden.files:
        assert_exact(current[key], golden[key], label=key)
        print(f"  ok: {key}")


if __name__ == '__main__':
    main_dispatch(generate, test, 'test_hermite')
