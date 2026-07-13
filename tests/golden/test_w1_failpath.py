"""
Error-path test: compute_wasserstein_2d's exception fallback.

The bare `except:` was narrowed to `except ValueError` (the only realistic
exception from scipy.stats.wasserstein_distance given the upstream guards),
and the fallback now logs instead of swallowing silently. This test injects
a ValueError via monkeypatching and asserts the inf fallback + warning line,
then confirms the happy path still returns a finite value (the numeric
contract is pinned separately by test_metrics.py). No golden data.
"""

import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import load_t2q, main_dispatch


def make_density(seed):
    import numpy as np
    rs = np.random.RandomState(seed)
    d = rs.rand(40, 40)
    return d / d.sum()


def test():
    import numpy as np
    load_t2q()
    w = sys.modules['src.metrics.wasserstein']

    p = make_density(0)
    q = make_density(1)

    # --- injected failure path ---
    def raiser(*a, **k):
        raise ValueError('injected')

    orig = w.wasserstein_distance
    buf = io.StringIO()
    try:
        w.wasserstein_distance = raiser
        with redirect_stdout(buf):
            result = w.compute_wasserstein_2d(p, q)
    finally:
        w.wasserstein_distance = orig

    out = buf.getvalue()
    assert result == float('inf'), (
        f"fallback value drifted: {result!r} != inf")
    assert 'Warning' in out and 'injected' in out and '(40, 40)' in out, (
        f"failure not logged as expected, got: {out!r}")
    print("  ok: injected ValueError -> inf + logged warning")

    # --- non-ValueError exceptions must propagate (except is narrow now) ---
    def raiser_type(*a, **k):
        raise TypeError('injected-type')

    try:
        w.wasserstein_distance = raiser_type
        try:
            w.compute_wasserstein_2d(p, q)
        except TypeError:
            pass
        else:
            raise AssertionError(
                'TypeError was swallowed — except clause is too broad')
    finally:
        w.wasserstein_distance = orig
    print("  ok: non-ValueError propagates (except is narrow)")

    # --- happy path untainted ---
    val = w.compute_wasserstein_2d(p, q)
    assert np.isfinite(val), f"happy path broke: {val!r}"
    print(f"  ok: happy path finite ({val:.6f})")

    # --- existing degenerate-input guards unchanged (upstream of try) ---
    assert w.compute_wasserstein_2d(np.zeros((40, 40)), p) == float('inf')
    bad = p.copy()
    bad[0, 0] = np.nan
    assert w.compute_wasserstein_2d(bad, q) == float('inf')
    print("  ok: degenerate-input guards unchanged")


def generate(force=False):
    # Structural/error-path test — no golden data to generate.
    test()


if __name__ == '__main__':
    main_dispatch(generate, test, 'test_w1_failpath')
