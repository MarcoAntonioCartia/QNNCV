"""Plain-Python checks of scripts/design_filters.py (run by run_tests.py or pytest)."""
import math
import re
import sys
from pathlib import Path

FPGA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FPGA / 'scripts'))
sys.path.insert(0, str(FPGA / 'tb'))
import design_filters as df  # noqa: E402
from dsp_ref import BiquadRef, FirRef, read_signed_mem  # noqa: E402


def test_committed_files_are_current():
    assert read_signed_mem(str(FPGA / 'data' / 'fir_coeffs.mem'), 18) == df.fir_coeffs()
    lut = [int(l.split('//')[0], 16) for l in (FPGA / 'data' / 'fringe_lut.mem').read_text().splitlines()
           if l.split('//')[0].strip()]
    assert lut == df.fringe_lut() and len(lut) == 4096 and min(lut) == 1 and max(lut) == 4095


def test_top_level_defaults_match_the_design():
    src = (FPGA / 'rtl' / 'arty_lock_demo.v').read_text()
    b0 = int(re.search(r"PI_B0\s*=\s*18'sd(\d+)", src).group(1))
    b1 = -int(re.search(r"PI_B1\s*=\s*-18'sd(\d+)", src).group(1))
    assert (b0, b1) == df.pi_coeffs()[:2], f"arty_lock_demo.v has ({b0}, {b1}), design gives {df.pi_coeffs()[:2]}"
    assert int(re.search(r"SAMPLE_DIV\s*=\s*(\d+)", src).group(1)) == round(100e6 / df.FS)
    assert int(re.search(r"PHASE_SHIFT\s*=\s*(\d+)", src).group(1)) == df.PHASE_SHIFT


def closed_loop(n, step=300, tau_samples=None):
    """Linearised loop with the fixed-point PI: detector deviation after a set-point step."""
    b0, b1, _ = df.pi_coeffs()
    pi = BiquadRef((b0, b1, 0, 32768, 0))
    tau = tau_samples or df.TAU * df.FS
    a = math.exp(-1.0 / tau)
    v, out = 0.0, []
    for _ in range(n):
        det = df.fringe_slope() * v
        out.append(det)
        u = pi.step(int(round(step - det)))
        v = a * v + (1 - a) * u
    return out


def test_closed_loop_time_constant():
    """The loop built from the quantised coefficients has the time constant asked for."""
    y = closed_loop(400)
    n63 = next(i for i, v in enumerate(y) if v >= 300 * (1 - math.exp(-1)))
    want = df.TAU_CL * df.FS
    assert abs(n63 - want) <= 0.15 * want, f"63 % reached after {n63} samples, design says {want:.0f}"
    assert max(y) <= 300 * 1.02, f"overshoot: peak {max(y):.1f} for a step of 300"
    assert abs(y[-1] - 300) <= 1.0, f"steady-state error {y[-1] - 300:.2f} codes"


def test_fir_is_matched_and_offset_free():
    h, m = df.fir_coeffs(), df.pulse_shape()
    assert sum(h) == 0
    for a, off in ((1000, 0), (1000, 1500), (-250, -700)):
        f = FirRef(h)
        y = [f.step(int(round(off + a * v))) for v in m][-1]
        assert abs(y - a) <= 2, f"amplitude {a}, offset {off}: {y}"


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f"{len(tests)} design checks passed")
