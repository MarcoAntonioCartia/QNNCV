"""
Coefficients and tables for the lock and readout demo (fpga/rtl/arty_lock_demo.v).

Writes, in fpga/data/:
  fringe_lut.mem   4096 words, 12 bits: 2048 + 2047 sin(2 pi k / 4096), the
                   emulated interferometer fringe seen by the detector
  fir_coeffs.mem   64 words, signed Q1.17: readout filter matched to the pulse
                   shape below, with zero sum (it rejects any constant offset)
and prints the PI coefficients (signed Q3.15) used as parameters of the top level.

Loop model. Plant: first-order lag of time constant TAU (the RC network that
stands for a thermo-optic phase shifter) followed by the fringe, whose slope
at mid-fringe is 2047 * 2 pi / 4096 codes per code (PHASE_SHIFT = 4: the full
DAC range is one fringe). Controller: PI with the zero on the plant pole
(Ti = TAU), discretised with the bilinear transform at FS:

    u[n] = u[n-1] + b0 e[n] + b1 e[n-1]
    b0 = Kp (1 + T / (2 Ti)),   b1 = -Kp (1 - T / (2 Ti))

With the pole cancelled the loop is an integrator of gain Kp * slope / Ti, so
the closed loop is first order with time constant TAU_CL = Ti / (Kp * slope).

    python fpga/scripts/design_filters.py            # write the files, print the numbers
Only the standard library is used.
"""

import math
import os

FS = 100e3            # sample rate, Hz (SAMPLE_DIV = 1000 at 100 MHz)
TAU = 1e-3            # plant time constant, s (R = 10 kOhm, C = 100 nF)
TAU_CL = 0.2e-3       # closed-loop time constant asked for, s
PHASE_SHIFT = 4       # phase (2^16 per turn) = (ADC code - 2048) << PHASE_SHIFT
FRINGE_AMPL = 2047
N_TAPS = 64
PULSE_SIGMA = 9.0     # width of the Gaussian pulse shape, samples
PULSE_CENTRE = 36.0   # its centre inside the 64-sample window (not symmetric on purpose)

C_ONE = 1 << 15       # Q3.15
H_ONE = 1 << 17       # Q1.17


LUT_BITS = 12         # one table entry per ADC code when PHASE_SHIFT = 4


def fringe_lut():
    n = 1 << LUT_BITS
    return [int(round(2048 + FRINGE_AMPL * math.sin(2 * math.pi * k / n))) for k in range(n)]


def fringe_slope():
    """d(detector code) / d(ADC code) at mid-fringe."""
    return FRINGE_AMPL * 2 * math.pi * (1 << PHASE_SHIFT) / 65536.0


def pi_coeffs(fs=FS, tau=TAU, tau_cl=TAU_CL):
    """(b0, b1) as Q3.15 integers, plus the real-valued Kp."""
    kp = tau / (tau_cl * fringe_slope())
    t = 1.0 / fs
    b0 = kp * (1 + t / (2 * tau))
    b1 = -kp * (1 - t / (2 * tau))
    return int(round(b0 * C_ONE)), int(round(b1 * C_ONE)), kp


def pulse_shape(n=N_TAPS):
    """Temporal mode of the emulated detector pulse, m[0] is the oldest sample."""
    return [math.exp(-0.5 * ((k - PULSE_CENTRE) / PULSE_SIGMA) ** 2) for k in range(n)]


def fir_coeffs(n=N_TAPS):
    """Matched filter for amplitude: y = sum_k h[k] x[n-k] with h[k] = g[N-1-k],
    g = (m - mean(m)) / sum((m - mean(m)) * m). For x = a * m + offset the
    output at the end of the window is a. Quantised so that the integers sum
    to exactly zero."""
    m = pulse_shape(n)
    mean = sum(m) / n
    zm = [v - mean for v in m]
    norm = sum(a * b for a, b in zip(zm, m))
    g = [int(round(v / norm * H_ONE)) for v in zm]
    g[g.index(max(g))] -= sum(g)                 # absorb the rounding residue in the largest tap
    assert sum(g) == 0 and all(-H_ONE <= v < H_ONE for v in g)
    return g[::-1]


def write_mem(path, words, bits, header):
    digits = (bits + 3) // 4
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        for line in header:
            fh.write(f"// {line}\n")
        for w in words:
            fh.write(f"{w & ((1 << bits) - 1):0{digits}X}\n")


def main():
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    write_mem(os.path.join(data, 'fringe_lut.mem'), fringe_lut(), 12,
              ["emulated fringe: 2048 + 2047 sin(2 pi k / 4096), written by design_filters.py"])
    h = fir_coeffs()
    write_mem(os.path.join(data, 'fir_coeffs.mem'), h, 18,
              ["readout FIR, signed Q1.17, h[0] first, written by design_filters.py",
               f"matched to a Gaussian pulse (sigma {PULSE_SIGMA}, centre {PULSE_CENTRE} of {N_TAPS}), zero sum"])
    b0, b1, kp = pi_coeffs()
    t = 1.0 / FS
    print(f"fringe slope at lock     {fringe_slope():.4f} codes/code")
    print(f"Kp {kp:.4f}   Ti {TAU * 1e3:.3f} ms   T {t * 1e6:.1f} us")
    print(f"PI_B0 = {b0}   PI_B1 = {b1}   (Q3.15; b0 + b1 = {b0 + b1} -> integral gain per sample {(b0 + b1) / C_ONE:.5f})")
    print(f"closed-loop time constant {TAU_CL * 1e3:.3f} ms = {TAU_CL * FS:.0f} samples")
    print(f"FIR: {len(h)} taps, sum {sum(h)}, max |h| {max(abs(v) for v in h) / H_ONE:.4f}")


if __name__ == '__main__':
    main()
