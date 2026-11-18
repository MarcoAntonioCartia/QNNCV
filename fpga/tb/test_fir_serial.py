"""cocotb tests of rtl/fir_serial.v against the bit-exact reference (dsp_ref.py)."""

import os
import random
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import design_filters  # noqa: E402
from dsp_ref import FirRef, read_signed_mem  # noqa: E402

H = read_signed_mem(os.environ['TB_FIR'], 18)
N = len(H)
SEED = int(os.environ.get('TB_SEED', 20261118))


def s13(v):
    v = int(v)
    return v - 8192 if v & 4096 else v


async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit='ns').start())
    dut.rst.value = 1
    dut.start.value = 0
    dut.x.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    for _ in range(N):                     # the sample memory is not cleared by reset
        await step(dut, 0)


async def step(dut, x):
    dut.x.value = x
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    dut.x.value = 0
    n = 0
    while True:
        await RisingEdge(dut.clk)
        n += 1
        if int(dut.done.value):
            await RisingEdge(dut.clk)
            return s13(dut.y.value), n


@cocotb.test()
async def test_coefficients_loaded(dut):
    """The coefficient memory holds the file, and the file is what design_filters.py produces."""
    await setup(dut)
    assert H == design_filters.fir_coeffs(), "fir_coeffs.mem is not the current output of design_filters.py"
    assert sum(H) == 0, f"coefficients sum to {sum(H)}, expected 0"
    for k in range(N):
        v = int(dut.coef[k].value)
        v = v - (1 << 18) if v >> 17 else v
        assert v == H[k], f"coef[{k}] = {v}, file has {H[k]}"


@cocotb.test()
async def test_random_bit_exact(dut):
    """Random samples over the full range match the reference bit for bit."""
    await setup(dut)
    rng = random.Random(SEED)
    ref = FirRef(H)
    xs = [rng.randint(-4096, 4095) for _ in range(400)] + [rng.randint(-30, 30) for _ in range(150)]
    for i, x in enumerate(xs):
        y, n = await step(dut, x)
        e = ref.step(x)
        assert y == e, f"sample {i}: y = {y}, reference {e}"
        assert n == N + 4, f"result {n} clocks after start, expected {N + 4}"
    dut._log.info(f"seed {SEED}: {len(xs)} samples")


@cocotb.test()
async def test_offset_rejected(dut):
    """A constant input gives exactly 0 once the window is full (the taps sum to zero)."""
    await setup(dut)
    for level in (4095, -4096, 1234):
        ys = [(await step(dut, level))[0] for _ in range(N + 20)]
        assert set(ys[N:]) == {0}, f"constant {level}: outputs {sorted(set(ys[N:]))}, expected only 0"


@cocotb.test()
async def test_pulse_amplitude(dut):
    """offset + a * pulse shape: the output at the end of the window is a (within 2 codes)."""
    await setup(dut)
    m = design_filters.pulse_shape(N)
    for a, off in ((1500, 0), (-900, 700), (37, -2000), (0, 300)):
        for _ in range(N):
            await step(dut, off)
        y = None
        for k in range(N):
            y, _ = await step(dut, int(round(off + a * m[k])))
        assert abs(y - a) <= 2, f"amplitude {a} on offset {off}: filter gives {y}"
