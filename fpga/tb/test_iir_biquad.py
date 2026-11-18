"""cocotb tests of rtl/iir_biquad.v against the bit-exact reference (dsp_ref.py).
Built twice by run_tests.py: a general second-order section and the PI of the demo."""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

from dsp_ref import BiquadRef

COEF = tuple(int(os.environ[f'TB_{k}']) for k in ('B0', 'B1', 'B2', 'NA1', 'NA2'))
Y_MAX, Y_MIN = int(os.environ['TB_Y_MAX']), int(os.environ['TB_Y_MIN'])
SEED = int(os.environ.get('TB_SEED', 20261118))


def s13(v):
    v = int(v)
    return v - 8192 if v & 4096 else v


async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit='ns').start())
    dut.rst.value = 1
    dut.clear.value = 0
    dut.start.value = 0
    dut.x.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def step(dut, x):
    dut.x.value = x
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    dut.x.value = 0                      # the input must have been latched
    n = 0
    while True:
        await RisingEdge(dut.clk)
        n += 1
        if int(dut.done.value):
            await RisingEdge(dut.clk)
            return s13(dut.y.value), int(dut.sat.value), n


async def compare(dut, xs, what):
    ref = BiquadRef(COEF, Y_MAX, Y_MIN)
    nsat = 0
    for i, x in enumerate(xs):
        y, sat, n = await step(dut, x)
        e = ref.step(x)
        assert y == e, f"{what}, sample {i} (x = {x}): y = {y}, reference {e}"
        assert sat == ref.sat, f"{what}, sample {i}: sat = {sat}, reference {ref.sat}"
        assert n == 9, f"{what}: result {n} clocks after start, expected 9"
        nsat += sat
    return nsat


@cocotb.test()
async def test_random_bit_exact(dut):
    """Small, large and saturating random inputs match the reference bit for bit."""
    await setup(dut)
    rng = random.Random(SEED)
    xs = [rng.randint(-40, 40) for _ in range(300)]
    xs += [rng.randint(-4096, 4095) for _ in range(600)]
    xs += [4095] * 60 + [-4096] * 60 + [0] * 100
    nsat = await compare(dut, xs, "random")
    dut._log.info(f"seed {SEED}: {len(xs)} samples, {nsat} clamped")
    assert nsat > 0, "the stimulus never reached the clamp; the saturation path is untested"


@cocotb.test()
async def test_step_and_impulse(dut):
    """Impulse and step responses match the reference."""
    await setup(dut)
    await compare(dut, [1000] + [0] * 150, "impulse")
    dut.clear.value = 1
    await ClockCycles(dut.clk, 2)
    dut.clear.value = 0
    await RisingEdge(dut.clk)
    await compare(dut, [50] * 200, "step")


@cocotb.test()
async def test_clear(dut):
    """clear zeroes the state: the output reads 0 and the next run starts from rest."""
    await setup(dut)
    await compare(dut, [700] * 30, "before clear")
    dut.clear.value = 1
    await ClockCycles(dut.clk, 3)
    assert s13(dut.y.value) == 0, f"y = {s13(dut.y.value)} while clear is high"
    dut.clear.value = 0
    await RisingEdge(dut.clk)
    await compare(dut, [-300] * 30, "after clear")


@cocotb.test(skip=not (COEF[3] == 32768 and COEF[2] == 0 and COEF[4] == 0))
async def test_pi_antiwindup(dut):
    """PI only. A constant error drives the output to the rail and holds it there four
    times longer than it took to get there. When the error reverses, the output must leave
    the rail on the very next sample: the stored value is the clamped one. An integrator
    that had kept counting would stay at the rail for hundreds of samples more."""
    await setup(dut)
    ys = [(await step(dut, 200))[0] for _ in range(2500)]
    first = ys.index(Y_MAX)
    assert first < 700 and ys[-1] == Y_MAX and int(dut.sat.value) == 1, \
        f"expected the rail {Y_MAX} within 700 samples, first reached at {first}, last y = {ys[-1]}"
    y, sat, _ = await step(dut, -200)
    kick = round((COEF[0] - COEF[1]) * 200 / 32768)          # proportional step for an error change of 400
    assert Y_MAX - kick - 3 <= y <= Y_MAX - kick + 3 and sat == 0, \
        f"after the error reversed y = {y} (sat {sat}), expected about {Y_MAX - kick}"
