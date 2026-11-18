"""cocotb tests of rtl/ad1_spi.v against the ADC model (ad1_model.py)."""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

from ad1_model import Ad1Model, now

CLK_NS = 10
HALF_DIV = int(os.environ.get('TB_HALF_DIV', 4))
GAP_DIV = int(os.environ.get('TB_GAP_DIV', 8))
SEED = int(os.environ.get('TB_SEED', 20261118))


async def setup(dut, codes):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit='ns').start())
    dut.rst.value = 1
    dut.start.value = 0
    it = iter(codes)
    adc = Ad1Model(dut.cs_n, dut.sclk, dut.d0, dut.d1, lambda t: next(it))
    cocotb.start_soon(adc.run())
    await ClockCycles(dut.clk, 5)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    return adc


async def convert(dut):
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    while True:
        await RisingEdge(dut.clk)
        if int(dut.done.value):
            await RisingEdge(dut.clk)
            return int(dut.ch0.value), int(dut.ch1.value)


@cocotb.test()
async def test_codes_bit_exact(dut):
    """Corner and random codes are read back exactly on both channels."""
    rng = random.Random(SEED)
    codes = [(0, 0), (4095, 4095), (0xAAA, 0x555), (0x555, 0xAAA), (1, 0x800), (0x800, 1), (4095, 0), (0, 4095)]
    codes += [(rng.randrange(4096), rng.randrange(4096)) for _ in range(64)]
    adc = await setup(dut, codes)
    assert int(dut.cs_n.value) == 1 and int(dut.sclk.value) == 1 and int(dut.busy.value) == 0, "not idle after reset"
    for i, exp in enumerate(codes):
        got = await convert(dut)
        assert got == exp, f"conversion {i}: read {got}, the ADC sent {exp}"
    assert len(adc.samples) == len(codes), f"{len(adc.samples)} conversions seen by the ADC, expected {len(codes)}"
    dut._log.info(f"seed {SEED}: {len(codes)} conversions, SCLK period {adc.min_period} ns")


@cocotb.test()
async def test_timing(dut):
    """SCLK period, sampling instant and conversion length."""
    adc = await setup(dut, [(0x123, 0xFED)] * 3)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    t_start = now()
    dut.start.value = 0
    while not int(dut.done.value):
        await RisingEdge(dut.clk)
    t_done = now()
    assert adc.min_period == 2 * HALF_DIV * CLK_NS, f"SCLK period {adc.min_period} ns, expected {2 * HALF_DIV * CLK_NS}"
    # the input is sampled (CS falls) at the clock edge that sees start
    assert adc.samples[0]['t'] == t_start, f"CS fell {adc.samples[0]['t'] - t_start} ns after start, expected 0"
    exp = (33 * HALF_DIV + GAP_DIV + 1) * CLK_NS
    assert t_done - t_start == exp, f"start to done {t_done - t_start} ns, expected {exp}"
    assert 1e3 / adc.min_period <= 20.0, "SCLK above 20 MHz"


@cocotb.test()
async def test_back_to_back_and_busy(dut):
    """start held high: conversions follow each other with a legal CS high time; nothing is lost."""
    codes = [(i * 37 % 4096, i * 91 % 4096) for i in range(1, 9)]
    adc = await setup(dut, codes)
    dut.start.value = 1
    got = []
    while len(got) < len(codes):
        await RisingEdge(dut.clk)
        if int(dut.done.value):
            await RisingEdge(dut.clk)
            got.append((int(dut.ch0.value), int(dut.ch1.value)))
            if len(got) == len(codes) - 1:
                dut.start.value = 0
    await ClockCycles(dut.clk, 300)
    assert got == codes, f"read {got}, expected {codes}"
    assert len(adc.samples) == len(codes), f"{len(adc.samples)} conversions, expected {len(codes)}"
