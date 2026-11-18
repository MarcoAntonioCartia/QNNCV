"""cocotb tests of rtl/da2_spi.v against the DAC model (da2_model.py)."""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge

from da2_model import Da2Model, now

CLK_NS = 10
HALF_DIV = int(os.environ.get('TB_HALF_DIV', 4))
GAP_DIV = int(os.environ.get('TB_GAP_DIV', 8))
SEED = int(os.environ.get('TB_SEED', 20261118))


async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit='ns').start())
    dut.rst.value = 1
    dut.start.value = 0
    dut.code_a.value = 0
    dut.code_b.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst.value = 0
    dac = Da2Model(dut.sync_n, dut.sclk, dut.dina, dut.dinb, dut._log)
    cocotb.start_soon(dac.run())
    await RisingEdge(dut.clk)
    return dac


async def write(dut, a, b):
    """One-cycle start pulse, then wait for done."""
    dut.code_a.value = a
    dut.code_b.value = b
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    dut.code_a.value = 0        # the driver must have latched the codes
    dut.code_b.value = 0
    while True:
        await RisingEdge(dut.clk)
        if int(dut.done.value):
            return


@cocotb.test()
async def test_idle_after_reset(dut):
    """Reset state: SYNC high, SCLK high, not busy, no frame sent."""
    dac = await setup(dut)
    await ClockCycles(dut.clk, 50)
    assert int(dut.sync_n.value) == 1, "SYNC must idle high"
    assert int(dut.sclk.value) == 1, "SCLK must idle high"
    assert int(dut.busy.value) == 0, "busy after reset"
    assert dac.frames == [], f"frames sent without start: {dac.frames}"


@cocotb.test()
async def test_frames_bit_exact(dut):
    """Corner codes and random codes arrive bit for bit on both channels."""
    dac = await setup(dut)
    rng = random.Random(SEED)
    dut._log.info(f"seed {SEED}")
    vectors = [(0x000, 0x000), (0xFFF, 0xFFF), (0xAAA, 0x555), (0x555, 0xAAA),
               (0x001, 0x800), (0x800, 0x001), (0xFFF, 0x000), (0x000, 0xFFF)]
    vectors += [(rng.randrange(4096), rng.randrange(4096)) for _ in range(64)]
    for i, (a, b) in enumerate(vectors):
        await write(dut, a, b)
        assert len(dac.frames) == i + 1, f"vector {i}: {len(dac.frames)} frames seen, expected {i + 1}"
        f = dac.frames[-1]
        assert f['word_a'] == a, f"vector {i} channel A: sent {a:#06x}, DAC got {f['word_a']:#06x}"
        assert f['word_b'] == b, f"vector {i} channel B: sent {b:#06x}, DAC got {f['word_b']:#06x}"
    dut._log.info(f"{len(vectors)} frames; min half period {dac.min_half} ns, "
                  f"min setup {dac.min_setup} ns, min hold {dac.min_hold} ns")


@cocotb.test()
async def test_timing_exact(dut):
    """Edge positions of one frame, in clock periods."""
    dac = await setup(dut)
    half = HALF_DIV * CLK_NS
    await write(dut, 0xA5A, 0x5A5)
    t_done = now()
    f = dac.frames[-1]
    assert dac.min_half == half, f"SCLK half period {dac.min_half} ns, expected {half}"
    assert dac.min_setup == half, f"data setup {dac.min_setup} ns, expected {half}"
    assert dac.min_hold == half, f"data hold {dac.min_hold} ns, expected {half}"
    # the DAC updates on the 16th falling edge: lead + 15 full periods after SYNC falls
    exp = half + 15 * 2 * half
    assert f['t_update'] - f['t_sync'] == exp, f"16th falling edge {f['t_update'] - f['t_sync']} ns after SYNC, expected {exp}"
    # frame length seen by the user logic: SYNC fall to done
    exp = (33 * HALF_DIV + GAP_DIV + 1) * CLK_NS
    assert t_done - f['t_sync'] == exp, f"SYNC fall to done {t_done - f['t_sync']} ns, expected {exp}"
    assert 1e3 / (2 * half) <= 30.0, f"SCLK {1e3 / (2 * half)} MHz above the 30 MHz limit"


@cocotb.test()
async def test_back_to_back(dut):
    """start held high: frames follow each other and SYNC high time stays legal."""
    dac = await setup(dut)
    codes = [(0x123, 0xFED), (0x456, 0xCBA), (0x789, 0x987), (0xABC, 0x654)]
    dut.start.value = 1
    dut.code_a.value, dut.code_b.value = codes[0]
    for nxt in codes[1:] + [None]:
        await FallingEdge(dut.sync_n)        # frame under way: present the next codes
        if nxt is None:
            dut.start.value = 0
        else:
            dut.code_a.value, dut.code_b.value = nxt
    await ClockCycles(dut.clk, 200)
    got = [(f['code_a'], f['code_b']) for f in dac.frames]
    assert got == codes, f"back-to-back frames {got}, expected {codes}"
    assert dac.min_sync_high >= 20.0, f"SYNC high {dac.min_sync_high} ns between frames (min 20)"
    dut._log.info(f"SYNC high between frames: {dac.min_sync_high} ns")


@cocotb.test()
async def test_start_ignored_while_busy(dut):
    """A start pulse in the middle of a frame must not disturb it."""
    dac = await setup(dut)
    dut.code_a.value, dut.code_b.value = 0x3C3, 0xC3C
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await ClockCycles(dut.clk, 10 * HALF_DIV)
    assert int(dut.busy.value) == 1, "busy must be high during a frame"
    dut.code_a.value, dut.code_b.value = 0xFFF, 0xFFF
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await ClockCycles(dut.clk, 80 * HALF_DIV)
    got = [(f['code_a'], f['code_b']) for f in dac.frames]
    assert got == [(0x3C3, 0xC3C)], f"frames {got}, expected only the first write"
    assert int(dut.busy.value) == 0, "busy stuck high"


@cocotb.test()
async def test_reset_mid_frame(dut):
    """Reset during a frame: lines go idle; the next write is a clean frame.

    The model rejects the truncated frame, so it is not attached here until
    after the reset (a real DAC121S101 discards a frame cut short by SYNC)."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit='ns').start())
    dut.rst.value = 1
    dut.start.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst.value = 0
    dut.code_a.value, dut.code_b.value = 0x111, 0x222
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await ClockCycles(dut.clk, 12 * HALF_DIV)
    assert int(dut.sync_n.value) == 0, "expected to be inside a frame"
    dut.rst.value = 1
    await ClockCycles(dut.clk, 3)
    assert int(dut.sync_n.value) == 1 and int(dut.sclk.value) == 1, "lines not idle in reset"
    dut.rst.value = 0
    dac = Da2Model(dut.sync_n, dut.sclk, dut.dina, dut.dinb)
    cocotb.start_soon(dac.run())
    await RisingEdge(dut.clk)
    await write(dut, 0x777, 0x888)
    got = [(f['code_a'], f['code_b']) for f in dac.frames]
    assert got == [(0x777, 0x888)], f"after reset got {got}"
