"""cocotb tests of the top level (ROM + sequencer + DAC driver) against a
Python reference model of the step timing. Times are compared exactly, in
clock periods; the DAC model (da2_model.py) checks every frame on the wire."""

import os
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer, ValueChange

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import phases_to_mem  # noqa: E402
from da2_model import Da2Model, now  # noqa: E402

CLK_NS = 10
P = {k: int(os.environ[f'TB_{k}']) for k in
     ('TICK_DIV', 'RAMP_LOG2', 'SETTLE_TICKS', 'GATE_TICKS', 'HOLD_TICKS', 'SPI_HALF_DIV', 'SPI_GAP_DIV')}
TICK = P['TICK_DIV']
RAMP = 1 << P['RAMP_LOG2']
T_GATE_ON = (RAMP + P['SETTLE_TICKS']) * TICK              # clock periods after the step starts
T_GATE_OFF = T_GATE_ON + P['GATE_TICKS'] * TICK
T_STEP = T_GATE_OFF + P['HOLD_TICKS'] * TICK
TABLE = [phases_to_mem.unpack(w) for w in phases_to_mem.table_of(phases_to_mem.read_mem(os.environ['TB_MEM']))]


def reference(table, layer, n_steps):
    """Expected behaviour of the first n_steps played steps after reset.
    layer None = play all. Returns a list of dicts with times in clock
    periods from the start of the first step."""
    out, t, a, b, addr, guard = [], 0, 0, 0, 0, 0
    while len(out) < n_steps:
        s = table[addr]
        nxt = 0 if s['last'] else addr + 1
        if layer is not None and s['layer'] != layer:
            t += 2 if out else 0                             # skipped: fetch + decode
            addr = nxt
            guard += 1
            assert guard <= len(table), "reference: no step of that layer"
            continue
        guard = 0
        if s['ramp']:
            writes = [(t + i * TICK + 1, a + ((i * (s['code_a'] - a)) >> P['RAMP_LOG2']),
                       b + ((i * (s['code_b'] - b)) >> P['RAMP_LOG2'])) for i in range(1, RAMP + 1)]
        else:
            writes = [(t + 1, s['code_a'], s['code_b'])]
        a, b = s['code_a'], s['code_b']
        assert writes[-1][1:] == (a, b)
        out.append({'addr': addr, 't0': t, 'writes': writes, 'layer': s['layer'], 'mark': s['mark'],
                    'gate': (t + T_GATE_ON, t + T_GATE_OFF) if s['gate'] else None,
                    'code_a': a, 'code_b': b})
        t += T_STEP + 2                                      # + fetch + decode of the next word
        addr = nxt
    return out


class Recorder:
    """Time-stamped record of what the design does."""

    def __init__(self, dut):
        self.dut = dut
        self.dac = Da2Model(dut.da2_sync_n, dut.da2_sclk, dut.da2_dina, dut.da2_dinb)
        self.steps = []          # (t, led, trig) at each step start
        self.gate = []           # (t, level, sync_n, dac code a, dac code b)
        self.trig = []           # (t, level)
        cocotb.start_soon(self.dac.run())
        cocotb.start_soon(self._steps())
        cocotb.start_soon(self._edges(dut.gate, self.gate, True))
        cocotb.start_soon(self._edges(dut.trig, self.trig, False))

    async def _steps(self):
        while True:
            await RisingEdge(self.dut.u_seq.step_start)
            self.steps.append((now(), int(self.dut.led.value), int(self.dut.trig.value)))

    async def _edges(self, sig, store, with_dac):
        while True:
            await ValueChange(sig)
            extra = (int(self.dut.da2_sync_n.value), self.dac.code_a, self.dac.code_b) if with_dac else ()
            store.append((now(), int(sig.value)) + extra)


async def start(dut, sw, clock=True):
    if clock:
        cocotb.start_soon(Clock(dut.clk100, CLK_NS, unit='ns').start())
    dut.btn0.value = 1
    dut.sw.value = sw
    await ClockCycles(dut.clk100, 8)
    rec = Recorder(dut)
    dut.btn0.value = 0
    return rec


def cyc(t_ns, t0_ns):
    c = (t_ns - t0_ns) / CLK_NS
    assert abs(c - round(c)) < 1e-6, f"event off the clock grid: {t_ns} ns"
    return round(c)


def check(rec, ref, what):
    """Compare a recording with the reference, step by step."""
    n = len(ref)
    assert len(rec.steps) >= n, f"{what}: {len(rec.steps)} steps played, expected at least {n}"
    t0 = rec.steps[0][0]
    t_end = ref[-1]['t0'] + T_STEP                            # only complete steps are compared
    frames = [f for f in rec.dac.frames if cyc(f['t_sync'], t0) < t_end]
    gates = [g for g in rec.gate if cyc(g[0], t0) <= t_end]
    exp_frames = [w for s in ref for w in s['writes']]
    got = [(cyc(f['t_sync'], t0), f['code_a'], f['code_b']) for f in frames]
    assert len(got) == len(exp_frames), f"{what}: {len(got)} DAC frames, expected {len(exp_frames)}"
    for i, (g, e) in enumerate(zip(got, exp_frames)):
        assert g == e, f"{what}: DAC frame {i} (clk, A, B) = {g}, expected {e}"
    exp_gate = [(t, lvl) for s in ref if s['gate'] for t, lvl in ((s['gate'][0], 1), (s['gate'][1], 0))]
    got_gate = [(cyc(g[0], t0), g[1]) for g in gates]
    assert got_gate == exp_gate, f"{what}: gate edges (clk, level) {got_gate}, expected {exp_gate}"
    k = 0
    for s in ref:
        t, led, trig = rec.steps[ref.index(s)]
        assert cyc(t, t0) == s['t0'], f"{what}: step at address {s['addr']} starts at clk {cyc(t, t0)}, expected {s['t0']}"
        assert led == s['layer'], f"{what}: led = {led} at address {s['addr']}, expected layer {s['layer']}"
        assert trig == s['mark'], f"{what}: trig = {trig} at address {s['addr']}, expected {s['mark']}"
        if s['gate']:
            # at the rising edge of the gate the DAC is idle and already holds the target codes
            _, lvl, sync_n, ca, cb = gates[2 * k]
            assert lvl == 1 and sync_n == 1, f"{what}: DAC frame in progress when the gate opens (address {s['addr']})"
            assert (ca, cb) == (s['code_a'], s['code_b']), \
                f"{what}: DAC holds ({ca}, {cb}) when the gate opens at address {s['addr']}, expected ({s['code_a']}, {s['code_b']})"
            _, lvl, sync_n, ca, cb = gates[2 * k + 1]
            assert sync_n == 1 and (ca, cb) == (s['code_a'], s['code_b']), \
                f"{what}: DAC disturbed while the gate was open (address {s['addr']})"
            k += 1
    # no DAC frame may overlap an open gate
    for s in ref:
        if s['gate']:
            for f in frames:
                a, b = cyc(f['t_sync'], t0), cyc(f['t_update'], t0)
                assert b < s['gate'][0] or a > s['gate'][1], f"{what}: DAC frame at clk {a} overlaps the gate"


def check_trig(rec, ref, what):
    t0 = rec.steps[0][0]
    t_end = ref[-1]['t0'] + T_STEP
    exp = [(t, lvl) for s in ref if s['mark'] for t, lvl in ((s['t0'], 1), (s['t0'] + T_STEP, 0))]
    got = [(cyc(t, t0), lvl) for t, lvl in rec.trig if cyc(t, t0) <= t_end]
    assert got == exp, f"{what}: trigger edges (clk, level) {got}, expected {exp}"


async def run_steps(dut, rec, n):
    """Run until n played steps are complete, plus a small margin."""
    ref_len = n * (T_STEP + 2) + 2 * len(TABLE) * (n + 2) + 50
    await ClockCycles(dut.clk100, ref_len)


@cocotb.test()
async def test_table_file(dut):
    """The ROM holds exactly the words of the .mem file; the rest reads as empty 'last' steps."""
    await Timer(1, unit='ns')
    words = phases_to_mem.read_mem(os.environ['TB_MEM'])
    depth = len(dut.u_rom.mem)
    assert len(words) == depth, f"{len(words)} words in the file for a ROM of {depth}"
    for i in range(depth):
        exp = words[i] if i < len(words) else 0x80000000
        got = int(dut.u_rom.mem[i].value)
        assert got == exp, f"ROM[{i}] = {got:08X}, expected {exp:08X}"


@cocotb.test()
async def test_play_all(dut):
    """Whole table twice (wrap included): codes, ramp points, gate, trigger, exact timing."""
    n = 2 * len(TABLE) + 2
    rec = await start(dut, 0xF)
    await run_steps(dut, rec, n)
    ref = reference(TABLE, None, n)
    check(rec, ref, "play all")
    check_trig(rec, ref, "play all")
    dut._log.info(f"{n} steps, {sum(len(s['writes']) for s in ref)} DAC frames, step = {T_STEP} clk; "
                  f"wire timing: half period {rec.dac.min_half:.1f} ns, setup {rec.dac.min_setup:.1f} ns, "
                  f"hold {rec.dac.min_hold:.1f} ns, SYNC high {rec.dac.min_sync_high:.1f} ns")


@cocotb.test()
async def test_ramp_is_linear(dut):
    """Ramp steps: RAMP writes one tick apart, monotonic, never overshooting, ending on the target."""
    rec = await start(dut, 0xF)
    await run_steps(dut, rec, len(TABLE))
    ref = reference(TABLE, None, len(TABLE))
    t0 = rec.steps[0][0]
    ramps = [s for s in ref if len(s['writes']) > 1]
    assert ramps, "the test table has no ramp step"
    for s in ramps:
        fr = [f for f in rec.dac.frames if s['t0'] <= cyc(f['t_sync'], t0) < s['t0'] + T_STEP]
        assert len(fr) == RAMP, f"address {s['addr']}: {len(fr)} ramp writes, expected {RAMP}"
        dts = {cyc(b['t_sync'], a['t_sync']) for a, b in zip(fr, fr[1:])}
        assert dts == {TICK}, f"address {s['addr']}: ramp writes spaced {sorted(dts)} clk, expected {TICK}"
        for ch in ('code_a', 'code_b'):
            v = [f[ch] for f in fr]
            d = [y - x for x, y in zip(v, v[1:])]
            assert all(x >= 0 for x in d) or all(x <= 0 for x in d), f"address {s['addr']} {ch}: not monotonic {v}"
            assert v[-1] == s[ch], f"address {s['addr']} {ch}: ramp ends at {v[-1]}, expected {s[ch]}"
            assert max(d) - min(d) <= 1, f"address {s['addr']} {ch}: uneven ramp increments {d}"


@cocotb.test()
async def test_single_layer(dut):
    """Switches select one layer: only its steps are played, in a loop, with one trigger per loop."""
    layers = sorted({s['layer'] for s in TABLE})
    for i, layer in enumerate(layers):
        per_loop = sum(1 for s in TABLE if s['layer'] == layer)
        n = 2 * per_loop + 1
        rec = await start(dut, layer, clock=(i == 0))
        await run_steps(dut, rec, n)
        ref = reference(TABLE, layer, n)
        assert {s['layer'] for s in ref} == {layer}
        check(rec, ref, f"layer {layer}")
        check_trig(rec, ref, f"layer {layer}")
        await RisingEdge(dut.gate)                           # reset while no DAC frame is on the wire


@cocotb.test()
async def test_missing_layer(dut):
    """A layer that is not in the table: nothing is played and the outputs stay idle."""
    missing = next(l for l in range(15) if l not in {s['layer'] for s in TABLE})
    rec = await start(dut, missing)
    await ClockCycles(dut.clk100, 3 * T_STEP)
    assert rec.steps == [] and rec.dac.frames == [] and rec.gate == [] and rec.trig == [], \
        f"activity with a missing layer: {len(rec.steps)} steps, {len(rec.dac.frames)} frames"
    assert int(dut.gate.value) == 0 and int(dut.da2_sync_n.value) == 1 and int(dut.da2_sclk.value) == 1


@cocotb.test()
async def test_layer_change_on_the_fly(dut):
    """Moving the switches while running takes effect at the next step; no step is cut short."""
    rec = await start(dut, 0xF)
    await ClockCycles(dut.clk100, T_STEP + T_STEP // 2)
    target = TABLE[-1]['layer']
    dut.sw.value = target
    n_before = len(rec.steps)
    await ClockCycles(dut.clk100, 4 * T_STEP)
    assert n_before == 2, f"expected to be inside the second step, saw {n_before} step starts"
    later = [led for _, led, _ in rec.steps[n_before:]]
    assert later and set(later) == {target}, f"layers played after the change: {later}, expected only {target}"
    widths = [cyc(b[0], a[0]) for a, b in zip(rec.gate[0::2], rec.gate[1::2])]
    assert set(widths) == {P['GATE_TICKS'] * TICK}, f"gate widths {widths} clk, expected {P['GATE_TICKS'] * TICK}"


@cocotb.test()
async def test_reset_while_gate_open(dut):
    """Reset in the middle of a gate pulse: gate and trigger drop, the table restarts at step 0."""
    rec = await start(dut, 0xF)
    await RisingEdge(dut.gate)
    await ClockCycles(dut.clk100, TICK)
    dut.btn0.value = 1
    await ClockCycles(dut.clk100, 4)                         # two synchronizer stages + one
    assert int(dut.gate.value) == 0, "gate still high in reset"
    assert int(dut.trig.value) == 0, "trigger still high in reset"
    await ClockCycles(dut.clk100, 300)                       # let the model see any frame end
    rec2 = Recorder(dut)
    dut.btn0.value = 0
    n = len(TABLE)
    await run_steps(dut, rec2, n)
    ref = reference(TABLE, None, n)
    check(rec2, ref, "after reset")
