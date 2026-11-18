"""cocotb tests of the lock and readout top level, with models of both Pmods and of
the RC network between DAC A and ADC channel 0 (the only analogue part of the loop)."""

import math
import os
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import design_filters  # noqa: E402
from ad1_model import Ad1Model  # noqa: E402
from da2_model import Da2Model  # noqa: E402
from dsp_ref import BiquadRef, FirRef, read_signed_mem  # noqa: E402

CLK_NS = 10
P = {k: int(os.environ[f'TB_{k}']) for k in
     ('SAMPLE_DIV', 'GATE_PERIOD', 'GATE_LEN', 'STEP_HALF', 'STEP_AMPL', 'DRIFT_STEP', 'WARM_SAMPLES', 'PI_B0', 'PI_B1')}
T_S = P['SAMPLE_DIV'] * CLK_NS                       # sample period, ns
TAU_SAMPLES = design_filters.TAU * design_filters.FS      # plant time constant: 100 samples, as on the board
TAU_NS = TAU_SAMPLES * T_S
TAU_CL = design_filters.TAU_CL * design_filters.FS        # closed loop: 20 samples
H = read_signed_mem(os.environ['TB_FIR'], 18)
N = len(H)
PULSE = design_filters.pulse_shape(N)
SW_LOOP, SW_DRIFT, SW_SHOT, SW_STEP = 1, 2, 4, 8


class Bench:
    """Both Pmods plus the RC network. ch1_fn(n) gives the code of ADC channel 1 for
    the n-th sample inside the gate period (None outside the gate window)."""

    def __init__(self, dut, tau_ns=TAU_NS, ch1_fn=None, ch0_stuck=None):
        self.dut = dut
        self.tau_ns, self.ch1_fn, self.ch0_stuck = tau_ns, ch1_fn, ch0_stuck
        self.v, self.t_v, self.applied = 0.0, 0.0, 0      # capacitor voltage in DAC codes
        self.target = 0.0
        self.dac = Da2Model(dut.da2_sync_n, dut.da2_sclk, dut.da2_dina, dut.da2_dinb)
        self.adc = Ad1Model(dut.ad1_cs_n, dut.ad1_sclk, dut.ad1_d0, dut.ad1_d1, self.sample)
        self.log = []                                     # per sample: n_gate, gate, codes, internals
        cocotb.start_soon(self.dac.run())
        cocotb.start_soon(self.adc.run())
        cocotb.start_soon(self._record())

    def _advance(self, t):
        self.v += (self.target - self.v) * (1 - math.exp(-(t - self.t_v) / self.tau_ns))
        self.t_v = t

    def sample(self, t):
        # apply the DAC updates that happened since the last sample, in order
        while self.applied < len(self.dac.frames):
            f = self.dac.frames[self.applied]
            self._advance(f['t_update'])
            self.target = float(f['code_a'])
            self.applied += 1
        self._advance(t)
        c0 = self.ch0_stuck if self.ch0_stuck is not None else min(max(int(round(self.v)), 0), 4095)
        n = int(self.dut.n_gate.value)
        c1 = self.ch1_fn(n) if self.ch1_fn else 2048
        return c0, min(max(int(c1), 0), 4095)

    async def _record(self):
        """One entry per sample, taken when the DAC write of that sample is issued."""
        d = self.dut
        while True:
            await RisingEdge(d.dac_start)
            s = self.adc.samples[-1]
            self.log.append({'n': int(d.n_gate.value), 'gate': int(d.gate.value), 'adc0': s['code0'],
                             'adc1': s['code1'], 'det': int(d.det.value), 'err': s13(d.err.value),
                             'ctrl': s13(d.ctrl.value), 'dac_a': int(d.dac_a.value), 'dac_b': int(d.dac_b.value),
                             'shot': s13(d.shot.value), 'sq': int(d.sq.value), 'led': int(d.led.value),
                             'drift': s16(d.drift.value), 'run': int(d.loop_run.value)})


def s13(v):
    v = int(v)
    return v - 8192 if v & 4096 else v


def s16(v):
    v = int(v)
    return v - 65536 if v & 32768 else v


async def start(dut, sw, clock=True, **kw):
    if clock:
        cocotb.start_soon(Clock(dut.clk100, CLK_NS, unit='ns').start())
    dut.btn0.value = 1
    dut.sw.value = sw
    dut.ad1_d0.value = 0
    dut.ad1_d1.value = 0
    await ClockCycles(dut.clk100, 8)
    b = Bench(dut, **kw)
    dut.btn0.value = 0
    return b


async def samples(dut, n, warm=True):
    """n samples, counted after the start-up wait unless warm is False."""
    await ClockCycles(dut.clk100, (n + (P['WARM_SAMPLES'] if warm else 0)) * P['SAMPLE_DIV'])


def running(log):
    """Samples taken once the controller is allowed to run."""
    return log[P['WARM_SAMPLES']:]


@cocotb.test()
async def test_open_loop_idle(dut):
    """Loop open, no drift: DAC A parked at mid-scale, detector at mid-fringe once the RC has charged."""
    b = await start(dut, 0)
    await samples(dut, 700, warm=False)
    assert {e['dac_a'] for e in b.log} == {2048}, f"DAC A codes with the loop open: {sorted({e['dac_a'] for e in b.log})}"
    last = b.log[-1]
    assert abs(last['adc0'] - 2048) <= 4, f"RC output {last['adc0']} after 7 time constants, expected 2048"
    assert abs(last['det'] - 2048) <= 16, f"detector {last['det']}, expected mid-fringe 2048"
    assert last['led'] & 0b1011 == 0, f"leds {last['led']:04b}: loop, drift and rail must be off"
    assert len(b.log) == len(b.dac.frames) or len(b.log) == len(b.dac.frames) + 1


@cocotb.test()
async def test_data_path_bit_exact(dut):
    """Closed loop with drift and set-point steps: every internal value of every sample
    (phase table, error, PI output, DAC codes) equals the reference computed from the ADC codes."""
    b = await start(dut, SW_LOOP | SW_DRIFT | SW_STEP)
    await samples(dut, 900)
    lut = design_filters.fringe_lut()
    pi = BiquadRef((P['PI_B0'], P['PI_B1'], 0, 32768, 0))
    for i, e in enumerate(b.log):
        phase = (((e['adc0'] - 2048) << design_filters.PHASE_SHIFT) + e['drift']) & 0xFFFF
        det = lut[phase >> 4]
        err = 2048 + (P['STEP_AMPL'] if e['sq'] else -P['STEP_AMPL']) - det
        if not e['run']:
            pi.clear()
        ctrl = pi.step(err) if e['run'] else 0
        got = (e['det'], e['err'], e['ctrl'], e['dac_a'], e['dac_b'])
        exp = (det, err, ctrl, 2048 + ctrl, det)
        assert got == exp, f"sample {i}: (det, err, ctrl, DAC A, DAC B) = {got}, reference {exp}"
    frames = [(f['code_a'], f['code_b']) for f in b.dac.frames]
    logged = [(e['dac_a'], e['dac_b']) for e in b.log][:len(frames)]
    assert frames == logged, "codes on the DAC wires differ from the internal ones"
    assert len(frames) >= 899 + P['WARM_SAMPLES'], f"{len(frames)} DAC frames in 900 samples: samples were dropped"
    drifts = [e['drift'] for e in b.log]
    assert max(abs(x - y) for x, y in zip(drifts, drifts[1:])) == P['DRIFT_STEP'], "drift is not a ramp of DRIFT_STEP per sample"


@cocotb.test()
async def test_lock_and_step_response(dut):
    """The loop locks at mid-fringe; set-point steps settle with the designed time constant."""
    b = await start(dut, SW_LOOP | SW_STEP)
    await samples(dut, 6 * P['STEP_HALF'] + 20)
    det = [e['det'] for e in running(b.log)]
    sq = [e['sq'] for e in running(b.log)]
    assert not any(e['led'] & 0b1000 for e in b.log), "controller at a rail during start-up or the steps"
    assert {e['dac_a'] for e in b.log[:P['WARM_SAMPLES'] - 2]} == {2048}, "DAC A must wait at mid-scale during start-up"
    edges = [i for i in range(1, len(sq)) if sq[i] != sq[i - 1]]
    assert len(edges) >= 5, f"only {len(edges)} set-point steps seen"
    t63 = []
    for i in edges[1:5]:                                   # skip the first one (start-up)
        new = 2048 + (P['STEP_AMPL'] if sq[i] else -P['STEP_AMPL'])
        old = 2048 - (P['STEP_AMPL'] if sq[i] else -P['STEP_AMPL'])
        seg = det[i:i + P['STEP_HALF']]
        assert abs(det[i - 1] - old) <= 4, f"step at sample {i}: detector at {det[i - 1]} before the step, expected {old}"
        assert abs(seg[-1] - new) <= 4, f"step at sample {i}: detector settles at {seg[-1]}, set-point {new}"
        frac = [(v - old) / (new - old) for v in seg]
        assert max(frac) <= 1.05, f"step at sample {i}: overshoot {100 * (max(frac) - 1):.1f} %"
        t63.append(next(k for k, f in enumerate(frac) if f >= 1 - math.exp(-1)))
    lo, hi = 0.7 * TAU_CL, 1.4 * TAU_CL
    assert all(lo <= t <= hi for t in t63), f"63 % rise after {t63} samples, design {TAU_CL:.0f} (allowed {lo:.0f}..{hi:.0f})"
    dut._log.info(f"closed-loop 63 % rise: {t63} samples (design {TAU_CL:.0f}); plant alone {TAU_SAMPLES:.0f}")


@cocotb.test()
async def test_drift_rejected(dut):
    """Same drift, loop open and closed: open, the detector sweeps the fringe; closed, it stays put."""
    n = 2 * 16384 // P['DRIFT_STEP'] + 300                 # half a triangle period and a bit
    b = await start(dut, SW_DRIFT)
    await samples(dut, n)
    open_dev = max(abs(e['det'] - 2048) for e in running(b.log))
    assert open_dev > 1800, f"loop open: detector deviates only {open_dev} codes under +-1/4 turn of drift"
    dut.btn0.value = 1
    await ClockCycles(dut.clk100, 8)
    b = await start(dut, SW_LOOP | SW_DRIFT, clock=False)
    await samples(dut, n)
    closed_dev = max(abs(e['det'] - 2048) for e in running(b.log))
    # ramp of DRIFT_STEP per sample into an integrating loop: error = rate * closed-loop time constant
    rate = P['DRIFT_STEP'] * 2 * math.pi / 65536 * 2047          # detector codes per sample
    bound = 1.6 * rate * TAU_CL + 24
    assert closed_dev <= bound, f"loop closed: detector deviates {closed_dev} codes, expected at most {bound:.0f}"
    ctrl = [e['ctrl'] for e in running(b.log)]
    assert max(ctrl) - min(ctrl) > 800, "the controller did not move to cancel the drift"
    dut._log.info(f"drift of +-1/4 turn: detector deviation {open_dev} codes open, {closed_dev} closed "
                  f"(x{open_dev / max(closed_dev, 1):.0f}); control swing {max(ctrl) - min(ctrl)} codes")


@cocotb.test()
async def test_rail_and_recovery(dut):
    """ADC channel 0 stuck (feedback path broken): the controller goes to the rail and says so.
    When the path is restored the loop relocks quickly, because the integrator did not wind up."""
    b = await start(dut, SW_LOOP, ch0_stuck=1024)          # phase -1/4 turn: detector at the bottom
    await samples(dut, 1500)
    last = b.log[-1]
    assert last['ctrl'] == 2047 and last['dac_a'] == 4095, f"controller at {last['ctrl']}, expected the rail 2047"
    assert last['led'] & 0b1000 and not last['led'] & 0b0100, f"leds {last['led']:04b}: expected rail on, locked off"
    b.v = b.target = 4095.0                                # the capacitor has followed DAC A all this time
    b.ch0_stuck = None
    k0 = len(b.log)
    await samples(dut, 700)
    errs = [abs(e['err']) for e in b.log[k0:]]
    relock = next((i for i in range(len(errs)) if max(errs[i:i + 50]) < 64), None)
    assert relock is not None and relock < 12 * TAU_CL, f"relock after {relock} samples, expected under {12 * TAU_CL:.0f}"
    assert b.log[-1]['led'] & 0b0100, "locked LED off after recovery"
    dut._log.info(f"after 1500 samples at the rail, error back inside the lock window in {relock} samples")


@cocotb.test()
async def test_gate_timing(dut):
    """Gate: GATE_LEN samples high out of GATE_PERIOD, rising with sample 0."""
    b = await start(dut, 0)
    await samples(dut, 3 * P['GATE_PERIOD'] + 5, warm=False)
    g = [(e['n'], e['gate']) for e in b.log]
    for n, gate in g:
        assert gate == int(n < P['GATE_LEN']), f"sample {n} of the period: gate = {gate}"
    ns = [n for n, _ in g]
    assert all((y - x) % P['GATE_PERIOD'] == 1 for x, y in zip(ns, ns[1:])), "sample index does not count modulo GATE_PERIOD"


@cocotb.test()
async def test_gated_readout(dut):
    """A pulse of known amplitude inside the gate, on an offset and with other signals outside
    the window: the shot value is the bit-exact FIR output for the last sample of the gate,
    it equals the amplitude within 3 codes, and it is what DAC B holds until the next shot."""
    amps = [900, -600, 1500, 0, -1500, 250]
    state = {'shot': -1}
    first = P['GATE_LEN'] - N                                # the window is the last N samples of the gate

    def ch1(n):
        if n == 0:
            state['shot'] += 1
        a = amps[state['shot'] % len(amps)]
        if first <= n < P['GATE_LEN']:
            return 2048 + 300 + a * PULSE[n - first]         # offset of 300 codes under the pulse
        return 2048 + 300 + (150 if (n // 7) % 2 else -150)  # activity outside the window

    b = await start(dut, SW_SHOT, ch1_fn=ch1)
    await samples(dut, len(amps) * P['GATE_PERIOD'] + P['GATE_LEN'] + 5, warm=False)
    fir = FirRef(H)
    ref_shot, got, k = 0, [], 0
    for i, e in enumerate(b.log):
        y = fir.step(e['adc1'] - 2048)
        if e['n'] == P['GATE_LEN'] - 1:
            ref_shot = y
            if k >= 1:                                       # the first window may hold start-up samples
                a = amps[k % len(amps)]
                assert abs(y - a) <= 3, f"shot {k}: FIR gives {y}, pulse amplitude {a}"
            got.append(y)
            k += 1
        if i >= N:
            assert e['shot'] == ref_shot, f"sample {i}: shot register {e['shot']}, reference {ref_shot}"
            assert e['dac_b'] == min(max(2048 + ref_shot, 0), 4095), f"sample {i}: DAC B = {e['dac_b']}, expected 2048 + {ref_shot}"
    assert k >= len(amps), f"only {k} shots seen"
    dut._log.info(f"shots {got} for amplitudes {[amps[i % len(amps)] for i in range(k)]}")
