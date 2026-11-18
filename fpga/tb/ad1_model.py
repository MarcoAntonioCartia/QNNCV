"""Behavioural model of the Pmod AD1 (two AD7476A) for the cocotb tests.

The falling edge of CS samples both inputs (sample_fn(t_ns) -> (code0, code1))
and puts the first leading zero on the data lines; every SCLK falling edge
clocks out the next bit. To catch a host that reads at the wrong moment, each
bit is driven as X from T_HOLD after the edge that replaces it until T_ACCESS
after that edge. A conversion without exactly 16 falling edges, an SCLK above
20 MHz or a CS high time below CS_HIGH_MIN is an assertion failure.

T_ACCESS, T_HOLD and CS_HIGH_MIN are deliberately pessimistic test values; the
datasheet figures behind them are listed as "por verificar" in fpga/README.md.
"""

import cocotb
from cocotb.triggers import FallingEdge, RisingEdge, First, Timer
from cocotb.types import Logic
from cocotb.utils import get_sim_time

SCLK_MAX_HZ = 20e6
T_ACCESS = 40.0       # ns from the clocking edge until the new bit is valid
T_HOLD = 7.0          # ns the old bit stays valid after the clocking edge
CS_HIGH_MIN = 50.0    # ns between conversions


def now():
    return get_sim_time(unit='ns')


class Ad1Model:
    def __init__(self, cs_n, sclk, d0, d1, sample_fn):
        self.cs_n, self.sclk, self.d0, self.d1 = cs_n, sclk, d0, d1
        self.sample_fn = sample_fn
        self.samples = []            # dicts: t, code0, code1
        self.t_cs_rise = None
        self.min_period = float('inf')
        self.d0.value = 0
        self.d1.value = 0

    async def _put(self, seq, b0, b1):
        """Old bit invalid after T_HOLD, new bit valid after T_ACCESS."""
        await Timer(T_HOLD, unit='ns')
        if seq != self.seq:
            return
        self.d0.value = Logic('X')
        self.d1.value = Logic('X')
        await Timer(T_ACCESS - T_HOLD, unit='ns')
        if seq != self.seq:
            return
        self.d0.value = b0
        self.d1.value = b1

    async def run(self):
        self.seq = 0
        while True:
            await FallingEdge(self.cs_n)
            t = now()
            if self.t_cs_rise is not None:
                assert t - self.t_cs_rise >= CS_HIGH_MIN, f"CS high for {t - self.t_cs_rise} ns (min {CS_HIGH_MIN})"
            assert int(self.sclk.value) == 1, "SCLK must be high when CS falls"
            c0, c1 = self.sample_fn(t)
            assert 0 <= c0 <= 4095 and 0 <= c1 <= 4095
            self.samples.append({'t': t, 'code0': c0, 'code1': c1})
            w0, w1 = [int(b) for b in f'{c0:016b}'], [int(b) for b in f'{c1:016b}']
            self.seq += 1
            cocotb.start_soon(self._put(self.seq, w0[0], w1[0]))
            nfall, t_fall = 0, None
            fall, end = FallingEdge(self.sclk), RisingEdge(self.cs_n)
            while True:
                got = await First(fall, end)
                if got is end:
                    break
                tf = now()
                if t_fall is not None:
                    self.min_period = min(self.min_period, tf - t_fall)
                t_fall = tf
                nfall += 1
                self.seq += 1
                if nfall < 16:
                    cocotb.start_soon(self._put(self.seq, w0[nfall], w1[nfall]))
                else:                                   # 16th edge: the line is released
                    self.d0.value = Logic('Z')
                    self.d1.value = Logic('Z')
            self.t_cs_rise = now()
            assert nfall == 16, f"conversion with {nfall} SCLK falling edges, expected 16"
            assert 1e9 / self.min_period <= SCLK_MAX_HZ * 1.000001, f"SCLK period {self.min_period} ns: above 20 MHz"
