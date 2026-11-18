"""Behavioural model of the Pmod DA2 (two DAC121S101) for the cocotb tests.

It watches SYNC, SCLK, DINA and DINB as the chip would: bits are sampled on
the falling edges of SCLK while SYNC is low, and a frame is accepted only if
it has exactly 16 falling edges. Every protocol or timing violation is an
assertion failure. Accepted frames are appended to `frames`.

Timing limits checked (ns). The first three are far stricter than the part
needs; the datasheet values behind them are listed in fpga/README.md.
"""

from cocotb.triggers import FallingEdge, RisingEdge, ValueChange, First
from cocotb.utils import get_sim_time

SCLK_MAX_HZ = 30e6      # datasheet: SCLK frequency, max
SYNC_HIGH_MIN = 20.0    # datasheet: SYNC high time, min over temperature at 2.7-3.6 V
SETUP_MIN = 20.0        # test margin for data setup before the falling edge
HOLD_MIN = 20.0         # test margin for data hold after the falling edge
LEAD_MIN = 20.0         # test margin, SYNC fall to first SCLK fall


def now():
    return get_sim_time(unit='ns')


class Da2Model:
    def __init__(self, sync_n, sclk, dina, dinb, log=None):
        self.sync_n, self.sclk, self.dina, self.dinb = sync_n, sclk, dina, dinb
        self.log = log
        self.frames = []            # dicts: t_sync, t_update, code_a, code_b, word_a, word_b
        self.code_a = 0             # DAC registers (power-on reset: zero scale)
        self.code_b = 0
        self.in_frame = False
        self.t_last_fall = None     # last SCLK falling edge inside the current frame
        self.t_change = {'dina': -1e9, 'dinb': -1e9}
        self.t_sync_rise = None
        self.min_setup = self.min_hold = self.min_half = self.min_sync_high = float('inf')

    async def _watch_data(self, name, sig):
        while True:
            await ValueChange(sig)
            t = now()
            self.t_change[name] = t
            if self.in_frame and self.t_last_fall is not None:
                hold = t - self.t_last_fall
                self.min_hold = min(self.min_hold, hold)
                assert hold >= HOLD_MIN, f"{name} changed {hold} ns after the SCLK falling edge (min {HOLD_MIN})"

    async def run(self):
        import cocotb
        cocotb.start_soon(self._watch_data('dina', self.dina))
        cocotb.start_soon(self._watch_data('dinb', self.dinb))
        while True:
            await FallingEdge(self.sync_n)
            t_sync = now()
            assert int(self.sclk.value) == 1, "SCLK must idle high when SYNC falls"
            if self.t_sync_rise is not None:
                high = t_sync - self.t_sync_rise
                self.min_sync_high = min(self.min_sync_high, high)
                assert high >= SYNC_HIGH_MIN, f"SYNC high for {high} ns between frames (min {SYNC_HIGH_MIN})"
            self.in_frame, self.t_last_fall = True, None
            wa = wb = 0
            nbits, t_edge, t_update = 0, t_sync, None
            fall, rise, end = FallingEdge(self.sclk), RisingEdge(self.sclk), RisingEdge(self.sync_n)
            while True:
                got = await First(fall, rise, end)
                t = now()
                if got is end:
                    break
                half = t - t_edge
                if got is fall:
                    if nbits == 0:
                        assert half >= LEAD_MIN, f"SYNC fall to first SCLK fall {half} ns (min {LEAD_MIN})"
                    else:
                        self.min_half = min(self.min_half, half)
                    for name in ('dina', 'dinb'):
                        setup = t - self.t_change[name]
                        self.min_setup = min(self.min_setup, setup)
                        assert setup >= SETUP_MIN, f"{name} setup {setup} ns before SCLK fall (min {SETUP_MIN})"
                    assert self.dina.value.is_resolvable and self.dinb.value.is_resolvable, "X/Z on a data line"
                    wa = (wa << 1) | int(self.dina.value)
                    wb = (wb << 1) | int(self.dinb.value)
                    nbits += 1
                    self.t_last_fall = t
                    if nbits == 16:
                        t_update = t
                else:
                    self.min_half = min(self.min_half, half)
                t_edge = t
            self.in_frame = False
            self.t_sync_rise = now()
            assert nbits == 16, f"frame with {nbits} SCLK falling edges, expected exactly 16"
            assert int(self.sclk.value) == 1, "SCLK must be high when SYNC rises"
            assert 1e9 / (2 * self.min_half) <= SCLK_MAX_HZ, \
                f"SCLK half period {self.min_half} ns is faster than {SCLK_MAX_HZ / 1e6} MHz"
            for ch, w in (('A', wa), ('B', wb)):
                assert (w >> 12) & 0x3 == 0, f"channel {ch}: power-down bits {(w >> 12) & 3:02b}, expected 00 (normal)"
            self.code_a, self.code_b = wa & 0xFFF, wb & 0xFFF
            self.frames.append({'t_sync': t_sync, 't_update': t_update, 'word_a': wa, 'word_b': wb,
                                'code_a': self.code_a, 'code_b': self.code_b})
