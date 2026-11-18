"""Bit-exact Python references of rtl/iir_biquad.v and rtl/fir_serial.v."""

FRAC, CFRAC, HSHIFT = 12, 15, 17


class BiquadRef:
    """coef = (B0, B1, B2, NA1, NA2) as Q3.15 integers; x and y integers (13 bits signed)."""

    def __init__(self, coef, y_max=2047, y_min=-2047):
        self.c, self.hi, self.lo = coef, y_max << FRAC, y_min << FRAC
        self.clear()

    def clear(self):
        self.x1 = self.x2 = self.y1 = self.y2 = 0
        self.sat = 0

    def step(self, x):
        x0 = x << FRAC
        acc = sum(c * d for c, d in zip(self.c, (x0, self.x1, self.x2, self.y1, self.y2)))
        res = (acc + (1 << (CFRAC - 1))) >> CFRAC
        self.sat = int(res > self.hi or res < self.lo)
        res = min(max(res, self.lo), self.hi)
        self.x2, self.x1, self.y2, self.y1 = self.x1, x0, self.y1, res
        return res >> FRAC


class FirRef:
    def __init__(self, h):
        self.h, self.x = list(h), [0] * len(h)      # x[0] newest

    def step(self, x):
        self.x = [x] + self.x[:-1]
        acc = sum(h * v for h, v in zip(self.h, self.x))
        res = (acc + (1 << (HSHIFT - 1))) >> HSHIFT
        self.sat = int(res > 4095 or res < -4096)
        return min(max(res, -4096), 4095)


def read_signed_mem(path, bits):
    out = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.split('//')[0].strip()
            if line:
                v = int(line, 16)
                out.append(v - (1 << bits) if v >> (bits - 1) else v)
    return out
