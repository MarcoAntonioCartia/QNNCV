"""
Phase table -> memory initialisation file for the FPGA demo (fpga/rtl/phase_rom.v).

Two input formats, told apart by the CSV header:

  gates   the gates_<run>.csv written by scripts/hw_sweep.py. Every
          Mach-Zehnder of the Clements meshes U1 and U2 of a layer becomes one
          step: internal phase on channel A, external phase on channel B.
          Rows that are not Mach-Zehnder phases (output phases, S, D, K) are
          ignored.
  simple  columns layer,phase_a,phase_b[,ramp,gate] with the phases in
          radians; one row per step.

Phase -> 12-bit code is the same illustrative linear map as hw_sweep.dac12:
code = round((phase mod 2 pi) / (2 pi) * 4095), i.e. a shifter with a 2 pi
swing over the DAC range. A real device needs its own calibration curve.

Word layout (32 bits, one per line, hexadecimal):
  [31] last  [30] ramp  [29] gate  [28] mark (first step of a layer)
  [27:24] layer  [23:12] code B  [11:0] code A

    python fpga/scripts/phases_to_mem.py analysis/hw_sweep/gates_<run>.csv \
        -o fpga/data/phase_table.mem --ramp alternate
    python fpga/scripts/phases_to_mem.py fpga/data/example_simple.csv -o out.mem

Only the standard library is used.
"""

import argparse
import csv
import math
import re
import sys

TWO_PI = 2.0 * math.pi
MAX_LAYER = 14            # 15 on the switches means "play all layers"
EMPTY = 0x80000000        # unused entry: an empty 'last' step
VREF = 3.3                # Pmod DA2: the reference is the supply rail


def dac12(phase):
    """Same map as scripts/hw_sweep.py::dac12."""
    return int(round((phase % TWO_PI) / TWO_PI * 4095))


def code_to_volts(code, vref=VREF):
    return vref * code / 4096.0


def pack(step):
    return ((step['last'] << 31) | (step['ramp'] << 30) | (step['gate'] << 29)
            | (step['mark'] << 28) | (step['layer'] << 24)
            | (step['code_b'] << 12) | step['code_a'])


def unpack(word):
    return {'last': (word >> 31) & 1, 'ramp': (word >> 30) & 1, 'gate': (word >> 29) & 1,
            'mark': (word >> 28) & 1, 'layer': (word >> 24) & 0xF,
            'code_b': (word >> 12) & 0xFFF, 'code_a': word & 0xFFF}


def ramp_flag(mode, layer):
    return {'none': 0, 'all': 1, 'alternate': layer % 2}[mode]


def read_gates(rows, elements, ramp):
    """hw_sweep gates CSV -> steps. One step per Mach-Zehnder, in file order."""
    mz = {}                                   # (layer, element) -> {'internal':..,'external':..}
    order = []
    for r in rows:
        m = re.fullmatch(r'(U[12])\.MZI(\d+)', r['element'])
        if not m or r['parameter'] not in ('phase_internal', 'phase_external'):
            continue
        if m.group(1) not in elements:
            continue
        key = (int(r['layer']), r['element'])
        if key not in mz:
            mz[key] = {'modes': r['modes']}
            order.append(key)
        phase = float(r['value_rad_or_r'])
        kind = r['parameter'].split('_')[1]
        mz[key][kind] = phase
        col = (r.get('dac12_illustrative') or '').strip()
        if col and int(col) != dac12(phase):
            # the CSV rounds the phase to 6 decimals, so the code can differ by one
            if abs(int(col) - dac12(phase)) > 1:
                raise ValueError(f"layer {key[0]} {key[1]}: code {dac12(phase)} from the phase "
                                 f"does not match the dac12_illustrative column ({col})")
            mz[key][kind + '_code'] = int(col)
    steps = []
    for key in order:
        e = mz[key]
        if 'internal' not in e or 'external' not in e:
            raise ValueError(f"layer {key[0]} {key[1]}: internal or external phase missing")
        steps.append({'layer': key[0], 'label': f"{key[1]} modes {e['modes']}",
                      'phase_a': e['internal'], 'phase_b': e['external'],
                      'code_a': e.get('internal_code', dac12(e['internal'])),
                      'code_b': e.get('external_code', dac12(e['external'])),
                      'ramp': ramp_flag(ramp, key[0]), 'gate': 1})
    return steps


def read_simple(rows, ramp):
    steps = []
    for i, r in enumerate(rows):
        layer = int(r['layer'])
        pa, pb = float(r['phase_a']), float(r['phase_b'])
        rflag = r.get('ramp')
        gflag = r.get('gate')
        steps.append({'layer': layer, 'label': f'row {i}', 'phase_a': pa, 'phase_b': pb,
                      'code_a': dac12(pa), 'code_b': dac12(pb),
                      'ramp': int(rflag) if rflag not in (None, '') else ramp_flag(ramp, layer),
                      'gate': int(gflag) if gflag not in (None, '') else 1})
    return steps


def finish(steps, addr_w):
    """Checks, plus the mark (first step of each layer) and last flags."""
    if not steps:
        raise ValueError("the table has no steps")
    if len(steps) > (1 << addr_w):
        raise ValueError(f"{len(steps)} steps do not fit in 2^{addr_w} words")
    prev = None
    for s in steps:
        if not 0 <= s['layer'] <= MAX_LAYER:
            raise ValueError(f"layer {s['layer']} outside 0..{MAX_LAYER}")
        if s['ramp'] not in (0, 1) or s['gate'] not in (0, 1):
            raise ValueError("ramp and gate must be 0 or 1")
        s['mark'] = int(s['layer'] != prev)
        s['last'] = 0
        prev = s['layer']
    steps[-1]['last'] = 1
    return steps


def load(path, elements=('U1', 'U2'), ramp='alternate', addr_w=8):
    with open(path, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(l for l in fh if not l.lstrip().startswith('#')))
    if not rows:
        raise ValueError(f"{path}: empty")
    if 'element' in rows[0]:
        steps = read_gates(rows, elements, ramp)
    elif 'phase_a' in rows[0]:
        steps = read_simple(rows, ramp)
    else:
        raise ValueError(f"{path}: header is neither the hw_sweep gates format nor the simple one")
    return finish(steps, addr_w)


def write_mem(steps, path, source, addr_w=8):
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(f"// phase table for fpga/rtl/phase_rom.v, written by phases_to_mem.py\n")
        fh.write(f"// source: {source}\n")
        fh.write("// word = last|ramp|gate|mark|layer[3:0]|code_b[11:0]|code_a[11:0]\n")
        for i, s in enumerate(steps):
            fh.write(f"{pack(s):08X}  // {i:3d} layer {s['layer']:2d} {s['label']:<22s} "
                     f"A={s['code_a']:4d} ({code_to_volts(s['code_a']):.3f} V) "
                     f"B={s['code_b']:4d} ({code_to_volts(s['code_b']):.3f} V) "
                     f"{'ramp' if s['ramp'] else 'step'}{' gate' if s['gate'] else ''}\n")
        # every word of the ROM is written, so nothing depends on how a tool fills the rest
        for i in range(len(steps), 1 << addr_w):
            fh.write(f"{EMPTY:08X}\n")


def table_of(words):
    """The words that are played: up to and including the first 'last' one."""
    n = next(i for i, w in enumerate(words) if w >> 31)
    return words[:n + 1]


def read_mem(path):
    """Words of a .mem file, as $readmemh sees them (comments dropped)."""
    words = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.split('//')[0].strip()
            if line:
                words.append(int(line, 16))
    return words


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('table', help='gates_<run>.csv from hw_sweep.py, or a simple CSV')
    ap.add_argument('-o', '--out', default='phase_table.mem')
    ap.add_argument('--elements', default='U1,U2', help='meshes to play (gates format)')
    ap.add_argument('--ramp', choices=('none', 'all', 'alternate'), default='alternate',
                    help='alternate: step on even layers, ramp on odd ones (default)')
    ap.add_argument('--addr-width', type=int, default=8, help='ADDR_W of the ROM')
    a = ap.parse_args(argv)
    steps = load(a.table, tuple(a.elements.split(',')), a.ramp, a.addr_width)
    write_mem(steps, a.out, a.table.replace('\\', '/'), a.addr_width)
    layers = sorted({s['layer'] for s in steps})
    print(f"{a.out}: {len(steps)} steps, layers {layers}")
    for s in steps:
        print(f"  layer {s['layer']:2d} {s['label']:<22s} A {s['phase_a']:.4f} rad -> {s['code_a']:4d}"
              f"   B {s['phase_b']:.4f} rad -> {s['code_b']:4d}   {'ramp' if s['ramp'] else 'step'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
