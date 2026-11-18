"""Builds and runs the cocotb tests with Icarus Verilog.

    python fpga/tb/run_tests.py            # script checks and every test bench
    python fpga/tb/run_tests.py spi fir    # only the named benches
    WAVES=1 python fpga/tb/run_tests.py    # also dump FST waveforms

Benches: spi (DAC driver), demo (phase table top level), adc (ADC reader),
iir, pi (the biquad, as a general section and as the PI of the demo), fir,
lock (closed-loop top level).
"""
import os
import subprocess
import sys
from pathlib import Path

from cocotb_tools.runner import get_runner

TB = Path(__file__).resolve().parent
RTL = TB.parent / 'rtl'
DATA = TB.parent / 'data'
sys.path.insert(0, str(TB.parent / 'scripts'))
import phases_to_mem  # noqa: E402
import design_filters  # noqa: E402

# top-level parameters for simulation: short ticks so a full table takes milliseconds
DEMO = {'TICK_DIV': 200, 'RAMP_LOG2': 3, 'SETTLE_TICKS': 2, 'GATE_TICKS': 3, 'HOLD_TICKS': 2,
        'SPI_HALF_DIV': 4, 'SPI_GAP_DIV': 8, 'ADDR_W': 4}
# a general second-order section (complex poles at radius 0.84) and the PI of the demo
B0, B1, _ = design_filters.pi_coeffs()
IIR = {'B0': 6554, 'B1': 9830, 'B2': 6554, 'NA1': 39322, 'NA2': -22938, 'Y_MAX': 2047, 'Y_MIN': -2047}
PI = {'B0': B0, 'B1': B1, 'B2': 0, 'NA1': 32768, 'NA2': 0, 'Y_MAX': 2047, 'Y_MIN': -2047}
# closed loop: fewer clocks per sample (the plant model scales with it), and shorter gate,
# step and start-up periods; the drift is 32 times faster than on the board
LOCK = {'SAMPLE_DIV': 400, 'GATE_PERIOD': 200, 'GATE_LEN': 80, 'STEP_HALF': 150, 'STEP_AMPL': 300,
        'DRIFT_STEP': 32, 'WARM_SAMPLES': 500, 'PI_B0': B0, 'PI_B1': B1}


def sv(n, bits=18):
    """Signed parameter value as a literal Icarus accepts on the command line (two's complement)."""
    return f"{bits}'sh{n & ((1 << bits) - 1):X}"


def bench(name, sources, top, module, parameters=None, env=None):
    waves = bool(os.environ.get('WAVES'))
    r = get_runner('icarus')
    r.build(sources=[RTL / s for s in sources], hdl_toplevel=top, always=True,
            parameters=parameters or {}, build_dir=TB / 'sim_build' / name,
            waves=waves, timescale=('1ns', '1ps'))
    return r.test(hdl_toplevel=top, test_module=module, test_dir=TB, waves=waves,
                  extra_env={k: str(v) for k, v in (env or {}).items()})


def run(name):
    fir_mem = (DATA / 'fir_coeffs.mem').as_posix()
    if name == 'spi':
        return bench(name, ['da2_spi.v'], 'da2_spi', 'test_da2_spi', {'HALF_DIV': 4, 'GAP_DIV': 8},
                     {'TB_HALF_DIV': 4, 'TB_GAP_DIV': 8})
    if name == 'demo':
        build = TB / 'sim_build' / name
        build.mkdir(parents=True, exist_ok=True)
        mem = build / 'tb_table.mem'
        steps = phases_to_mem.load(str(TB / 'tb_table.csv'), addr_w=DEMO['ADDR_W'])
        phases_to_mem.write_mem(steps, str(mem), 'tb/tb_table.csv', DEMO['ADDR_W'])
        return bench(name, ['da2_spi.v', 'phase_rom.v', 'phase_sequencer.v', 'arty_phase_demo.v'],
                     'arty_phase_demo', 'test_arty_phase_demo',
                     {**DEMO, 'MEM_FILE': f'"{mem.as_posix()}"'},
                     {**{f'TB_{k}': v for k, v in DEMO.items()}, 'TB_MEM': mem})
    if name == 'adc':
        return bench(name, ['ad1_spi.v'], 'ad1_spi', 'test_ad1_spi', {'HALF_DIV': 4, 'GAP_DIV': 8},
                     {'TB_HALF_DIV': 4, 'TB_GAP_DIV': 8})
    if name in ('iir', 'pi'):
        p = IIR if name == 'iir' else PI
        par = {k: sv(v, 18 if k[0] in 'BN' else 13) for k, v in p.items()}
        return bench(name, ['iir_biquad.v'], 'iir_biquad', 'test_iir_biquad', par,
                     {f'TB_{k}': v for k, v in p.items()})
    if name == 'fir':
        return bench(name, ['fir_serial.v'], 'fir_serial', 'test_fir_serial',
                     {'COEF_FILE': f'"{fir_mem}"'}, {'TB_FIR': fir_mem})
    if name == 'lock':
        par = {k: (sv(v) if k.startswith('PI_') else v) for k, v in LOCK.items()}
        par['FIR_FILE'] = f'"{fir_mem}"'
        par['FRINGE_FILE'] = f'"{(DATA / "fringe_lut.mem").as_posix()}"'
        return bench(name, ['da2_spi.v', 'ad1_spi.v', 'iir_biquad.v', 'fir_serial.v', 'arty_lock_demo.v'],
                     'arty_lock_demo', 'test_arty_lock_demo', par,
                     {**{f'TB_{k}': v for k, v in LOCK.items()}, 'TB_FIR': fir_mem,
                      'TB_FRINGE': (DATA / 'fringe_lut.mem').as_posix()})
    raise SystemExit(f"unknown bench '{name}'")


ALL = ('spi', 'demo', 'adc', 'iir', 'pi', 'fir', 'lock')

if __name__ == '__main__':
    from cocotb_tools.check_results import get_results
    names = sys.argv[1:] or ALL
    if not sys.argv[1:]:
        for script in ('test_phases_to_mem.py', 'test_design_filters.py'):
            subprocess.run([sys.executable, str(TB / script)], check=True)
    total = failed = 0
    summary = []
    for name in names:
        t, f = get_results(run(name))
        total, failed = total + t, failed + f
        summary.append(f"{name} {t - f}/{t}")
    print(f"\n{total} tests, {total - failed} passed, {failed} failed   ({', '.join(summary)})")
    sys.exit(1 if failed else 0)
