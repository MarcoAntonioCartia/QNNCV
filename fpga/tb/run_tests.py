"""Builds and runs the cocotb tests with Icarus Verilog.

    python fpga/tb/run_tests.py            # script checks and both test benches
    python fpga/tb/run_tests.py spi        # only the DAC driver
    python fpga/tb/run_tests.py demo       # only the top level
    WAVES=1 python fpga/tb/run_tests.py    # also dump FST waveforms
"""
import os
import sys
from pathlib import Path

from cocotb_tools.runner import get_runner

TB = Path(__file__).resolve().parent
RTL = TB.parent / 'rtl'
sys.path.insert(0, str(TB.parent / 'scripts'))
import phases_to_mem  # noqa: E402

# top-level parameters for simulation: short ticks so a full table takes milliseconds
DEMO = {'TICK_DIV': 200, 'RAMP_LOG2': 3, 'SETTLE_TICKS': 2, 'GATE_TICKS': 3, 'HOLD_TICKS': 2,
        'SPI_HALF_DIV': 4, 'SPI_GAP_DIV': 8, 'ADDR_W': 4}


def run(which):
    waves = bool(os.environ.get('WAVES'))
    results = []
    if which in ('all', 'spi'):
        r = get_runner('icarus')
        r.build(sources=[RTL / 'da2_spi.v'], hdl_toplevel='da2_spi', always=True,
                parameters={'HALF_DIV': 4, 'GAP_DIV': 8}, build_dir=TB / 'sim_build' / 'spi',
                waves=waves, timescale=('1ns', '1ps'))
        results.append(r.test(hdl_toplevel='da2_spi', test_module='test_da2_spi', test_dir=TB,
                              waves=waves, extra_env={'TB_HALF_DIV': '4', 'TB_GAP_DIV': '8'}))
    if which in ('all', 'demo'):
        build = TB / 'sim_build' / 'demo'
        build.mkdir(parents=True, exist_ok=True)
        mem = build / 'tb_table.mem'
        steps = phases_to_mem.load(str(TB / 'tb_table.csv'), addr_w=DEMO['ADDR_W'])
        phases_to_mem.write_mem(steps, str(mem), 'tb/tb_table.csv')
        r = get_runner('icarus')
        r.build(sources=[RTL / f for f in ('da2_spi.v', 'phase_rom.v', 'phase_sequencer.v', 'arty_phase_demo.v')],
                hdl_toplevel='arty_phase_demo', always=True, build_dir=build, waves=waves,
                parameters={**DEMO, 'MEM_FILE': f'"{mem.as_posix()}"'}, timescale=('1ns', '1ps'))
        env = {f'TB_{k}': str(v) for k, v in DEMO.items()}
        env['TB_MEM'] = str(mem)
        results.append(r.test(hdl_toplevel='arty_phase_demo', test_module='test_arty_phase_demo',
                              test_dir=TB, waves=waves, extra_env=env))
    return results


if __name__ == '__main__':
    from cocotb_tools.check_results import get_results
    import subprocess
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which == 'all':
        subprocess.run([sys.executable, str(TB / 'test_phases_to_mem.py')], check=True)
    total = failed = 0
    for xml in run(which):
        t, f = get_results(xml)
        total, failed = total + t, failed + f
    print(f"\n{total} tests, {total - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
