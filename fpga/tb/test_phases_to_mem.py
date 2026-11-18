"""Plain-Python checks of scripts/phases_to_mem.py (run by run_tests.py or pytest)."""
import math
import sys
import tempfile
from pathlib import Path

FPGA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FPGA / 'scripts'))
import phases_to_mem as p2m  # noqa: E402


def test_dac12_map():
    assert p2m.dac12(0.0) == 0
    assert p2m.dac12(math.pi) == 2048                 # round(2047.5)
    assert p2m.dac12(2 * math.pi - 1e-9) == 4095
    assert p2m.dac12(2 * math.pi) == 0                # the phase wraps
    assert p2m.dac12(-math.pi / 2) == p2m.dac12(3 * math.pi / 2)
    assert all(0 <= p2m.dac12(k * 0.001) <= 4095 for k in range(-7000, 7000))


def test_pack_roundtrip():
    s = {'last': 1, 'ramp': 0, 'gate': 1, 'mark': 1, 'layer': 9, 'code_b': 0xABC, 'code_a': 0x123}
    assert p2m.pack(s) == 0xB9ABC123
    assert p2m.unpack(0xB9ABC123) == s


def test_gates_example():
    """hw_sweep format: 3 layers x (U1, U2) x 3 Mach-Zehnder = 18 steps; other rows ignored."""
    steps = p2m.load(str(FPGA / 'data' / 'gates_example.csv'))
    assert len(steps) == 18
    assert [s['layer'] for s in steps] == [0] * 6 + [1] * 6 + [2] * 6
    assert [s['mark'] for s in steps] == ([1] + [0] * 5) * 3
    assert [s['last'] for s in steps] == [0] * 17 + [1]
    assert [s['ramp'] for s in steps] == [0] * 6 + [1] * 6 + [0] * 6       # alternate
    for s in steps:
        assert abs(s['code_a'] - p2m.dac12(s['phase_a'])) <= 1
        assert abs(s['code_b'] - p2m.dac12(s['phase_b'])) <= 1
    only_u1 = p2m.load(str(FPGA / 'data' / 'gates_example.csv'), elements=('U1',), ramp='all')
    assert len(only_u1) == 9 and all(s['ramp'] for s in only_u1)


def test_committed_mem_is_current():
    """data/phase_table.mem is what the script produces from data/gates_example.csv."""
    steps = p2m.load(str(FPGA / 'data' / 'gates_example.csv'))
    words = p2m.read_mem(str(FPGA / 'data' / 'phase_table.mem'))
    assert len(words) == 256 and words[18:] == [p2m.EMPTY] * 238
    assert p2m.table_of(words) == [p2m.pack(s) for s in steps]


def test_simple_format_and_limits():
    steps = p2m.load(str(FPGA / 'data' / 'example_simple.csv'))
    assert [(s['layer'], s['code_a'], s['code_b'], s['ramp'], s['gate']) for s in steps] == [
        (0, 0, 2048, 0, 1), (0, 1024, 3071, 0, 1), (0, 2048, 0, 0, 0), (1, 4095, 683, 1, 1), (1, 512, 3583, 1, 1)]
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / 'bad.csv'
        bad.write_text('layer,phase_a,phase_b\n15,0,0\n')
        try:
            p2m.load(str(bad))
        except ValueError as e:
            assert 'layer 15' in str(e)
        else:
            raise AssertionError('layer 15 accepted')
        big = Path(d) / 'big.csv'
        big.write_text('layer,phase_a,phase_b\n' + '0,0,0\n' * 17)
        try:
            p2m.load(str(big), addr_w=4)
        except ValueError as e:
            assert 'do not fit' in str(e)
        else:
            raise AssertionError('17 steps accepted in 16 words')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f"{len(tests)} script checks passed")
