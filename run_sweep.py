"""
Parallel hyperparameter sweep for 2D CV-QGAN.

Each experiment launches in its own PowerShell window so you can
monitor all runs live. The main script waits for completion and
prints a ranked summary table.

Range syntax (MATLAB-style):  start stop step
  --g-lr 0.001 0.01 0.001     -> [0.001, 0.002, ..., 0.01]
  --n-layers 4 10 2           -> [4, 6, 8, 10]

Multiple ranges produce the cartesian product of all combinations.

Usage:
    python run_sweep.py --family ring --g-lr 0.001 0.01 0.003
    python run_sweep.py --family ring --g-lr 0.001 0.01 0.003 --n-layers 4 8 2
    python run_sweep.py --family ring --g-lr 0.005 --n-layers 4 10 2  # fix one, sweep other
    python run_sweep.py --dry-run --family ring --g-lr 0.001 0.01 0.001
    python run_sweep.py --parallel 4 --family ring --n-layers 4 10 2
    python run_sweep.py --silent --family ring --g-lr 0.001 0.005 0.001
"""

import argparse
import itertools
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

CONDA_ENV = "qnncv"

# ─────────────────────────────────────────────────────────────────────────────
# Sweepable hyperparameters: (cli_flag, type, abbreviation, min, max)
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_PARAMS = {
    "g_lr":              ("--g-lr",              float, "glr",   1e-6, 1.0),
    "d_lr":              ("--d-lr",              float, "dlr",   1e-6, 1.0),
    "n_layers":          ("--n-layers",          int,   "nl",    1,    20),
    "cutoff_dim":        ("--cutoff-dim",        int,   "c",     4,    20),
    "n_modes":           ("--n-modes",           int,   "m",     2,    6),
    "epochs":            ("--epochs",            int,   "ep",    50,   5000),
    "n_critic":          ("--n-critic",          int,   "nc",    1,    10),
    "gp_weight":         ("--gp-weight",         float, "gp",    0.0,  50.0),
    "gp_warmup":         ("--gp-warmup",         int,   "gpwup", 0,    500),
    "supervised_weight": ("--supervised-weight",  float, "sw",    0.0,  1.0),
    "supervised_warmup": ("--supervised-warmup",  int,   "swup",  0,    1000),
    "instance_noise":    ("--instance-noise",    float, "noise", 0.0,  1.0),
    "noise_anneal":      ("--noise-anneal",      int,   "nann",  0,    1000),
    "noise_floor":       ("--noise-floor",       float, "nfloor", 0.0, 1.0),
    "critic_blur":       ("--critic-blur",       float, "blur",  0.0,  5.0),
    "d_dropout":         ("--d-dropout",         float, "drop",  0.0,  0.9),
    "latent_scale":      ("--latent-scale",      float, "ls",    0.01, 10.0),
    "ket_penalty_weight":("--ket-penalty-weight", float, "kpen",  0.0,  500.0),
    "g_grad_clip":       ("--g-grad-clip",       float, "gclip", 0.0,  100.0),
    "grid_size":         ("--grid-size",         int,   "gs",    10,   100),
    "seed":              ("--seed",              int,   "seed",  0,    999999),
    "batch_size":        ("--batch-size",        int,   "bs",    1,    128),

}

MAX_EXPERIMENTS_WARN = 20


# ─────────────────────────────────────────────────────────────────────────────
# Range parsing & validation
# ─────────────────────────────────────────────────────────────────────────────

def parse_range(values, param_name, param_type, vmin, vmax):
    """Parse 1 or 3 values into a list.

    1 value:  fixed override (no sweep)
    3 values: start stop step  (MATLAB-style, inclusive of stop)
    """
    if len(values) == 1:
        v = param_type(values[0])
        _validate_bounds(v, param_name, vmin, vmax)
        return [v]

    if len(values) == 3:
        start, stop, step = [param_type(x) for x in values]
        if step == 0:
            print(f"  ERROR: Step for {param_name} cannot be zero.")
            sys.exit(1)
        if (stop - start) / step < 0:
            print(f"  ERROR: {param_name} range {start}:{step}:{stop} goes in the wrong "
                  f"direction (start={'>' if start > stop else '<'}stop but step "
                  f"is {'positive' if step > 0 else 'negative'}).")
            sys.exit(1)

        _validate_bounds(start, param_name, vmin, vmax)
        _validate_bounds(stop, param_name, vmin, vmax)

        # Generate values inclusive of stop
        if param_type is int:
            result = list(range(start, stop + 1, step))
        else:
            result = []
            v = start
            while v <= stop + step * 1e-9:  # float tolerance
                result.append(round(v, 10))
                v += step

        if not result:
            print(f"  ERROR: {param_name} range {start}:{step}:{stop} produced no values.")
            sys.exit(1)
        return result

    print(f"  ERROR: {param_name} expects 1 value (fixed) or 3 values (start stop step), "
          f"got {len(values)}.")
    sys.exit(1)


def _validate_bounds(value, param_name, vmin, vmax):
    """Check that a value is within allowed bounds."""
    if value < vmin or value > vmax:
        print(f"  ERROR: {param_name} = {value} is out of bounds [{vmin}, {vmax}].")
        sys.exit(1)


def format_val(v):
    """Format a value for labels (strip trailing zeros from floats)."""
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_experiments(args):
    """Build experiment list from swept parameters (cartesian product)."""
    swept = {}  # param_name -> list of values
    fixed = {}  # param_name -> single value

    for param_name, (cli_flag, ptype, abbrev, vmin, vmax) in SWEEP_PARAMS.items():
        raw = getattr(args, param_name, None)
        if raw is None:
            continue
        values = parse_range(raw, param_name, ptype, vmin, vmax)
        if len(values) == 1:
            fixed[param_name] = values[0]
        else:
            swept[param_name] = values

    if not swept and not fixed:
        print("  ERROR: No hyperparameters specified. Provide at least one to sweep or fix.")
        print("  Example: python run_sweep.py --family ring --g-lr 0.001 0.01 0.003")
        sys.exit(1)

    # Build cartesian product of swept params
    if swept:
        sweep_names = list(swept.keys())
        sweep_values = [swept[k] for k in sweep_names]
        combos = list(itertools.product(*sweep_values))
    else:
        # Only fixed overrides, single experiment
        sweep_names = []
        combos = [()]

    experiments = []
    for combo in combos:
        # Merge swept values with fixed overrides
        params = dict(fixed)
        for name, val in zip(sweep_names, combo):
            params[name] = val

        # Build CLI flags and label
        cli_flags = []
        label_parts = []
        for name, val in sorted(params.items()):
            cli_flag = SWEEP_PARAMS[name][0]
            abbrev = SWEEP_PARAMS[name][2]
            cli_flags.extend([cli_flag, str(val)])
            label_parts.append(f"{abbrev}{format_val(val)}")

        label = "_".join(label_parts) if label_parts else "baseline"
        experiments.append((label, cli_flags))

    return experiments, swept


# ─────────────────────────────────────────────────────────────────────────────
# Execution helpers (unchanged logic from before)
# ─────────────────────────────────────────────────────────────────────────────

def build_batches(experiments, batch_size):
    """Split experiments into batches of batch_size."""
    return [experiments[i:i + batch_size]
            for i in range(0, len(experiments), batch_size)]


def find_log_dir(label, family, start_time):
    """Find the log directory created by a run (matches timestamp after start_time)."""
    logs_dir = Path("./logs")
    if not logs_dir.exists():
        return None

    candidates = []
    prefix = f"qgan_2d_{family}_"
    for d in logs_dir.iterdir():
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        rest = d.name[len(prefix):]
        ts_str = rest[:15]  # YYYYMMDD_HHMMSS
        try:
            ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        if ts >= start_time:
            candidates.append(d)

    if not candidates:
        return None

    # Match by HP suffix in dir name
    if label == "baseline":
        for c in candidates:
            rest = c.name[len(prefix):]
            if len(rest) == 15:
                return c
    else:
        # Check if all label parts appear in the dir name
        for c in candidates:
            if all(part in c.name for part in label.split("_")):
                return c

    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def extract_best_w1(log_dir):
    """Extract best validation Wasserstein distance from history.npz."""
    npz_path = Path(log_dir) / "history.npz"
    if not npz_path.exists():
        return None
    try:
        data = np.load(npz_path, allow_pickle=True)
        if 'val_wasserstein' in data:
            vals = data['val_wasserstein']
            valid = [v for v in vals if v > 0]
            return min(valid) if valid else None
    except Exception:
        return None
    return None


def run_batch(batch, batch_num, family, dry_run=False, silent=False):
    """Run a batch of experiments in parallel."""
    start_time = datetime.now().replace(microsecond=0)
    processes = []
    cwd = str(Path(__file__).parent)

    print(f"\n{'='*60}")
    print(f"  BATCH {batch_num}  ({len(batch)} experiments)")
    print(f"{'='*60}")

    for label, extra_flags in batch:
        train_args = ["--family", family] + extra_flags
        train_args_str = " ".join(train_args)

        if dry_run:
            print(f"  [{label:25s}]  python train_2d_qgan.py {train_args_str}")
            processes.append((label, None, None))
            continue

        # Force UTF-8 output to avoid cp1252 errors with Unicode chars
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        if silent:
            log_file = Path(f"./logs/sweep_{label}_{start_time:%Y%m%d_%H%M%S}.log")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "train_2d_qgan.py"] + train_args
            print(f"  Starting [{label:25s}]  (log: {log_file})")
            fh = open(log_file, "w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd, env=env,
            )
            processes.append((label, proc, fh, None, start_time))
        else:
            script_dir = Path("./logs/.sweep_scripts")
            script_dir.mkdir(parents=True, exist_ok=True)
            ps1_file = script_dir / f"{label}.ps1"
            transcript = str((script_dir / f"{label}_output.log").resolve())
            ps1_file.write_text("\n".join([
                f"$env:PYTHONIOENCODING = 'utf-8'",
                f"$env:TF_CPP_MIN_LOG_LEVEL = '3'",
                f"$env:TF_ENABLE_ONEDNN_OPTS = '0'",
                f"$env:PYTHONUNBUFFERED = '1'",
                f"$Host.UI.RawUI.WindowTitle = 'QGAN sweep: {label}'",
                f"Start-Transcript -Path '{transcript}' -Force",
                f"conda activate {CONDA_ENV}",
                f"python -u -W ignore train_2d_qgan.py {train_args_str}",
                f"Stop-Transcript",
                f"Write-Host ''",
                f"Write-Host '=== DONE: {label} ===' -ForegroundColor Green",
                f"Read-Host 'Press Enter to close'",
            ]), encoding="utf-8")

            print(f"  Starting [{label:25s}]  (window)")
            proc = subprocess.Popen(
                ["powershell", "-Command",
                 f"Start-Process powershell -ArgumentList "
                 f"'-ExecutionPolicy','Bypass','-File','{ps1_file.resolve()}'"],
                cwd=cwd, env=env,
            )
            processes.append((label, proc, None, None, start_time))

    return processes


def _check_run_done(label, family, start_time):
    """Check if a run is done by looking for its history.npz output file."""
    log_dir = find_log_dir(label, family, start_time)
    if log_dir is None:
        return False
    return (Path(log_dir) / "history.npz").exists()


def wait_for_batch(processes, family, silent=False):
    """Wait for all processes in a batch to complete. Returns results."""
    results = []
    if not processes or processes[0][1] is None:  # dry-run
        return [(label, None, None) for label, *_ in processes]

    print(f"\n  Waiting for {len(processes)} experiments to finish...")
    print(f"  (checking every 30s)\n")
    remaining = list(processes)

    while remaining:
        still_running = []
        for entry in remaining:
            label, proc, fh, _, start_time = entry

            if silent:
                done = proc.poll() is not None
            else:
                done = _check_run_done(label, family, start_time)

            if done:
                if fh and not fh.closed:
                    fh.close()
                elapsed = time.time() - start_time.timestamp()
                log_dir = find_log_dir(label, family, start_time)
                best_w1 = extract_best_w1(log_dir) if log_dir else None
                ret = proc.poll() if proc else 0
                status = "OK" if (ret is None or ret == 0) else f"FAIL(rc={ret})"
                w1_str = f"{best_w1:.4f}" if best_w1 is not None else "N/A"
                print(f"  Finished [{label:25s}]  {status}  "
                      f"Best W1={w1_str}  ({elapsed/60:.1f} min)")
                results.append((label, log_dir, best_w1, ret or 0))
            else:
                still_running.append(entry)

        remaining = still_running
        if remaining:
            time.sleep(30)

    return results


def print_summary(all_results):
    """Print a final summary table sorted by best W1."""
    print(f"\n{'='*70}")
    print(f"  SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Experiment':<30s} {'Best Val W1':>12s}  {'Status':>8s}  Log Dir")
    print(f"  {'-'*30} {'-'*12}  {'-'*8}  {'-'*30}")

    sorted_results = sorted(all_results,
                            key=lambda x: x[2] if x[2] is not None else 999)

    for label, log_dir, best_w1, retcode in sorted_results:
        w1_str = f"{best_w1:.4f}" if best_w1 is not None else "N/A"
        status = "OK" if retcode == 0 else f"FAIL({retcode})" if retcode else "N/A"
        dir_str = str(log_dir.name) if log_dir else "not found"
        print(f"  {label:<30s} {w1_str:>12s}  {status:>8s}  {dir_str}")

    if any(r[2] is not None for r in sorted_results):
        best = sorted_results[0]
        print(f"\n  Winner: {best[0]} with Best Val W1 = {best[2]:.4f}")
        print(f"  Log dir: {best[1]}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parallel HP sweep for 2D CV-QGAN",
        epilog=(
            "Range syntax (MATLAB-style):  param START STOP STEP\n"
            "  Example: --g-lr 0.001 0.01 0.003  ->  [0.001, 0.004, 0.007, 0.01]\n"
            "  Example: --n-layers 4 10 2        ->  [4, 6, 8, 10]\n"
            "  Single value = fixed override:  --g-lr 0.01\n\n"
            "Multiple swept params produce cartesian product of all combos."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Sweep control
    parser.add_argument("--family", type=str, default="ring",
                        choices=["gaussian", "ring", "correlated", "four_gaussians", 'vibronic'],
                        help="Distribution family (default: ring)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--parallel", type=int, default=6,
                        help="Max parallel runs per batch (default: 6)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Run only this batch number (1-indexed)")
    parser.add_argument("--silent", action="store_true",
                        help="No popup windows -- redirect output to log files")

    # Sweepable hyperparameters: each accepts 1 value (fixed) or 3 (start stop step)
    for param_name, (cli_flag, ptype, abbrev, vmin, vmax) in SWEEP_PARAMS.items():
        parser.add_argument(
            cli_flag, nargs="+", type=float if ptype is float else int,
            default=None, dest=param_name,
            help=f"[{vmin}, {vmax}] — single value or START STOP STEP",
        )

    args = parser.parse_args()

    # Generate experiments from ranges
    experiments, swept = generate_experiments(args)
    n_total = len(experiments)

    # Show sweep summary
    print(f"\nSweep configuration:")
    print(f"  Family: {args.family}")
    for name, values in swept.items():
        abbrev = SWEEP_PARAMS[name][2]
        print(f"  {name}: {len(values)} values  {[format_val(v) for v in values]}")

    print(f"\n  Total experiments: {n_total}")

    if n_total > MAX_EXPERIMENTS_WARN:
        print(f"\n  WARNING: {n_total} experiments will take a long time!")
        print(f"  Estimated wall time: ~{n_total / args.parallel * 20:.0f} min "
              f"({n_total / args.parallel:.0f} batches x ~20 min each)")
        resp = input("  Continue? [y/N]: ").strip().lower()
        if resp != "y":
            print("  Aborted.")
            return

    batches = build_batches(experiments, args.parallel)
    print(f"  Batches: {len(batches)} (up to {args.parallel} parallel)\n")

    if args.dry_run:
        print("  DRY RUN -- commands that would be executed:\n")
        for i, batch in enumerate(batches, 1):
            if args.batch and i != args.batch:
                continue
            run_batch(batch, i, args.family, dry_run=True)
        return

    # Register signal handler for clean shutdown
    child_processes = []

    def signal_handler(sig, frame):
        print("\n\nInterrupted! Killing all running experiments...")
        for _, proc, fh, *_ in child_processes:
            if proc and proc.poll() is None:
                proc.terminate()
            if fh and not fh.closed:
                fh.close()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)

    all_results = []
    for i, batch in enumerate(batches, 1):
        if args.batch and i != args.batch:
            continue
        procs = run_batch(batch, i, args.family, silent=args.silent)
        child_processes.extend(procs)
        results = wait_for_batch(procs, args.family, silent=args.silent)
        all_results.extend(results)

    print_summary(all_results)


if __name__ == "__main__":
    main()
