# QNNCV — 2D Continuous-Variable Quantum GAN

A continuous-variable quantum GAN (WGAN-GP) that learns 2D probability
densities with a CV quantum circuit generator (Strawberry Fields TF
backend) and a deliberately weak classical critic. Latent vectors enter as
front displacement gates (optionally also as a z-dependent squeeze
modulation inside every layer, `--encoding interleaved_squeeze`); the
Fock-space ket is read out into a 40×40 position density through a
Hermite-function basis (ħ = 1 units), with optional ancilla modes traced
out. For the thesis benchmark the generator and the critic are swappable
for discrete-variable (PennyLane) and classical backends
(`docs/backends.md`).

The repository also contains an older 1D Killoran-style CV-QNN codebase
(`src/models/generators`, `src/training/trainer.py`, `qnncv.py`, …). It is
legacy: kept importable but no longer part of the active pipeline.

## Environment

Everything below is the environment all baselines and golden tests were
produced on. Cross-machine bit-identity is *not* expected (see
`tests/golden/README.md`).

| Component | Version |
|---|---|
| OS | Windows 11 (PowerShell) |
| Python | 3.11.14 (conda env `qnncv`) |
| TensorFlow | 2.18.0 |
| NumPy | 2.0.2 |
| SciPy | 1.15.3 |
| Strawberry Fields | 0.23.0 |

```powershell
conda activate qnncv           # C:\Users\<user>\.conda\envs\qnncv
pip install -r requirements.txt   # exact pins recorded from this env
```

Note: SF 0.23.0 uses `scipy.integrate.simps`, removed in SciPy 1.14+. The
entry point (`train_2d_qgan.py`), `scripts/_bootstrap.py`, and
`src/__init__.py` each alias `simps = simpson` before SF is imported — keep
that ordering if you add new entry points.

## Layout

```
train_2d_qgan.py       CLI entry point + prologue (simps alias, --deterministic
                       env scan BEFORE the TF import) + thin re-export shim
                       that the golden tests load via importlib
run_sweep.py           numeric sweeps over the CLI (START STOP STEP per axis,
                       cartesian product, resumable); every other flag passes
                       through unchanged to each run
src/
  quantum/             hermite.py (Fock->position basis), circuit.py
                       (CVQGANGenerator: SF program, batching, readout;
                       EncodingSpec: input_displacement | interleaved_squeeze),
                       weights_io.py (best_weights.npy v1 / v2 contract)
  backends/            benchmark generators (cv, classical, dv) and critics
                       (dense_2d, coord_2d, cv_qnn_2d, dv_qnn_2d) behind one
                       contract — see docs/backends.md
  training/qgan_2d.py  training loop, WGANGP/PureSupervised trainers, seeding
  models/discriminators/qgan2d_discriminator.py   Discriminator2D (critic)
  metrics/             wasserstein.py, energy.py, validation.py
  families/            gaussian, ring, correlated, four_gaussians, vibronic
                       (+ registry.get_family)
  data/dataset.py      pre-generated train/val dataset
  critic_input/        peak normalization, blur, gradient penalty
  viz/plots.py         comparison plots + training dashboard
  models/, utils/, training/ (other files)   legacy 1D stack
scripts/               verify_batching / verify_tasks / verify_determinism;
                       analyze_sweep (per-arm tables, paired-by-seed compare),
                       rescore + summarize_rescore (200-sample re-scoring),
                       z_sweep_diagnostic (density response to z),
                       run_matrix (job lists N at a time), smoke_backend
                       (+ _bootstrap.py: sys.path + simps alias)
tests/golden/          golden regression suites — see tests/golden/README.md
docs/                  experiments_interleaved.md (encoding A/B protocol and
                       findings), benchmark_plan.md, backends.md,
                       references_benchmark.md; KEY_LEARNINGS.md and
                       QGAN_TRAINING_ANALYSIS_20260123.md are earlier notes
logs/                  training runs (one folder per run; baselines live here)
analysis/              script outputs (untracked, regenerable)
```

## Running

```powershell
# Single run (defaults in build_parser() are the single source of truth)
python train_2d_qgan.py --family ring --n-modes 3 --d-lr 0.0005 --gp-weight 10 --seed 0

# Sweep: each numeric axis takes one value (fixed) or START STOP STEP;
# any other flag is forwarded unchanged to every run
python run_sweep.py --family ring --silent --seed 0 29 1 --n-modes 3 --d-lr 0.0005 --gp-weight 10

# Interleaved-squeeze encoding (default-off: scale 0 is exactly the baseline)
python train_2d_qgan.py --family ring --n-modes 3 --d-lr 0.0005 --gp-weight 10 --seed 0 --encoding interleaved_squeeze --squeeze-scale 0.1 --inj-latent dedicated

# Benchmark backends (docs/backends.md)
python train_2d_qgan.py --family ring --generator classical --critic coord_2d --epochs 300 --seed 0
```

Every run writes `run_config.json` (all args + versions + seed + env
provenance), `history.npz`, `norm_monitor.npz` (per-sample ket norms),
`backend_info.json` (parameter and Hilbert-space sizes, device, wall time),
`best_weights.npy` and plots into
`./logs/qgan_2d_<family>_<timestamp><hp-suffix>/`, where the suffix encodes
non-default hyperparameters (e.g. `_m3_dlr0.0005_gp10_seed4`,
`_encIS_ss0.1_zlded`; `_warm` marks `--init-weights`). Folders are claimed
atomically, so parallel launches never share one.

## Baselines

- **Reference point:** all baseline numbers correspond to the refactor
  milestone commit `6ab6591` (phase 0–9 behavior-preserving extraction;
  fully merged into `main` via PR #19). The post-milestone `cleanup` branch
  is verified behavior-identical by the golden suites.
- **Milestone tag:** `v0.2-baseline` (`main` @ `ae1bc29`, 18 Jul 2026); the
  30-seed per-family sweep is attributed to it.
- **Production ring configuration (frozen):** CLI defaults plus
  `--n-modes 3 --d-lr 0.0005 --gp-weight 10`; 30 seeds per family. Run
  folders in `logs/` follow `qgan_2d_<family>_<ts>_m3_dlr0.0005_gp10_seed<n>`.
  At 300 epochs the same model's 5-seed median nearest-member W1 on ring
  came out 0.67 and 0.81 in two realizations that differ only in how the
  latent stream is consumed — the spread to keep in mind before reading any
  single-arm number (`docs/experiments_interleaved.md`).
- **Regression anchor:** the tiny frozen ring config pinned end-to-end by
  `tests/golden/test_e2e_gan_cli.py` (its `CLI_ARGS`, plus the golden
  `tests/golden/data/e2e_gan/run_config.json`).
- Report seed-averaged distributions, not single runs.

## Verification

```powershell
python tests\golden\run_all.py        # all golden suites, fresh subprocess each
python scripts\verify_batching.py     # batched==sequential, gradients, blur
python scripts\verify_tasks.py        # GAN-mode critic health, Path A inertness
python scripts\verify_determinism.py  # same-seed reproducibility + seed round-trip
python scripts\analyze_sweep.py --since 20261107   # read-only: per-arm tables into analysis\sweep\
```

Use the `qnncv` python for all of these. On consoles that pipe output,
set `PYTHONUTF8=1` (legacy compatibility prints contain ✓/✗ glyphs that
crash cp1252 pipes).

## Known issues / upcoming decisions

- **ħ convention — settled: the output is in ħ = 1 units.** The live 2D
  code never passes `hbar` to Strawberry Fields and never calls an SF
  quadrature or Wigner routine, so the engine default (ħ = 2) touches
  nothing: a Fock ket is convention-free, and the position scale is set
  entirely by the readout basis in `src/quantum/hermite.py`, whose vacuum
  is φ₀(x) ∝ e^(−x²/2) — variance 1/2, i.e. x̂ = (â + â†)/√2, the ħ = 1
  convention. Survey targets written in ħ = 1 units are therefore on the
  readout's own scale; the √2 mismatch this entry used to warn about does
  not exist. Only the legacy 1-D stack sets `hbar = 2.0`, and it is out of
  the pipeline. State ħ = 1 in the thesis and keep new families in those
  units.
- `interleaved_squeeze` is implemented (default-off) and its A/B against the
  baseline is in `docs/experiments_interleaved.md`: the width channel works
  mechanically, but at 5 seeds × 300 epochs no scale beats the baseline and
  none is shown worse than its control. The CV / DV / classical benchmark
  backends and their plan are in `docs/backends.md` and
  `docs/benchmark_plan.md`; one grid for every backend (32×32 vs 64×64) is
  still open. MMD/energy trainers and the noise-floor sweep remain future
  work.
