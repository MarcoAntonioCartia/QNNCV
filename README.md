# QNNCV — 2D Continuous-Variable Quantum GAN

A continuous-variable quantum GAN (WGAN-GP) that learns 2D probability
densities with a CV quantum circuit generator (Strawberry Fields TF
backend) and a deliberately weak classical critic. Latent vectors are
encoded via displacement gates; the Fock-space ket is read out into a
40×40 position density through a Hermite-function basis, with optional
ancilla modes traced out.

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
run_sweep.py           family x seed sweep driver (invokes the CLI, resumable)
src/
  quantum/             hermite.py (Fock->position basis), circuit.py
                       (CVQGANGenerator: SF program, batching, readout)
  training/qgan_2d.py  training loop, WGANGP/PureSupervised trainers, seeding
  models/discriminators/qgan2d_discriminator.py   Discriminator2D (critic)
  metrics/             wasserstein.py, energy.py, validation.py
  families/            gaussian, ring, correlated, four_gaussians, vibronic
                       (+ registry.get_family)
  data/dataset.py      pre-generated train/val dataset
  critic_input/        peak normalization, blur, gradient penalty
  viz/plots.py         comparison plots + training dashboard
  models/, utils/, training/ (other files)   legacy 1D stack
scripts/               verify_batching / verify_tasks / verify_determinism
                       (+ _bootstrap.py: sys.path + simps alias)
tests/golden/          golden regression suites — see tests/golden/README.md
logs/                  training runs (one folder per run; baselines live here)
```

## Running

```powershell
# Single run (defaults in build_parser() are the single source of truth)
python train_2d_qgan.py --family ring --n-modes 3 --d-lr 0.0005 --gp-weight 10 --seed 0

# Sweep (family x seeds)
python run_sweep.py --family ring --seeds 30
```

Every run writes `run_config.json` (all args + versions + seed + env
provenance), `history.npz`, `best_weights.npy` and plots into
`./logs/qgan_2d_<family>_<timestamp><hp-suffix>/`, where the suffix encodes
non-default hyperparameters (e.g. `_m3_dlr0.0005_gp10_seed4`).

## Baselines

- **Reference point:** all baseline numbers correspond to the refactor
  milestone commit `6ab6591` (phase 0–9 behavior-preserving extraction;
  fully merged into `main` via PR #19). The post-milestone `cleanup` branch
  is verified behavior-identical by the golden suites.
- **Production ring configuration (frozen):** CLI defaults plus
  `--n-modes 3 --d-lr 0.0005 --gp-weight 10`; 30 seeds per family. Run
  folders in `logs/` follow `qgan_2d_<family>_<ts>_m3_dlr0.0005_gp10_seed<n>`.
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
```

Use the `qnncv` python for all of these. On consoles that pipe output,
set `PYTHONUTF8=1` (legacy compatibility prints contain ✓/✗ glyphs that
crash cp1252 pipes).

## Known issues / upcoming decisions

- **ħ convention.** The live 2D code never sets `hbar`; Strawberry Fields'
  default **ħ = 2** applies everywhere (engine creation in
  `src/quantum/circuit.py` passes no `hbar`; the Hermite readout in
  `src/quantum/hermite.py` uses σ = 1 units, consistent with ħ = 2
  conventions; the legacy 1D generators set `hbar = 2.0` explicitly).
  Survey target distributions assume **ħ = 1**, which differs by a √2
  position-space scale. This is documented — not fixed — because every
  golden and baseline was produced under ħ = 2. Decide the convention
  BEFORE generating datasets for new families.
- `interleaved_squeeze` encoding, MMD/energy trainers, and the noise-floor
  sweep are out of scope of the current cleanup and remain future work.
