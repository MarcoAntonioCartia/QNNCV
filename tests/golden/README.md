# Golden regression suites

Behavior freezes for the 2D CV-QGAN. Every suite is a plain-assert script
(no pytest); `run_all.py` discovers `test_*.py` and runs each in a fresh
subprocess (avoids TF/Keras/SF global-state bleed), forcing UTF-8 output.

## Running

```powershell
C:\Users\mende\.conda\envs\qnncv\python.exe tests\golden\run_all.py   # all suites
C:\Users\mende\.conda\envs\qnncv\python.exe tests\golden\test_metrics.py   # one suite
```

Exit code 0 iff everything passes. Must run with the `qnncv` conda python
(hard error if TF is not importable). Suites load the code under test
through the root shim `train_2d_qgan.py` via `_harness.load_t2q()`, so they
stay valid across refactors of the internal module layout.

## What each suite pins

| Suite | Golden data | Freezes |
|---|---|---|
| test_signatures | `data/signatures.json` | exact parameter lists, kinds and defaults of the 13 public callables (formerly-divergent knobs are required keyword-only; CLI defaults are the single source of truth) |
| test_default_sync | — (structural) | every param shared between `build_parser()` and the internal signatures agrees with the CLI default or has none |
| test_sweep_sync | — (structural) | `run_sweep.SWEEP_PARAMS` ↔ CLI flag surface (types, bounds, folder-suffix abbreviations) |
| test_w1_failpath | — (structural) | `compute_wasserstein_2d` error path: injected ValueError → inf + logged warning; other exceptions propagate |
| test_hermite | `data/hermite.npz` | Hermite basis values + recommended cutoffs |
| test_readout | `data/readout.npz` | Fock ket → 2D density readout (sequential, batched, slice paths; with/without ancilla trace) |
| test_circuit_forward | `data/circuit_forward.npz` | full SF circuit forward pass, fixed weights and z, 2 and 3 modes, batched vs sequential |
| test_critic_input | `data/critic_input.npz` | peak normalization, blur kernel, gradient penalty values |
| test_metrics | `data/metrics.npz` | marginal-W1 (exact), energy distances, validate() outputs |
| test_families | `data/families.npz`, `families_params.json` | family densities + canonical parameters |
| test_blur_once | `data/blur_once.npz` | blur applied exactly once per critic tensor path (conv2d call count) + losses |
| test_e2e_gan_cli | `data/e2e_gan/` | real-CLI GAN run: history.npz, best_weights, `run_config.json` exact minus timestamp (pins the --deterministic prologue provenance), folder suffix, console markers |
| test_e2e_patha | `data/e2e_patha/` | pure-supervised (Path A) run: D identically inert, history, weights, fixed-z density |

Notes:
- `data/e2e_gan/console.txt` and `data/e2e_patha/stdout.txt` are *stored*
  but only console **markers** are asserted (e.g. `Seed: 0`, Path A
  banner) — the stored text may be cosmetically stale (e.g. legacy
  import chatter removed in the post-milestone cleanup).
- Several child runners pass former signature defaults explicitly
  (`d_lr=0.0002, latent_scale=1.0, …`); the numeric goldens were generated
  under those values, so do not "simplify" them away.

## Tolerances and reproducibility

- `assert_close`: rtol 1e-6, atol 1e-8; signatures and the W1 arrays are
  compared exactly.
- Goldens are **same-machine** freezes (Windows 11, CPU, conda `qnncv`,
  TF 2.18.0 / NumPy 2.0.2 / SF 0.23.0). Cross-machine or cross-OS
  bit-identity is NOT expected — TF kernel selection and BLAS differ.
  For scientific results, report seed-averaged distributions rather than
  single-run values.

## Regeneration policy

Regenerate with the suite's own generator:

```powershell
C:\Users\mende\.conda\envs\qnncv\python.exe tests\golden\test_<name>.py --generate --force
```

**Rule: a golden regeneration is a deliberate act.** Every regenerated
golden gets its own commit whose message states exactly which frozen fact
changed and why the new behavior is correct (see commit `e697e16` for the
template). Never bundle a regeneration with unrelated changes, and audit
the golden diff before committing.
