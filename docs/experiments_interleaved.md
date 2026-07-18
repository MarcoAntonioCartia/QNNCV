# Interleaved-squeeze encoding: A/B experiment protocol

Status: **pre-registered design — experiments NOT yet launched.**
Baseline milestone: tag `v0.2-baseline` (main @ `ae1bc29`); the 30-seed
baseline sweep is attributed to it.

## What is being tested

The baseline (`input_displacement`) injects the latent z only as a front
displacement, D(z) before the Killoran layers. The z-sweep diagnostic on
trained baselines is expected to show z exploiting mostly **rigid motions**
(translation/orientation) of the output density.

The `interleaved_squeeze` encoding keeps the front D(z) unchanged and
additionally modulates every injected layer's squeeze magnitude:

    r_{l,m} = w_r[l,m] + w_s[l,m] * tanh(z[2m])

- z[2m] is the same latent component that drives mode m's front-displacement
  magnitude; the latent space is **identical** in both arms (latent_dim =
  2·n_modes), so the A/B isolates encoding *placement*, not latent capacity.
- `w_s` (shape `(n_layers, n_modes)`) initializes to **zero**: at init the
  interleaved model is exactly the frozen baseline, and the baseline is a
  strict special case (w_s ≡ 0) of the new model.
- The symbolic circuit is unchanged (U1–S–U2–D–K per layer); the injection
  lives in the parameter binding only.

## Design

| Factor            | Setting                                                    |
|-------------------|------------------------------------------------------------|
| Config            | Frozen ring config from the baseline sweep — `--encoding` is the ONLY difference between arms |
| Seeds             | 0–29 (mirror the baseline sweep); 0–9 acceptable first pass |
| Arm A (control)   | `--encoding input_displacement` (or reuse the existing baseline runs at matching seeds) |
| Arm B (treatment) | `--encoding interleaved_squeeze`                           |

Two starts for Arm B:

1. **Cold**: fresh init (`w_s = 0` automatically).
2. **Warm**: `--init-weights logs/<baseline-run-seedN>/best_weights.npy`
   per seed — legacy v1 file loads with `w_s = 0`, i.e. a warm start from
   that seed's frozen baseline.

Example commands (fill in the frozen ring hyperparameters):

```bash
# Arm B cold, seed N
python train_2d_qgan.py --family ring <frozen-ring-flags> \
    --encoding interleaved_squeeze --seed N

# Arm B warm, seed N
python train_2d_qgan.py --family ring <frozen-ring-flags> \
    --encoding interleaved_squeeze --seed N \
    --init-weights logs/<baseline-ring-seedN>/best_weights.npy
```

Run folders are tagged `encIS`; `run_config.json` records
`encoding/inj_layers/r_max/init_weights/norm_floor` for every run.

## Metrics

Per run (from `history.npz` / final validation, as in the baseline sweep):

- **nearest-member W1** (checkpoint-selection metric),
- **diversity** vs the validation-set self-diversity reference,
- **nearED** (nearest-member energy distance).

Plus, on each best checkpoint:

- `scripts/z_sweep_diagnostic.py --run-dir logs/<run>` — descriptor
  response (centroid / orientation / widths / mean radius / radial width)
  vs z, for the before/after figure pair.
- `norm_monitor.npz` — per-sample ket-norm min/mean/max and warn counts;
  z-dependent squeezing makes truncation error z-dependent, so watch for
  warm-started runs driving `w_s` into the truncation-warning regime
  (mitigation if needed: `--r-max`, e.g. 0.3–0.5 — off by default).

## Pre-registered predictions

1. **Diversity rises** toward the validation-set self-diversity: z gains a
   non-rigid channel (width/photon content), so samples can differ in shape,
   not just placement.
2. **nearW1 improves on families whose intra-family variance is
   width/radius/spacing** (ring radius/width; correlated/four_gaussians
   spread) — the new channel matches exactly the variation the front
   encoding cannot express.
3. **Single-ring chirality lobing does NOT improve** — that failure is
   mode-count/readout physics, not encoding placement; predicting no change
   there guards against over-attributing any gains.
4. z-sweep after-figures show width/radius descriptors responding to z in
   Arm B where Arm A's respond only via rigid motions.

## Ablations noted for LATER (out of scope here)

- Dedicated inserted encoding squeezers between layers (Quandela-literal;
  a structural circuit change, unlike this bind-time modulation).
- z injected into interferometer phases.
- Per-layer subset injection (`--inj-layers`, already implemented) as a
  depth-of-injection ablation.
- A dedicated third latent component per mode for the squeeze channel
  (changes latent_dim, so it is NOT a clean A/B against the baseline).

## Bookkeeping rules

- Never write into the existing baseline/sweep run directories.
- `--encoding` is categorical and cannot be a `run_sweep.py` sweep axis
  (numeric-only SWEEP_PARAMS); pass it as a fixed passthrough per arm.
- Weight-file compatibility: v1 (raw array, baseline) loads into either
  encoding (`w_s = 0` warm start + notice); v2 (interleaved dict) into an
  `input_displacement` model is a hard error by design.
