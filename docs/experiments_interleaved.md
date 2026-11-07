# Interleaved-squeeze encoding: A/B experiment protocol

Status: **revised 7 Nov 2026 — fixed-scale design added and run the same
evening (Arms A and B at scales 0.05/0.1/0.2), then the dedicated-latent
variant (Arm B′ at 0.1/0.2); results under "Findings so far". Arms C/D
(learned gain) NOT yet run at the 300-epoch budget.** A first cold pass of
the original learned-gain design was run on 18 Jul 2026 (ring, seeds 0–4,
100 and 200 epochs); it motivated the fixed-scale revision.
Baseline milestone: tag `v0.2-baseline` (main @ `ae1bc29`); the 30-seed
baseline sweep is attributed to it.

## What is being tested

The baseline (`input_displacement`) injects the latent z only as a front
displacement, D(z) before the Killoran layers: z can move and rotate the
output density but cannot change its shape. The z-sweep diagnostic on
trained baselines shows z exploiting mostly **rigid motions**.

The `interleaved_squeeze` encoding keeps the front D(z) unchanged and
additionally modulates every injected layer's squeeze magnitude — the knob
that controls feature *width*:

    r_{l,m} = w_r[l,m] + w_s[l,m] * tanh(z[2m])

- z[2m] is the same latent component that drives mode m's front-displacement
  magnitude; the latent space is **identical** in both arms (latent_dim =
  2·n_modes), so the A/B isolates encoding *placement*, not latent capacity.
- `w_s` (shape `(n_layers, n_modes)`) is a **gain filled with
  `--squeeze-scale` (default 0)**. It is **fixed (non-trainable)** unless
  `--train-squeeze-scale` is given, in which case it is trainable and
  initialized at `--squeeze-scale`. At w_s ≡ 0 the model is exactly the
  frozen baseline; the baseline is a strict special case of the new model
  and the same 186 trainable parameters are used in the fixed-scale arms.
- The symbolic circuit is unchanged (U1–S–U2–D–K per layer); the injection
  lives in the parameter binding only.

### Why a fixed scale is the primary design

The learned-gain design (`--train-squeeze-scale` with scale 0, the only
design before 7 Nov) can only open the width channel if the gradient on
`w_s` is consistently signed, and that gradient is the batch correlation
between "this sample wanted more squeezing" and the sign of its z. The
critic rewards z-dependent width only once the generator produces it, and
the generator produces it only once the gain is open. At `w_s ≈ 0` the
gradient therefore averages to ~0 and Adam random-walks; the 18 Jul runs
confirm it (gain stayed ≈ 0.02–0.05, see below). With one generator update
per epoch ("epoch" = n_critic critic steps + 1 generator step, batch 8),
training length alone does not fix this. A fixed scale forces the channel
on so the critic can grade it; the learned gain stays as an ablation.

## Design

| Factor            | Setting                                                    |
|-------------------|------------------------------------------------------------|
| Config            | Frozen ring config `--n-modes 3 --d-lr 0.0005 --gp-weight 10` + CLI defaults; `--encoding` / `--squeeze-scale` / `--train-squeeze-scale` are the ONLY differences between arms |
| Budget            | **One budget for every arm** (the 18 Jul ep100 vs ep200 pair was confounded). First pass 300 epochs; confirmation 500 (the budget the other families got in the baseline sweep) |
| Seeds             | 0–4 first pass; 0–9 confirmation; 0–29 only if the effect is real |
| Arm A (control)   | `--encoding input_displacement` — or `--encoding interleaved_squeeze` with the defaults (scale 0, fixed), which is numerically identical and shares the folder tag `encIS` |
| Arm B (primary)   | `--encoding interleaved_squeeze --squeeze-scale s --r-max 0.5`, s ∈ {0.1, 0.2, 0.3, 0.4} (range 0.1–0.4 step 0.1): from "comparable to" to "dominating" the baseline's learned squeeze magnitudes (median \|w_r\| = 0.145, IQR 0.075–0.219 over the 30-seed ring sweep) |
| Arm B′ (decoupled) | Arm B plus `--inj-latent dedicated`: the squeeze channel of mode m reads its own latent component z[2·n_modes + m] instead of the door's z[2m], so size and position are separate dials (latent_dim 9 for 3 modes; still 186 trainable parameters). Motivated by the z-sweep on Arm B, where radius and centroid moved together |
| Arm C (ablation)  | `--encoding interleaved_squeeze --train-squeeze-scale` (learned gain, cold start at 0): does the model open the channel by itself? |
| Arm D (ablation)  | Arm C warm-started from the matching baseline seed: `--init-weights logs/<baseline-ring-seedN>/best_weights.npy` (v1 file loads with w_s = 0). The sharpest critic the learned gain can get |

Run folders are tagged `encIS`, `ss<s>` and `sstrain`; `run_config.json`
records `encoding/inj_layers/r_max/squeeze_scale/train_squeeze_scale/
init_weights/norm_floor` for every run. `best_weights.npy` (v2) stores
`w_s` for every interleaved run, fixed or trained.

### Commands

`run_sweep.py` sweeps the numeric axes (`--squeeze-scale`, `--seed`, …)
and forwards the categorical/boolean flags unchanged to every run
(passthrough). Each command below is one arm; run the baseline arm only
for seeds/budgets the 30-seed sweep does not already cover.

```bash
# Arm B: fixed scale, 4 values (0.1..0.4) x 5 seeds = 20 runs, 300 epochs
python run_sweep.py --family ring --silent --seed 0 4 1 --squeeze-scale 0.1 0.4 0.1 \
    --n-modes 3 --d-lr 0.0005 --gp-weight 10 --epochs 300 \
    --encoding interleaved_squeeze --r-max 0.5

# Arm A at the same budget (seeds 0-4, 300 epochs)
python run_sweep.py --family ring --silent --seed 0 4 1 \
    --n-modes 3 --d-lr 0.0005 --gp-weight 10 --epochs 300

# Arm B': decoupled (dedicated latent components), scales 0.1 and 0.2
python run_sweep.py --family ring --silent --seed 0 4 1 --squeeze-scale 0.1 0.2 0.1 \
    --n-modes 3 --d-lr 0.0005 --gp-weight 10 --epochs 300 \
    --encoding interleaved_squeeze --inj-latent dedicated

# Arm C: learned gain, cold
python run_sweep.py --family ring --silent --seed 0 4 1 \
    --n-modes 3 --d-lr 0.0005 --gp-weight 10 --epochs 300 \
    --encoding interleaved_squeeze --train-squeeze-scale --r-max 0.5

# Arm D: learned gain, warm start (one command per seed N)
python train_2d_qgan.py --family ring --n-modes 3 --d-lr 0.0005 --gp-weight 10 \
    --epochs 300 --seed N --encoding interleaved_squeeze --train-squeeze-scale \
    --r-max 0.5 --init-weights logs/<baseline-ring-seedN>/best_weights.npy
```

Cost on the GPU PC (CPU-bound, ~5 s/epoch for this config): 300 epochs ≈
25 min per run; arms A+B+C at 5 seeds = 30 runs ≈ 12.5 h sequential,
≈ 2–2.5 h with 6 in parallel (`--parallel 6`, the default; the July sweeps
ran 6-way at the same per-run speed).

## Metrics

Per run (from `history.npz` / final validation, as in the baseline sweep):

- **nearest-member W1** (checkpoint-selection metric),
- **diversity** vs the validation-set self-diversity reference,
- **nearED** (nearest-member energy distance).

`scripts/analyze_sweep.py` produces the per-arm tables and the paired-by-
seed comparison (`--compare A B`, Wilcoxon p, budget-mismatch warning).

Plus, on each best checkpoint:

- `scripts/z_sweep_diagnostic.py --run-dir logs/<run>` — descriptor
  response (centroid / orientation / widths / mean radius / radial width)
  vs z, for the before/after figure pair.
- `norm_monitor.npz` — per-sample ket-norm min/mean/max and warn counts;
  z-dependent squeezing makes truncation error z-dependent (mitigation:
  `--r-max`, on by default in the arms above; `--norm-floor` for the warn
  threshold).
- the learned/fixed `w_s` in `best_weights.npy` (v2) — for Arms C/D, how
  far the gain moved from its init is the first thing to read.

**Reading rule for Arm B:** diversity rising is expected *by construction*
(the channel is forced on), so it is not evidence on its own. The claim
rests on nearest-member W1 / nearED improving (the width variation matches
the family's) and on the z-sweep diagnostic showing a controlled, monotone
width response to z. A large s that raises diversity but worsens nearW1 is
a *failure* of that scale, not a success.

## Pre-registered predictions

1. **Arm B at s ≈ 0.1–0.2 improves nearW1 and nearED on ring** (the family's
   intra-family variance is radius/width, the variation the front encoding
   cannot express) with diversity closer to the validation self-diversity.
2. **s = 0.4 degrades**: more truncation warnings and worse nearW1 than
   s = 0.2 (the modulation dominates the learned squeezes).
3. **Arm C does not open the channel** (|w_s| stays ≲ 0.05) and matches the
   baseline within seed noise; **Arm D opens it more than C** but it is an
   open question whether it reaches Arm B.
4. **Single-ring chirality lobing does NOT improve** in any arm — that
   failure is mode-count/readout physics, not encoding placement.

Outcome (7 Nov 2026, 300 epochs, 5 seeds): 1 not supported (shared or
dedicated); 2 holds for truncation, and nearW1 is worst at s = 0.2 in both
variants; 3 and the chirality claim not yet tested at this budget (Arms C/D
pending).

## Findings so far (18 Jul 2026 cold pass, learned gain, from `scripts/analyze_sweep.py`)

| arm (ring, seeds 0–4)         | best nearW1 median | diversity | nearED | learned \|w_s\| (median) |
|-------------------------------|-------------------:|----------:|-------:|-------------------------:|
| baseline, 100 epochs          | 0.894              | 1.58      | 0.021  | —                        |
| learned gain, 100 epochs      | 0.865 (2/5 better) | 1.60      | 0.024  | 0.02–0.05                |
| learned gain, 200 epochs      | 0.720 (5/5 better) | 1.33      | 0.017  | 0.02–0.04                |

- At matched budget: no detectable effect (Wilcoxon p = 0.63 on nearW1).
- The 200-epoch gain is confounded by training length (no 200-epoch
  baseline arm; the 500-epoch ring baseline reaches ≈ 0.70 without the
  channel) and those runs entered the truncation regime (per-sample
  ket-norm min 0.70–0.89, 124–601 warnings per run, `--r-max` off).
- The gain modulated the squeezes by ≈ 10 % of their size: the channel never
  opened, so the hypothesis was not tested. Hence the fixed-scale design.

### 7 Nov 2026 evening: fixed scale, shared and dedicated latent (ring, 300 epochs, seeds 0–4, no r_max)

| arm                          | best nearW1 median [IQR] | vs baseline per seed (nearW1) | nearED | per-sample ket-norm min (median) | best epoch (median) |
|------------------------------|-------------------------:|-------------------------------|-------:|---------------------------------:|--------------------:|
| A baseline                   | **0.673** [0.635, 0.704] | —                             | 0.020  | 0.856                            | 160                 |
| B shared, s = 0.05           | 0.750 [0.638, 0.756]     | worse 3/5 (within noise)      | 0.018  | 0.854                            | 200                 |
| B shared, s = 0.1            | 0.715 [0.695, 0.788]     | worse 4/5, median Δ +0.04     | 0.018  | 0.802                            | 200                 |
| B shared, s = 0.2            | 0.831 [0.810, 0.851]     | worse 4/5, median Δ +0.15     | 0.020  | 0.758                            | 240                 |
| B′ dedicated, s = 0.1        | 0.779 [0.708, 0.814]     | worse 4/5, median Δ +0.11     | 0.021  | 0.853                            | 240                 |
| B′ dedicated, s = 0.2        | 0.786 [0.779, 0.847]     | worse 5/5, median Δ +0.14     | 0.020  | 0.789                            | 200                 |

Paired tables: `analysis/sweep/compare_*.md` (from `scripts/analyze_sweep.py`).
Validation noise for one checkpoint, re-validated with fresh z at the 10
samples the trainer uses: nearW1 sd ≈ 0.08 (0.04 at 40 samples); the
best-checkpoint value is optimistic by ≈ 0.03 (minimum of 15 noisy readings).

- **No tested variant beats the baseline.** Prediction 1 is not supported at
  any scale, shared or dedicated. nearED is flat (0.018–0.021) everywhere.
- Truncation grows with the scale in both variants (prediction 2 holds);
  the treatment arms reach their best checkpoint later than the baseline.
- z-sweep diagnostic (`analysis/zsweep/`, seed-0 checkpoints): in the
  baseline the radius depends only on |z| (Spearman vs |t| ≈ −1: size is a
  side effect of displacement). Shared s = 0.2 makes the radius monotone in
  the door component (Spearman −0.89, range 0.78–1.26) while the same
  component shifts the centroid by 0.70. Dedicated s = 0.2 separates the
  dials exactly as designed: the dedicated components move the radius
  monotonically (Spearman −1.00, 1.07–1.43) with centroid shift ≤ 0.03,
  while the door components keep moving position without resizing. **The
  mechanism is right and the fidelity still drops**, so the coupling was not
  the whole explanation.
- Open readings, not conclusions: (i) budget — 4 of 10 dedicated runs peak
  at epoch 240–300 and may still be improving; (ii) the forced variation is
  a harder matching problem for the critic within 300 generator steps;
  (iii) the ket penalty fights the extra squeezing (ket-norm minima 0.73–0.89
  at s = 0.2). The validation metric (10 samples, marginal W1) is also
  noisier than the effects at s ≤ 0.1.
- Baseline at 300 epochs ≈ the July 500-epoch runs (≈ 0.70): the 100-epoch
  ring sweep was under-trained relative to the other families.

## Ablations noted for LATER (out of scope here)

- Dedicated inserted encoding squeezers between layers (Quandela-literal;
  a structural circuit change, unlike this bind-time modulation).
- z injected into interferometer phases.
- Per-layer subset injection (`--inj-layers`, already implemented) as a
  depth-of-injection ablation.
- A dedicated third latent component per mode for the squeeze channel
  (changes latent_dim, so it is NOT a clean A/B against the baseline).
- A per-layer or per-mode fixed scale (currently one scalar fills `w_s`).

## Bookkeeping rules

- Never write into the existing baseline/sweep run directories.
- `--encoding` and `--train-squeeze-scale` are categorical/boolean and
  cannot be `run_sweep.py` sweep axes; they are passthrough flags applied to
  every run of a command. `--squeeze-scale` IS a sweep axis.
- Weight-file compatibility: v1 (raw array, baseline) loads into either
  encoding (w_s stays at its configured init + notice); v2 (interleaved
  dict) into an `input_displacement` model is a hard error by design; v2
  into a FIXED-scale model loads only if its `w_s` equals the configured
  scale (a fixed scale is never silently overwritten).
- Analysis outputs go to `analysis/` (csv/png gitignored).
