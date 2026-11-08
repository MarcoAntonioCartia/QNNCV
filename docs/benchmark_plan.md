# Benchmark framework plan: CV vs DV vs classical, pieces first, plumbing last

Status: 8 Nov 2026 — framework built (Phases 0–2), ring baselines tuned and confirmed at 20 seeds (Phase 3, first result below). Objective 2 of the thesis: compare the CV-QGAN
against a discrete-variable QGAN and purely classical models "para encontrar
en cuáles casos el modelo propuesto es mejor, peor o igual".

## Where the repo stands

| piece | status | evidence |
|---|---|---|
| CV generator (Strawberry Fields) | done | `src/quantum/circuit.py` (`CVQGANGenerator`, 186 params at the frozen config, `EncodingSpec` seam) |
| classical critic (Keras MLP) | done, the only critic | `src/models/discriminators/qgan2d_discriminator.py` (`Discriminator2D`, 25,809 params); `CRITIC_REGISTRY = {'dense_2d'}` |
| classical generator | not started | — |
| DV generator (PennyLane) | not started | a June-2025 chat draft never entered the repo; PennyLane is not installed in `qnncv` |
| DV critic (PennyLane) | not started | — |
| CV quantum critic (SF) | spec only | 12 Jul 2026 chat: point-wise `D(x,y)`, 16 params/layer, 96 at 6 layers |
| legacy 1-D models | reference only | `src/models/{generators,discriminators}/*` are the pre-2D sample-based stack (`quantum_sf_discriminator.py` is a usable sketch for the CV critic) |

The trainer (`src/training/qgan_2d.py`) already reaches the generator only
through `latent_dim`, `batch_size`, `generate_batch`,
`generate_distribution_2d`, `trainable_variables`, `weights`. Its three
remaining ties to Strawberry Fields: it constructs `CVQGANGenerator`
directly, it always adds the ket-norm penalty, and checkpoints assume
`weights` is one tensor. Everything else (families, dataset, metrics,
critic input, plots) is backend-free.

## Principle: add the pieces, wire them last

Every backend is built and tested on its own, against one written contract,
with a standalone smoke script, before `train_2d_qgan.py` learns about it.
The trainer and the goldens change once, in the final plumbing phase. Until
then `main` behaviour (CV generator + `dense_2d` critic) stays byte-identical
and every golden stays green.

## Phase 0 — contracts and registry (no trainer change)

New package `src/backends/`:

- `base.py`: `GeneratorBackend` — `latent_dim`, `batch_size`,
  `generate_batch(z, xvec, yvec) -> (prob[B,G,G], aux)`,
  `generate_distribution_2d(z, xvec, yvec)`, `trainable_variables`,
  `regularizer(aux) -> scalar` (CV: ket-norm penalty; others: 0),
  `state_dict() / load_state_dict()`, `describe() -> {backend, n_params,
  hilbert_dim, latent_dim, grid}`. `CriticBackend` — `__call__(x) ->
  (B,1)`, `trainable_variables`, `describe()`.
- `registry.py`: `GENERATOR_REGISTRY`, `CRITIC_REGISTRY` (the existing one
  moves here, `dense_2d` kept).
- `cv.py`: adapter around `CVQGANGenerator`; `state_dict` = the v1/v2
  `weights_io` payload, so existing `best_weights.npy` files load unchanged.
- `tests/golden/test_backend_contract.py`: every registered backend at toy
  size: shapes, dtype, probabilities sum to 1, gradients reach every
  trainable variable, `describe()` keys, state round-trip exact.

Done when: the CV adapter reproduces `test_circuit_forward` numbers exactly
and the contract suite is green. No golden regenerated.

## Phase 1 — the pieces (independent, one PR each)

| piece | design | fairness rule | effort |
|---|---|---|---|
| **1a classical generator** | Keras coordinate network `f(x, y, z)` evaluated on the grid, softplus + normalize; two sizes: parameter-matched (≈186) and a large reference | same grid, same critic, same latent dim as CV; parameters reported | 1 day |
| **1b DV generator** | PennyLane `lightning.qubit`, TF interface; `n_qubits = 2·log2(G)` (32×32 → 10 qubits); angle encoding of z, fixed, never trainable (Jan 2026 rule); RY/RZ + entangling ring layers; Born probabilities reshaped to the grid | report parameters AND Hilbert-space size (CV: cutoff^modes = 1728 at cutoff 12; DV: 2^n) | spike 1 day, build 1 week |
| **1c CV quantum critic** | point-wise Killoran circuit `D(x, y) -> R` on sampled grid points, `fake_score = (G²/|S|) Σ P_fake(s) D(s)` (12 Jul spec); 16 params/layer | the 96-param parity configuration is the headline | 3 days |
| **1d DV critic** | the PennyLane twin of 1c (same point-wise contract) | same point-sampling, parameter count reported | 2 days after 1b |
| **1e classical critic** | exists (`dense_2d`); add a point-wise coordinate critic so the critic family is also comparable across backends | — | 0.5 day |
| **1f common evaluation** | `scripts/rescore.py`: 200 fixed-seed samples per checkpoint; fidelity + coverage in W1 and energy distance; MMD two-sample test vs the validation set; self-diversity reference; writes `analysis/rescore/` | one grid for every backend (decide 32×32 vs 64×64; CV re-rendered from checkpoints); `--val-samples` raised | 2 days |

Each piece ships with: its contract test, a `scripts/smoke_<piece>.py` that
trains it for a few epochs against a frozen critic without touching
`qgan_2d.py`, and a `docs/backends/<piece>.md` note (design, parameter
count, known limits).

Environment rule: PennyLane goes into a cloned env (`qnncv-dv`) first; the
`qnncv` env that produced the goldens is not touched until the DV backend's
TF-2.18 compatibility is confirmed.

## Phase 2 — plumbing (last)

- `train_2d_qgan.py`: `--generator {cv,classical,dv}`, `--critic
  {dense_2d,coord_2d,cv_qnn_2d,dv_qnn_2d}`, `--grid-size` honoured by all
  backends; folder tags `g<..>`/`c<..>`; `run_config.json` adds backend
  names, `n_params`, `hilbert_dim`, device and wall time.
- `qgan_2d.py`: build through the registry; `g_loss += generator.regularizer(aux)`;
  checkpoints through `state_dict`.
- Goldens: the default path must stay byte-identical; only `run_config.json`
  gains keys (one audited regen commit). Sync tests (`test_default_sync`,
  `test_signatures`, `test_sweep_sync`) learn the flags.
- `run_sweep.py` passthrough already covers categorical flags;
  `analyze_sweep.py` groups by the new tags for free.

## Phase 3 — experiments

### Pilot result (8 Nov 2026): ring, 32×32, 300 epochs, seeds 0–4, `dense_2d` critic, CV hyper-parameters for every arm

Re-scored with `scripts/rescore.py` (200 fixed-seed samples; W1 in 32×32 grid
cells, not comparable to the 40×40 numbers; validation self-diversity 1.92).

| arm | params | hilbert | fidelity W1 median (range) | coverage W1 median | diversity | fid ED / cov ED | wall / run |
|---|---|---|---|---|---|---|---|
| cv (frozen config) | 186 | 1728 | **0.66** (0.61–0.69) | **0.82** (0.75–0.99) | 1.3 (under) | 0.023 / 0.044 | 22 min |
| classical coord net (9, 9) | 181 | — | 1.57 (1.30–3.83) | 0.97 (0.94–1.12) | 2.0–6.0 (over; seeds 2, 4 diverged) | 0.095 / 0.037 | 2 min |
| dv 10 qubits × 9 layers | 180 | 1024 | 2.08 (1.93–2.17) | 3.0 (2.8–3.2) | **0.26** (collapsed: z ignored) | 0.10 / 0.24 | 45 min |

Every arm is distinguishable from the real family (MMD p = 0.005). Reading:
the CV generator leads at matched parameter count, but the two baselines ran
with the CV's hyper-parameters and zero tuning of their own, which Bowles et
al. (2024) rightly call unfair in the other direction. Before any claim:
(a) classical — lower learning rate or SIREN/Fourier features (two of five
seeds diverged); (b) dv — the cyclic RY angle encoding with 9 layers
collapses to one output regardless of z, and the output itself is the
speckle a hardware-efficient 10-qubit Born distribution produces (1024
near-random amplitudes, `final_samples.png`), nothing like a smooth ring;
Zoufal et al. used 3 qubits (8 bins) per distribution for this reason. Try a
coarser DV grid (16×16 = 8 qubits, CV re-rendered to match), data
re-uploading of z between layers, fewer layers, a stronger encoding (RY+RZ)
and its own learning rate;
(c) then 20 seeds or paired restarts per the statistical-design rules.

### Tuned baselines (8 Nov 2026): the ring, 300 epochs, `dense_2d` critic

Everything below is the re-score (`scripts/rescore.py`, 200 fixed-seed
samples per checkpoint, W1 in cells of the grid named in the column);
tables produced by `scripts/summarize_rescore.py`; job lists under
`analysis/matrix/`. The reliability criterion was fixed in that script
before any ladder result was read: every seed within x2 of the validation
self-diversity and the arm's median fidelity within x1.5 of the best arm on
the same grid.

**Search space and rule.** Each baseline was tuned on its own knobs with
the CV arm frozen at its July configuration (never re-tuned): the
comparison reads "CV as frozen vs baselines at best effort", the
conservative direction. The WGAN-GP penalty weight turned out to be a
baseline knob too (the CV's 10 makes the classical net diverge on some
seeds; 5 does not), so each baseline runs at its own penalty and this is
stated with every number.

**Classical** (`analysis/matrix/classical_sweep{,2,3}.txt`, 85 runs, 5 seeds
each; 32x32):

| arm (181 params unless noted) | g_lr | gp | fidelity W1 med (range) | coverage | div / real |
|---|---|---|---|---|---|
| tanh (9, 9) — pilot settings | 0.005 | 10 | 1.56 (1.30-3.83) | 0.97 | 1.47 (2 seeds diverged) |
| tanh | 0.005 | 5 | 1.17 (0.68-1.48) | 0.84 | 1.41 |
| tanh | 0.02 | 5 | 0.56 (0.50-0.81) | 0.69 | 0.78 |
| **tanh** | **0.05** | **5** | **0.52 (0.41-0.62)** | **0.58** | 0.79 |
| tanh | 0.1 | 5 | 0.59 (0.47-0.62) | 0.61 | 0.77 |
| tanh | 0.2 | 5 | 0.63 (0.50-0.72) | 0.74 | 0.75 |
| tanh | 0.02 | 10 | 0.71 (0.61-2.34) | 0.82 | 0.88 (1 seed diverged) |
| Fourier 8 features, (6, 6), 199 params | 0.02 | 5 | 0.90 (0.70-1.50) | 1.26 | 0.52 |
| SIREN (w0 = 30, inputs scaled to [-1, 1]) | 0.005-0.02 | 5 | 2.99-3.06 | 3.2-3.7 | 0.4-1.0 (noise level) |

The classical net wanted a learning rate ten times the CV's; lower rates
(5e-4, 5e-5) sit at 2.9-3.2. The two spectral-bias cures did not help at
this size: Fourier features cost coverage, SIREN never left noise level
(before and after the input-scaling fix of 151c631). Baseline =
tanh (9, 9), g_lr 0.05, gp 5.

**DV** (`analysis/matrix/dv_ladder_step{1,2,3}.txt`, 72 runs, 3 seeds each;
latent_scale 1.0 everywhere — the pilot's 0.3 fed z as +-0.3 rad angles,
one cause of its collapse):

| arm (8 qubits, 16x16, RY encoding once) | layers | params | g_lr | fidelity W1 med (range) | coverage | div / real | min / run |
|---|---|---|---|---|---|---|---|
| ladder step 1 | 3 | 48 | 0.005 | 1.72 (1.61-1.94) | 1.73 | 0.89 | 11 |
| ladder step 1 | 6 | 96 | 0.005 | 1.51 (1.44-1.56) | 1.77 | 0.60 | 25 |
| ladder step 1 | 9 | 144 | 0.005 | 1.35 (1.31-1.39) | 1.46 | 0.52 | 44 |
| step 1, re-upload + RY/RZ, 3 / 6 / 9 layers | | 48-144 | 0.005 | 1.59 / 1.63 / 1.62 | 1.4-1.9 | 0.86 / 0.56 / 0.50 | 14-58 |
| step 2 | 6 | 96 | 0.02 | 0.99 (0.72-1.02) | 0.84 | 0.82 | 27 |
| **step 2** | **6** | **96** | **0.05** | **0.82 (0.78-0.85)** | **0.64** | 0.88 | 27 |
| step 2, latent_scale 1.57 | 3 / 6 | 48 / 96 | 0.02-0.05 | 1.00-1.58 | 0.6-1.4 | 0.8-1.5 | 11-27 |
| step 3 | 6 / 9 | 96 / 144 | 0.1 | 0.93 / 0.92 | 0.75 / 0.69 | 0.99 / 0.87 | 24 / 43 |
| step 3 | 6 / 9 | 96 / 144 | 0.2 | 0.96 / 0.96 | 0.75 / 0.92 | 0.95 / 0.85 | 24 / 43 |
| 6 qubits (8x8 grid), 3 / 6 / 9 layers | | 36-108 | 0.005 | 0.73 / 0.67 / 0.68 (8x8 cells) | 0.6-0.7 | 0.9-1.1 | 7 / 16 / 29 |

Reading: the learning rate was the knob (1.35 -> 0.82), depth beyond 6
layers and re-uploading were not, and the pass criterion is met by every
8-qubit cell at g_lr >= 0.02 — but none of them comes within x1.5 of the
CV or the classical arm re-rendered on the same grid (0.33 / 0.26 at 16x16
with 5 seeds, below), so the DV arm fails the cross-arm criterion on both
grids. 6 qubits is the cost floor (7 min a run, passes on its own 8x8
grid, kept out of the thesis table: an 8x8 ring is barely a ring).
Baseline = 8 qubits x 6 layers, RY, latent_scale 1.0, g_lr 0.05, gp 5.
Cheapest DV cell that is "somewhat reliable" on 16x16: 8 qubits x 3 layers
at g_lr 0.02-0.05 (0.97-0.98, 11 min a run).

**Confirmation, 20 seeds each** (seeds 0-19; `analysis/matrix/confirm_*.txt`;
CV and classical trained at 32x32 and re-rendered on 16x16 with
`rescore.py --grid 16`, DV trained at 16x16):

| arm | params | hilbert | grid | fidelity W1 med (range) | coverage W1 med (range) | div / real med (range) | min / run |
|---|---|---|---|---|---|---|---|
| CV frozen (3 modes, 6 layers, cutoff 12, gp 10) | 186 | 1728 | 32 | 0.64 (0.53-0.73) | 0.88 (0.74-1.14) | 0.62 (0.44-0.87) | 26 |
| classical tanh (9, 9), g_lr 0.05, gp 5 | 181 | — | 32 | 0.52 (0.41-0.65) | 0.58 (0.42-0.84) | 0.81 (0.49-0.99) | 2 |
| CV frozen, re-rendered | 186 | 1728 | 16 | 0.32 (0.26-0.36) | 0.44 (0.37-0.56) | 0.62 (0.44-0.87) | — |
| classical tanh, re-rendered | 181 | — | 16 | 0.26 (0.21-0.32) | 0.29 (0.21-0.41) | 0.81 (0.49-0.99) | — |
| DV 8 qubits x 6 layers, g_lr 0.05, gp 5 | 96 | 256 | 16 | 0.85 (0.72-1.07) | 0.66 (0.55-0.85) | 0.94 (0.70-1.26) | 27 |

Every arm remains distinguishable from the real family (MMD p = 0.005 on
every run). **Result on the ring at matched parameter count: the tuned
classical coordinate network beats the frozen CV generator** (0.52 vs 0.64
at 32x32, ranges barely overlapping over 20 seeds; 0.26 vs 0.32 at 16x16),
the CV generator keeps the better coverage (0.88 vs 0.58 at 32x32: the
classical net is sharper but misses more of the real family), and the DV
generator trails both by a factor 2.5-3 on the common grid at a quarter of
the Hilbert dimension and half the parameters. This is the outcome Bowles,
Ahmed & Schuld (2024) describe and it is the honest baseline for the
thesis: the pilot's "CV leads" read was the untuned-baseline artefact.
What it does not say: anything about the other four families, about the
quantum critics, or about the CV arm's own tuning headroom (never
re-tuned here). Caveat on the criterion: with 20 seeds one seed of each
32x32 arm falls just under the x2 diversity bound (0.44, 0.49), so the
strict flag reads "no" for both; the bound was set for 3-5 seeds.

Open follow-ups, in order: (1) the same three tuned arms on the other four
families and on the three point-wise critics (the matrix of step 2 below);
(2) the CV arm at the classical net's learning rate (0.05) and at gp 5, as
a fairness check in the other direction; (3) a DV arm at 10 qubits with
g_lr 0.05 to see whether the 32x32 DV result of the pilot was only the
learning rate.

1. Pilot: 3 generators × `dense_2d` on ring, 5 seeds, 300 epochs.
2. Matrix: generators × critics × 5 families at one budget and one grid;
   re-scored with Phase 1f; results as paired comparisons per family.
3. Edge finding: the survey families (cat fringe, Fock egg-crate, cubic
   phase, ħ = 1 formulas) where CV is expected to win or lose.

### Statistical design (from the 7–8 Nov interleaved runs)

- **Realization noise is large.** The exact baseline model run with a
  9-component latent stream instead of 6 (nothing else changed) moved the
  5-seed median nearest-member W1 from 0.673 to 0.814 at 300 epochs — the
  same size as every "effect" the interleaved arms showed. A different RNG
  consumption is a different experiment.
- **Rule 1 — matched controls.** Every arm is compared against a control
  with the same latent dimension and RNG consumption (same `latent_dim`,
  same sampling order), or against a per-backend control; CV, DV and
  classical generators either share `latent_dim` or each carries its own
  control arm.
- **Rule 2 — seed counts.** Five seeds at 300 epochs resolve differences of
  about 0.10–0.15 in nearest-member W1. An effect of 0.05 needs 20 or more
  seeds, or a paired-restart design (same seed, same data, arms differing
  only in the knob) with the re-scored metrics (`scripts/rescore.py`,
  200 samples, ±0.02 per run instead of ±0.08).
- **Rule 3 — report the stick with the number.** Every table states the
  validation sample count, the seed count, the budget and whether the
  value is the trainer's best-checkpoint reading or the re-score.

## Fences

- Never retrain the July/November baselines; re-render CV checkpoints
  when the grid changes.
- Metrics are fixed (Phase 1f) before the matrix is filled.
- No classical front-end before any quantum circuit (keeps the "quantum
  generator/critic" claims honest).
- One backend per PR; merge commits; goldens green at every commit.

## Order

0 → 1a → 1f → 1b spike → 1c → 1b build → 1d → 1e → 2 → 3. The classical
generator goes first because it is the cheapest proof that the contract is
sufficient; the evaluation goes second so every later piece is scored the
same way.
