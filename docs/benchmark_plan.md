# Benchmark framework plan: CV vs DV vs classical, pieces first, plumbing last

Status: draft, 7 Nov 2026. Objective 2 of the thesis: compare the CV-QGAN
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
