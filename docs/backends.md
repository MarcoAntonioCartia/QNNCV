# Benchmark backends (`src/backends/`)

Status: Phases 0–2 of `docs/benchmark_plan.md` done (8 Nov 2026): every backend is
contract-tested (`tests/golden/test_backend_contract.py`) and selectable from the
CLI — `--generator {cv,classical,dv}`, `--critic {dense_2d,coord_2d,cv_qnn_2d,
dv_qnn_2d}`, plus `--critic-points`, `--critic-layers`, `--gen-hidden`,
`--dv-qubits`, `--dv-layers`. The defaults are the frozen CV + dense path,
byte-identical to before. Non-CV runs save a backend-state `best_weights.npy`
(read by `scripts/rescore.py` and `src/backends/trainer_shim.py`), and every
run writes `backend_info.json` (parameter and Hilbert-space sizes, device, wall
time). The DV backends need the cloned env `qnncv-dv` (PennyLane 0.45.1,
pennylane-lightning; `qnncv` untouched) and a power-of-two `--grid-size`.

```bash
# the classical generator against the point-wise classical critic, ring, 300 epochs
python train_2d_qgan.py --family ring --generator classical --critic coord_2d --epochs 300 --seed 0
# the frozen CV generator against the 96-parameter CV quantum critic
python train_2d_qgan.py --family ring --n-modes 3 --d-lr 0.0005 --gp-weight 10 --critic cv_qnn_2d --epochs 300 --seed 0
# the DV generator (qnncv-dv env) on a 32x32 grid against its qubit critic
python train_2d_qgan.py --family ring --generator dv --grid-size 32 --critic dv_qnn_2d --epochs 300 --seed 0
```

## Contract (`base.py`)

| `GeneratorBackend` | `CriticBackend` |
|---|---|
| `latent_dim`, `batch_size` | `__call__(x) -> (B, 1)` on the trainer's critic input `(B, G, G, 1)` |
| `generate_batch(z, xvec, yvec) -> (probs (B,G,G), aux)` | `trainable_variables` |
| `regularizer(aux)` — the backend's own penalty (CV: ket norm; others 0) | `describe()` → `backend, n_params, hilbert_dim` |
| `state_dict() / load_state_dict()` exact round-trip | |
| `describe()` → `backend, n_params, hilbert_dim, latent_dim` | |

`hilbert_dim` = size of the simulated state space; reported next to `n_params`
for every backend (the fairness rule).

## Generators

| name | class | design | params (default) | hilbert_dim | grid |
|---|---|---|---|---|---|
| `cv` | `CVGeneratorBackend` | adapter around `CVQGANGenerator`, bit-identical; checkpoints = existing `weights_io` v1/v2 files | 186 (3 modes, 6 layers) | cutoff^modes = 1728 | any |
| `classical` | `ClassicalCoordGenerator` | Keras coordinate net `f(x, y, z)` on the grid, softplus + normalise; `parameter_matched()` → closest to 186; `large()` → 4,801; `features='xyr'` = explicit-radius ablation | 181 ((9, 9), latent 6) | — | any |
| `dv` | `DVGeneratorBackend` | PennyLane `lightning.qubit`, `n_qubits = 2k` for a `2^k × 2^k` grid; z as fixed RY angles (cyclic); layers RY/RZ + CNOT ring; `qml.probs` reshaped `[y, x]`; `diff_method='best'` (adjoint does not support probs) | 2·n·L (10 qubits × 9 layers = 180) | 2^n = 1024 | powers of two only (`grid_sizes`) |

z is never trainable in any backend (January-2026 rule).

## Critics

| name | class | design | params (default) | hilbert_dim |
|---|---|---|---|---|
| `dense_2d` | `DenseCriticBackend` | adapter around the live `Discriminator2D` (grid MLP 16-8-1) | 25,809 | — |
| `coord_2d` | `CoordCritic` | point-wise Keras MLP `D(x, y)` | 105 ((8, 8)) | — |
| `cv_qnn_2d` | `CVQuantumCritic` | point-wise 2-mode Killoran circuit (same machinery as the generator), point enters as a fixed displacement (`point_scale` 0.5), readout `<n_0> − <n_1>` | 16/layer → 96 at 6 layers | cutoff² = 100 |
| `dv_qnn_2d` | `DVQuantumCritic` | point-wise qubit circuit, point as fixed RY angles, readout `<Z_0>`; parameter broadcasting over the point set | 2·n·L (4 qubits × 12 layers = 96) | 2^n = 16 |

Point-wise critics share `pointwise.py`: `score_b = (G²/|S|) Σ_{s∈S} x_b(s) D(s)`
over `n_points` grid indices drawn per call (0 = full grid, exact). The score is
linear in the input, so the WGAN gradient penalty sees `(G²/|S|) D(S)`. The
contract suite checks the full-grid score equals `Σ P·D` exactly and the sampled
estimator is unbiased (within 4 SE over 128 draws).

## Standalone checks

- `scripts/smoke_backend.py --generator {cv,classical,dv}` fits a generator to a
  family's canonical member by supervised KL (+ regularizer) without the GAN
  trainer. Its yardstick is **supervised/canonical W1** (Path A); it is never
  comparable to the GAN sweep's nearest-member W1.
- `scripts/rescore.py <run dirs>` re-scores saved checkpoints with 200 fixed-seed
  samples: fidelity + coverage (W1 and energy distance), diversity vs the
  validation set's self-diversity, MMD two-sample test; `analysis/rescore/`.

## Known constraints

- Strawberry Fields' `FreeParameter` is a sympy `Symbol` cached by name: two
  circuits in one process must use distinct parameter names (fixed in
  `circuit.py` with per-instance `g{uid}_` prefixes). Without it a quantum
  critic next to the quantum generator corrupts parameter bindings after many
  runs.
- One grid for every backend is still an open decision (32×32 or 64×64); the CV
  backend re-renders checkpoints on any grid, the DV backend cannot produce
  40×40.
- Speeds on this PC (CPU): DV generator ≈ 0.13 s per sample fwd+bwd at
  10 qubits / 3 layers; CV ≈ 0.5 s per batch of 8 at the frozen config.
