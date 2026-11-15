# QNNCV implementation document

State of the code on `main` as of 14 Nov 2026. Everything below was read from the source, not from older notes; where the code and an older document disagree, section 13 lists the disagreement. File references are relative to the repository root.

Contents: 1 repository map · 2 one training run end to end · 3 data · 4 the generator contract · 5 the three generators · 6 the critic contract and input pipeline · 7 the four critics · 8 the trainer · 9 metrics · 10 run outputs · 11 tooling and tests · 12 configuration traps and canonical commands · 13 known inconsistencies · 14 how to extend.

---

## 1. Repository map

The live path is the 2-D QGAN. It is small: about 3,000 lines. The rest of `src/` is the legacy 1-D work from 2025, kept because old entry points and notebooks import it.

```
train_2d_qgan.py            CLI shim: argparse, folder naming, run_config.json, calls the trainer
run_sweep.py                numeric hyper-parameter sweeps (cartesian product, one window per run)

src/
  quantum/                  THE CV CIRCUIT
    circuit.py                EncodingSpec + CVQGANGenerator (symbolic SF program, readout)   612 lines
    hermite.py                Fock -> position basis functions, cutoff recommendation
    weights_io.py             best_weights.npy formats v1 / v2 and their compatibility rules
  backends/                 BENCHMARK FRAMEWORK (interchangeable generators and critics)
    base.py                   GeneratorBackend / CriticBackend abstract contracts
    registry.py               GENERATOR_REGISTRY, CRITIC_REGISTRY, toy factories for the tests
    cv.py                     adapter: CVQGANGenerator behind the contract
    classical.py              classical coordinate-network generator
    dv.py                     qubit generator (PennyLane)
    pointwise.py              shared scoring rule of the point-wise critics
    classical_critic.py       adapter: the dense grid critic behind the contract
    coord_critic.py           classical point-wise critic
    cv_critic.py              CV quantum point-wise critic
    dv_critic.py              qubit point-wise critic (PennyLane)
    trainer_shim.py           how the trainer reaches non-default backends; backend-state checkpoints
  models/discriminators/
    qgan2d_discriminator.py   Discriminator2D, the default dense critic
  critic_input/pipeline.py  peak normalisation, blur, gradient penalty
  training/qgan_2d.py       trainer classes and the training loop                             917 lines
  families/                 target distribution families (gaussian, correlated, four_gaussians, ring, vibronic)
  data/dataset.py           draws the 400 + 100 family members
  metrics/                  marginal W1, energy distance, validate()
  viz/plots.py              per-epoch comparison plots, history plot
  utils/                    compatibility patches (SciPy/TF/NumPy), warning suppression

scripts/
  run_matrix.py             run a list of argument lines N at a time, with a free-RAM guard
  rescore.py                re-score checkpoints with 200 fixed samples (fidelity, coverage, MMD)
  summarize_rescore.py      per-arm table with the pre-registered pass criterion
  analyze_sweep.py          group runs by tag, paired comparisons with bootstrap intervals
  z_sweep_diagnostic.py     how the output density depends on each latent component
  smoke_backend.py          fit one backend to a single target by supervised KL

tests/golden/               19 regression suites; run_all.py runs each in a fresh process
docs/                       benchmark_plan.md, backends.md, experiments_interleaved.md, this file
analysis/matrix/*.txt       tracked job lists (everything else under analysis/ is ignored)
logs/                       one folder per run (ignored by git)
```

Legacy, not on the 2-D path: `src/models/generators/*`, `src/models/discriminators/{classical,distribution,quantum_sf}_discriminator.py`, `src/training/{trainer,killoran_trainer,distribution_trainer,qgan_sf_trainer}.py`, the root scripts `qnncv.py`, `killoran_cvqnn_multimode.py`, `simple_quantum_neural_network.py` and the notebooks.

Two environments: `qnncv` (TensorFlow 2.18 CPU, Strawberry Fields 0.23, NumPy 2.0.2) runs everything except the qubit backends; `qnncv-dv` is a clone with PennyLane 0.45.1 and `pennylane-lightning`. In `qnncv` the registry simply omits `dv` and `dv_qnn_2d` (`registry.py`, lines 21–27).

How the code is divided, in one sentence per layer:

- **Physics** (`src/quantum`): one class builds and runs the CV circuit and turns its ket into a density on a grid. It knows nothing about GANs.
- **Contracts** (`src/backends`): two abstract classes say what a generator and a critic must offer; every model is either written against them or wrapped by a thin adapter.
- **Training** (`src/training/qgan_2d.py`): one loop owns setup, validation, checkpointing and outputs; two small trainer classes own the per-epoch critic phase and generator phase.
- **Evaluation** (`src/metrics`, `scripts/rescore.py`, `scripts/analyze_sweep.py`): cheap metrics during training, careful metrics afterwards from the saved checkpoint.

---

## 2. One training run end to end

```
python train_2d_qgan.py --family ring --n-modes 3 --d-lr 0.0005 --gp-weight 10 --grid-size 32 --epochs 300 --seed 0
```

1. `train_2d_qgan.py::main` parses the flags, resolves the seed, builds the folder name from every flag that differs from its default (`build_hp_suffix`), claims the folder atomically and writes `run_config.json`.
2. `src/training/qgan_2d.py::train_2d_qgan` seeds Python, NumPy and TensorFlow, builds the family, draws 500 members (400 train, 100 validation) and hashes the validation set.
3. It builds the generator (`CVQGANGenerator` directly for `--generator cv`; anything else through `trainer_shim.build_generator` wrapped in `GeneratorShim`) and the critic (`Discriminator2D` directly for `--critic dense_2d`; anything else through `trainer_shim.build_critic`).
4. It picks `WGANGPTrainer` (or `PureSupervisedTrainer` when the supervised weight never decays) and loops over epochs: `discriminator_epoch` then `generator_epoch`, then bookkeeping.
5. Every `val_every` epochs it runs `validate()` and snapshots the generator if the nearest-member W1 improved.
6. At the end it restores the best snapshot and writes `best_weights.npy`, `backend_info.json`, `history.npz`, `norm_monitor.npz` and the plots.

---

## 3. Data

`src/families/base.py`: a family owns the grid. `xvec = yvec = linspace(-x_range, x_range, grid_size)` with `x_range = 3.0`, and `X, Y = np.meshgrid(xvec, yvec)`. **Convention of the families, the point-wise critics and the DV generator: a density is an array `[y, x]`, flat index `k = i*G + j` is the point `(x = xvec[j], y = yvec[i])`. The CV and classical generators output `[x, y]` instead (CV: mode 0 on axis 0; classical: `meshgrid(..., indexing='ij')`).** Training and every metric are unaffected, because arrays are compared as they are and the generators simply learn the transposed picture; but any statement linking a latent component or a mode to "the x position" must account for it (corrected 15 Nov). A family offers `sample()` (a random member and its parameters) and `get_canonical()` (the central member). Every member is normalised to sum 1.

| family | random parameters | canonical member |
|---|---|---|
| `gaussian` | centre ±1.0, sigma 0.3–0.8 | centred, mid sigma |
| `correlated` | centre ±0.5, rho 0.4–0.85, scale 0.6–1.2 | centred, mid values |
| `four_gaussians` | centre ±0.3, spread 1.0–1.8, sigma 0.2–0.4 | centred, mid values |
| `ring` | centre ±0.5, radius 0.8–1.5, width 0.15–0.35 | centred, radius 1.15, width 0.25 |
| `vibronic` | one pair of normal modes of the formic-acid Doktorov state (an analytic bivariate Gaussian), standardised to the grid | the pair with the largest combined displacement |

`src/data/dataset.py::generate_dataset` calls `family.sample()` `n_train + n_val` times from the NumPy global stream, so the same seed always gives the same split; the trainer records the first 16 hex digits of the validation set's SHA-1 and `rescore.py` refuses to score against a different split.

The target of the GAN is the *family*, not one density: each generator sample (one latent vector) should look like *some* member.

---

## 4. The generator contract

`src/backends/base.py::GeneratorBackend`:

| member | meaning |
|---|---|
| `latent_dim` | width of the latent vectors |
| `batch_size` | batch evaluated in one call, or `None` for any size |
| `generate_batch(z, xvec, yvec) -> (probs (B, G, G) float32, aux)` | densities summing to 1; `aux` is backend-specific (CV: the `(B,)` ket norms) |
| `generate_distribution_2d(z, xvec, yvec)` | one latent vector |
| `trainable_variables` | list of `tf.Variable` |
| `regularizer(aux)` | the backend's own penalty; default 0 |
| `state_dict()` / `load_state_dict()` | exact checkpoint round trip |
| `describe()` | at least `backend`, `n_params`, `hilbert_dim`, `latent_dim` |

Rule for every backend: **the latent vector is an input, never a trainable quantity, and no trainable classical layer sits in front of a quantum circuit.**

The trainer was written against the CV class before the contract existed, so it still calls the CV surface (`generate_batch(..., return_ket_norm=True)`, `.weights`, `.w_s`). `trainer_shim.GeneratorShim` presents that surface for any contract backend: it returns ket norms equal to 1 (so the trainer's ket penalty is exactly 0) and routes snapshots through `state_dict()`.

---

## 5. The three generators

### 5.1 CV generator — `src/quantum/circuit.py::CVQGANGenerator` (adapter `src/backends/cv.py`)

**Circuit.** `n_modes` qumodes, of which the first two are read out and the rest are ancillas. Per sample:

```
front encoding   Dgate(latent_scale * z[2m], z[2m+1])  on every mode m
then n_layers times (Killoran et al. layer):
   U1   interferometer
   S    Sgate(r, phi)         on every mode
   U2   interferometer
   D    Dgate(r, phi)         on every mode
   K    Kgate(kappa)          on every mode   (omitted with --no-kerr)
```

The interferometer (`_apply_interferometer`) is a rectangular mesh: `N` columns, in column `l` a `BSgate(theta, phi)` on neighbours `(k, k+1)` whenever `l + k` is even, which gives `N(N-1)/2` beamsplitters, followed by `N-1` `Rgate`s. One mode: a single `Rgate`.

**Parameter count.** Interferometer `M = N(N-1) + N - 1`. Per layer `2M + 2N + 2N + N` (with Kerr).

| modes | M | per layer | 6 layers | Fock dimension at cutoff 12 |
|---|---|---|---|---|
| 2 | 3 | 16 | 96 | 144 |
| 3 (frozen config) | 8 | 31 | **186** | **1,728** |

**Weights.** One flat `tf.Variable` `qnn_weights` of length `total_params`, consumed in circuit order. Initialisation: normal, standard deviation 0.1 for everything (`active_sd`, `passive_sd`), 0.05 for Kerr. See section 13 for a labelling quirk in `_init_weights` that is harmless at these defaults.

**Symbolic program.** The Strawberry Fields `Program` is built once (`_build_symbolic_circuit`) with free parameters for the `2 * n_modes` front latent components and every weight. Each call binds tensors to those names (`_run_circuit`) and runs the TensorFlow backend engine. Parameter names carry a per-instance prefix `g<uid>_` because SF's `FreeParameter` is a SymPy `Symbol`, SymPy caches symbols by name, and two circuits in one process (generator and quantum critic) would otherwise share bindings.

**Batching.** The engine is created with `batch_size` (the training batch, 8). A batched call binds one `(B,)` tensor per latent parameter and scalar weights that broadcast; one SF run evaluates the whole batch. A call with a different batch size falls back to a sequential loop over single latents.

**Readout** (`generate_distribution_2d`, `_generate_batched`). The engine returns the ket `c[n_0, n_1, ..., n_{N-1}]`. With the Hermite functions `phi_n(x) = H_n(x) exp(-x^2/2) / sqrt(2^n n! sqrt(pi))` from `hermite.py` (natural units, vacuum variance 1/2):

```
no ancilla:    psi(x, y)  = sum_{n,m} c[n,m] phi_n(x) phi_m(y)             P = |psi|^2
with ancilla:  psi_k(x,y) = sum_{n,m} c[n,m,k] phi_n(x) phi_m(y)           P = sum_k |psi_k|^2
```

where `k` runs over all Fock states of the ancilla modes: the partial trace is the sum over `k`. Implemented as one `einsum` (`'bnmk,xn,ym->bxyk'`). The density is then divided by its sum on the grid. The output is the position-quadrature probability density of the two output modes, evaluated exactly on the grid; no sampling is involved.

**Ket norm.** `sum |c|^2` inside the cutoff is returned alongside the density. It is 1 when the cutoff holds the state and below 1 when squeezing or displacement push weight past the cutoff. The trainer penalises `mean((1 - norm)^2)` and warns per sample below `--norm-floor`.

**Latent encoding — `EncodingSpec`.** The strategy object decides how `z` enters.

- `input_displacement` (default): the front `Dgate` only. `latent_dim = 2 * n_modes`.
- `interleaved_squeeze`: additionally, in every injected layer the squeeze magnitude becomes `r = w_r + w_s * tanh(z_c)`. `w_s` is an `(n_layers, n_modes)` variable filled with `--squeeze-scale`; it is fixed unless `--train-squeeze-scale`. `z_c` is `z[2m]` (`--inj-latent shared`, the component that also sets the front displacement) or an extra component `z[2*n_modes + m]` (`dedicated`, `latent_dim = 3 * n_modes`). `--inj-layers` restricts the layers, `--r-max` clips `|r|`. **The symbolic program is unchanged**: the injection happens in the binding step, which replaces the tensor bound to each squeeze-magnitude parameter. At `w_s = 0` the model is numerically the baseline (golden-tested).

**Checkpoints** (`weights_io.py`). `input_displacement` saves the raw flat array (format v1, loads without pickle). `interleaved_squeeze` saves a dict with `weights`, `w_s` and metadata (v2). Loading rules: v1 into an interleaved model is a warm start with `w_s` at its configured value; v2 into a plain model is an error; a v2 file whose `w_s` differs from a fixed scale is an error.

### 5.2 Classical generator — `src/backends/classical.py::ClassicalCoordGenerator`

A coordinate network: `f(x, y, z) -> R`, evaluated at every grid point with the same latent vector, then `P = softplus(f) / sum softplus(f)`.

Why this shape: a network that emitted the grid directly would need `G^2` output weights per hidden unit, so no parameter match to a 186-parameter circuit would be possible. A coordinate network's size is independent of the grid.

- Input: `[x, y] (+ r = sqrt(x^2+y^2) with features='xyr') (+ 2K random Fourier features) + z`, with `z` fed raw.
- Body: `Dense(h1, tanh) -> Dense(h2, tanh) -> Dense(1)`, weights `N(0, 0.5^2)`.
- Default `(9, 9)` with latent dimension 6: `(8+1)*9 + (9+1)*9 + 10 = ` **181 parameters**, the closest two-layer size to 186 (`parameter_matched()`). `large()` is `(64, 64)`, 4,801 parameters.
- Variants kept as ablations: `--gen-activation sine` (SIREN, `w0 = 30`, SIREN initialisation, inputs divided by the half-range) and `--gen-fourier K` (fixed random frequency matrix drawn from the seed). Both lost to plain tanh on the ring.
- `generate_batch` tiles the cached grid features `(G^2, F)` against the batch `(B, L)`, runs the network once on `B * G^2` rows and reshapes. No `aux`, no regulariser, `hilbert_dim = None`.

### 5.3 DV generator — `src/backends/dv.py::DVGeneratorBackend` (PennyLane)

- `n = 2k` qubits produce a `2^k × 2^k` density: the Born distribution over all basis states, the first `k` qubits indexing the row and the last `k` the column. 10 qubits give 32×32 (dimension 1,024). The backend refuses any other grid.
- Encoding: `RY(latent_scale * z[i mod D])` on qubit `i` (`--dv-encoding ry`), optionally followed by `RZ` of the next component (`ryrz`); `--dv-reupload` repeats the encoding before every layer. No parameters.
- Layers: `n_layers × [RY(w), RZ(w) on every qubit; CNOT ring]`, weights `(n_layers, n_qubits, 2)`, `N(0, 0.1^2)`. **10 qubits × 9 layers = 180 parameters.**
- Readout `qml.probs` over all wires, reshaped to `(G, G)`. Device `lightning.qubit`, `diff_method='best'` (adjoint does not support `probs`), one device call per sample.

### 5.4 Size summary

| generator | parameters | simulated state dimension | latent | grid |
|---|---|---|---|---|
| CV, 3 modes × 6 layers, cutoff 12 | 186 | 1,728 | 6 | any |
| classical (9, 9) | 181 | none | 6 | any |
| DV, 10 qubits × 9 layers | 180 | 1,024 | follows `--n-modes` (see section 12) | 32×32 only |

---

## 6. The critic contract and the input pipeline

`CriticBackend`: `__call__(x, training) -> (B, 1)` on an input `(B, G, G)` or `(B, G, G, 1)`, `trainable_variables`, `describe()`.

Every critic sees inputs that went through `src/critic_input/pipeline.py`, identically for real and generated densities:

1. `critic_blur(x, kernel)` — optional Gaussian blur (`--critic-blur`, off by default), applied once.
2. `to_critic_input(x)` — divide each density by its own maximum. Sum-normalised densities have values near 0.001–0.01, so a Lipschitz-constrained critic could not separate them; peak normalisation restores an order-one range.
3. Instance noise (critic phase only): `relu(x + sigma * N(0, 1))`, then peak-normalise again. `sigma` decays linearly from `--instance-noise` (0.1) to `--noise-floor` (0) over `--noise-anneal` (200) epochs.
4. `compute_gradient_penalty` — one interpolation coefficient per batch element between the real and generated inputs the critic actually saw; the penalty is `mean((||grad_x D||_2 - 1)^2)`.

---

## 7. The four critics

### 7.1 `dense_2d` — `Discriminator2D` (default)

`Flatten -> [Dense(16) -> LayerNorm -> LeakyReLU(0.2)] -> [Dense(8) -> LayerNorm -> LeakyReLU(0.2)] -> Dense(1)`, weights `N(0, 0.05^2)`, optional dropout. Its size is set by the grid: `G^2 * 16 + 16 + 32 + 136 + 16 + 9`.

| grid | parameters |
|---|---|
| 40×40 | 25,809 |
| 32×32 | 16,593 |

### 7.2 The point-wise family — `src/backends/pointwise.py::PointwiseCritic`

A point-wise critic is a scalar function of a grid point, `D(x, y)`. It scores a density by a weighted sum over a subset `S` of grid cells:

```
score_b = sqrt(G^2 / |S|) * sum_{s in S} x_b(s) * D(s)
```

- `|S| = --critic-points` (64); 0 means the whole grid, where the factor is 1 and the score is the exact inner product of density and critic function.
- The factor is the square root, not the unbiased Monte-Carlo factor `G^2/|S|`. Reason: the score is linear in the input, so the gradient penalty constrains `c * ||D restricted to S||`. With `c = sqrt(G^2/|S|)` the expected gradient norm equals the full-grid norm, so critics with different `|S|` learn the same function; the score itself is then the exact sum times a constant, which a Wasserstein critic absorbs.
- `new_points()` draws a fresh subset; the trainer calls it once at the start of every critic step and every generator step, so real, generated and interpolated inputs of one step use the same cells.

Subclasses only implement `evaluate(points (n, 2)) -> (n,)`.

| critic | `D(x, y)` | parameters | simulated dimension |
|---|---|---|---|
| `coord_2d` (`coord_critic.py`) | `Dense(8, tanh) -> Dense(8, tanh) -> Dense(1)` on `(x, y)` | `3*8 + 9*8 + 9 = ` **105** | none |
| `cv_qnn_2d` (`cv_critic.py`) | 2-mode Killoran circuit, see below | 16 per layer, **96** at 6 layers | cutoff² (144 at cutoff 12) |
| `dv_qnn_2d` (`dv_critic.py`) | 4-qubit circuit, see below | `2 * 4 * layers`, **96 at 12 layers** | 16 |

**`cv_qnn_2d`.** It instantiates a second `CVQGANGenerator` with two modes and no ancilla and uses only its circuit machinery. The grid point enters where the latent vector would: `z = (point_scale * x, 0, point_scale * y, 0)` with `point_scale = 0.5`, so mode 0 is displaced by `0.5 x` and mode 1 by `0.5 y`. The readout is `<n_0> - <n_1>`, the difference of mean photon numbers, computed from `|ket|^2`. The SF batch is the point set: one engine run evaluates all 64 points. Its cutoff follows the generator's `--cutoff-dim`. This is the expensive critic: five critic steps per epoch, each with a real, a generated and a penalty pass through the circuit, about 50 s per epoch against 4 s for the dense critic.

**`dv_qnn_2d`.** `RY(0.5 x)` on even wires and `RY(0.5 y)` on odd wires, then `layers × [RY, RZ on every qubit; CNOT ring]`, readout `<Z_0>`. Adjoint differentiation on `lightning.qubit`, and the whole point set in one device call through parameter broadcasting when the device allows it.

---

## 8. The trainer — `src/training/qgan_2d.py`

**Structure.** `_BaseTrainer2D` holds the schedules and the whole generator step. `WGANGPTrainer` adds the critic phase and the adversarial loss combination. `PureSupervisedTrainer` ("Path A") disables the critic and fits one fixed target by KL; it is selected automatically when `supervised_weight >= 1` and `supervised_warmup >= epochs`, and is used only for representability tests.

**One epoch = `n_critic` (5) critic updates, then one generator update.** With 300 epochs the generator takes 300 Adam steps.

Critic update (`WGANGPTrainer.discriminator_epoch`), repeated `n_critic` times:

1. new point subset for point-wise critics;
2. a random batch of 8 training members;
3. 8 latent vectors `z ~ N(0, I)`, generated densities computed **outside** the tape and detached;
4. blur, peak normalisation, instance noise;
5. `L_D = mean(D(fake)) - mean(D(real)) + lambda(epoch) * GP`, with `lambda` ramping linearly from 0 to `--gp-weight` over `--gp-warmup` (50) epochs;
6. Adam step.

Generator update (`_BaseTrainer2D.generator_epoch`):

1. new point subset; a batch of real members (used only by the supervised term); 8 fresh latent vectors;
2. inside the tape: densities and ket norms, then the critic score of the blurred, peak-normalised densities;
3. `L_G = (1 - s) * (-mean(D(fake))) + s * KL(real || fake) + ket_penalty_weight * mean((1 - ket_norm)^2)`, where `s` decays from `--supervised-weight` to 0; **the default supervised weight is 0, so the generator loss is the adversarial term plus the ket penalty (weight 20)**;
4. gradients clipped per variable to norm `--g-grad-clip` (5), Adam step;
5. per-sample ket-norm monitoring.

Both optimisers are Adam with `beta_1 = 0.5`, `beta_2 = 0.9`. Non-finite generator output skips the epoch; more than 30 in a row stops the run.

**Validation and checkpoint selection.** Every `--val-every` (20) epochs `validate()` draws 10 samples and reports the nearest-member W1, the canonical W1, the diversity and two energy distances. The snapshot with the lowest nearest-member W1 becomes `best_weights.npy`. Ten samples give a noise of about ±0.08 per reading and the minimum over 15 readings is optimistic by about 0.03, which is why results are quoted from `rescore.py`, not from the trainer.

**Reproducibility.** The order of random draws per epoch is fixed and golden-tested: per critic step NumPy batch indices, then TensorFlow `z`, then noise, then the penalty coefficients; then the generator's NumPy indices and `z`; then validation and plot draws. `--deterministic` additionally enables TensorFlow op determinism and single-threaded execution. Bit-identity holds on one machine, not across machines.

---

## 9. Metrics

All distances are between densities on the grid.

- **Marginal W1** (`src/metrics/wasserstein.py`): the mean of the 1-D Wasserstein-1 distances between the x-marginals and between the y-marginals, **in grid cells**. It ignores the joint structure; two densities with equal marginals are at distance 0.
- **Energy distance** (`src/metrics/energy.py`): `2 p'Dq - p'Dp - q'Dq` with `D` the Euclidean distance matrix between grid points, in x/y units. Sensitive to the full 2-D geometry. Reported alongside, never used for selection.
- **Trainer validation** (`src/metrics/validation.py`): canonical W1, nearest-member W1, diversity (mean pairwise W1 among the generated samples), from 10 samples.
- **Re-score** (`scripts/rescore.py`): rebuilds the validation set, loads the checkpoint, draws **200** samples with a fixed seed and computes the full fake × validation W1 matrix `W`:
  - fidelity W1 = mean over generated samples of the distance to the nearest validation member (`W.min(1).mean()`);
  - coverage W1 = mean over validation members of the distance to the nearest generated sample (`W.min(0).mean()`);
  - both are distances, **lower is better**;
  - diversity against the validation set's own self-diversity;
  - unbiased MMD² with an RBF kernel on the flattened densities, median-heuristic bandwidth, and a permutation p-value;
  - the same fidelity and coverage in energy distance; ket-norm minimum and mean.
  - `--grid G` re-renders CV and classical checkpoints on another grid.

---

## 10. Run outputs

Folder: `logs/qgan_2d_<family>_<YYYYMMDD_HHMMSS><tags>`. A tag appears **only for a flag that differs from its CLI default**, for example `_m3_ep300_dlr0.0005_gp10_gs32_critcvq_seed0`. Tag codes are in `train_2d_qgan.py::build_hp_suffix` (`m` modes, `ep` epochs, `glr`/`dlr` learning rates, `gp`, `gs` grid, `ls` latent scale, `g` generator `cl`/`dv`, `crit` critic `coord`/`cvq`/`dvq`, `clay` critic layers, `nq`/`dvl` qubits and layers, `enc`/`ss`/`zl` encoding).

| file | content |
|---|---|
| `run_config.json` | every argument, the seed, library versions |
| `best_weights.npy` | CV: weights v1 or v2. Other generators: a `qnncv-backend-state-v1` payload under the same file name |
| `backend_info.json` | `describe()` of generator and critic, epochs run, wall seconds, validation-set hash |
| `history.npz` | per-epoch losses, scores, gradient norms, schedules, ket norm, validation series, seed |
| `norm_monitor.npz` | per-epoch minimum, mean, maximum ket norm, warning counts, worst latent vector |
| `epoch_*.png`, `final_samples.png`, `training_history.png` | plots |

---

## 11. Tooling and tests

| tool | use |
|---|---|
| `run_sweep.py` | numeric sweeps as `--flag START STOP STEP`; other flags are passed through to every run |
| `scripts/run_matrix.py jobs.txt --parallel N --tag T [--min-free-gb G]` | arbitrary job lists; console per run in `logs/matrix_<tag>/NNN.log`; the guard starts a run only while free RAM exceeds `G` |
| `scripts/rescore.py <run dirs> [--grid G]` | writes `analysis/rescore/<run>.json` and `rescore.csv` |
| `scripts/summarize_rescore.py rescore.csv --filter gs32` | per-arm medians and the pass criterion |
| `scripts/analyze_sweep.py --compare A B --rescore rescore.csv` | paired by seed: bootstrap intervals, probability of improvement, IQM, Wilcoxon |
| `scripts/z_sweep_diagnostic.py` | which latent component moves position, size, width |
| `scripts/smoke_backend.py` | single-target supervised fit of one backend |

Tests: `tests/golden/run_all.py` runs 19 suites, each in its own process. They pin the Hermite basis, the readout with and without ancilla, the full circuit forward pass, the critic-input pipeline, the metrics, the families, two end-to-end runs through the real CLI, the interleaved-squeeze equivalences, the checkpoint formats, and, for every registered backend, shapes, batched-equals-sequential, gradient reach and exact state round trip (`test_backend_contract.py`). Three structural suites keep the CLI defaults, the internal signatures and the sweep tool in sync.

---

## 12. Configuration traps and canonical commands

The CLI defaults are **not** the benchmark configuration. Four defaults have already produced wrong runs; always pass these explicitly.

| flag | CLI default | benchmark value | consequence of forgetting it |
|---|---|---|---|
| `--d-lr` | 0.005 | 0.0005 | critic ten times too fast; different failure mode |
| `--n-modes` | 2 | 3 | CV: 96 parameters and no ancilla. Classical and DV: latent dimension 4 instead of 6 (classical drops to 163 parameters) |
| `--latent-scale` | 0.3 | CV 0.3, **DV 1.0** | DV encoding angles of ±0.3 rad, which contributes to collapse |
| `--critic-layers` | 6 | 6 for `cv_qnn_2d`, **12 for `dv_qnn_2d`** | DV critic with 48 parameters instead of 96 |
| `--grid-size` | 40 | 32 | DV cannot produce 40×40; the dense critic changes size |
| `--gp-weight` | 5 | CV 10, classical and DV 5 | per-generator tuned value |
| `--g-lr` | 0.005 | CV 0.005, classical and DV 0.05 | classical and DV under-trained |

Canonical lines (ring, 300 epochs, add `--seed S` and optionally `--critic ...`):

```bash
# CV, frozen configuration (186 parameters)
python train_2d_qgan.py --family ring --n-modes 3 --d-lr 0.0005 --gp-weight 10 --grid-size 32 --epochs 300
# classical, tuned (181 parameters)
python train_2d_qgan.py --family ring --n-modes 3 --generator classical --gen-activation tanh --g-lr 0.05 --d-lr 0.0005 --gp-weight 5 --grid-size 32 --epochs 300
# DV, tuned (180 parameters; qnncv-dv environment)
python train_2d_qgan.py --family ring --n-modes 3 --generator dv --dv-qubits 10 --dv-layers 9 --latent-scale 1.0 --g-lr 0.05 --d-lr 0.0005 --gp-weight 5 --grid-size 32 --epochs 300
# critics at parity
... --critic coord_2d
... --critic cv_qnn_2d
... --critic dv_qnn_2d --critic-layers 12
```

Status of the 9–14 Nov matrix against this table: the CV and classical cells of the corrected lists are right. **Every DV-generator cell of the 9–14 Nov lists ran with `--latent-scale 0.3` and latent dimension 4** and is an untuned DV arm; the corrected DV cells (`ring_matrix_dvfix_dv.txt`, remainder `dvfix2`, re-queue `dvfix3`) are running since 14 Nov. A drafted linter that checks every job list against this table lives outside the repository (`Tesis/agents_qc/drafts/joblist_lint.py`).

---

## 13. Known inconsistencies

1. `docs/backends.md` gives the point-wise score with the factor `G²/|S|`; the code uses `sqrt(G²/|S|)` (section 7.2 here follows the code).
2. `docs/backends.md` and the `cv_critic.py` docstring quote the CV critic at cutoff 10 (dimension 100). The class default is 10, but the trainer passes the generator's `--cutoff-dim`, so benchmark runs use cutoff 12 (dimension 144).
3. `--critic-layers` has one default (6) for both quantum critics, while the DV critic class itself defaults to 12. Through the CLI the DV critic gets 48 parameters unless `--critic-layers 12` is passed.
4. `CVQGANGenerator._init_weights` builds each layer's block as `[U1 | squeeze magnitudes | squeeze phases | U2 | displacement magnitudes | displacement phases | Kerr]`, but the circuit consumes squeeze and displacement parameters interleaved per mode (`r_0, phi_0, r_1, phi_1, ...`). With `active_sd = passive_sd = 0.1` every entry has the same distribution, so nothing changes numerically; it would matter only if the two standard deviations were set apart. The interleaved-squeeze binding uses the consumption order and is correct.
5. The non-CV generators reach the trainer through `GeneratorShim`, which reports ket norms of 1 instead of calling `regularizer(aux)`. Correct today because the classical and DV regularisers are zero; a future backend with a real regulariser needs the trainer changed.
6. The docstring of `Discriminator2D` still says the quantum generator has about 96 parameters and the critic about 25k; the benchmark generator has 186 and the critic 16,593 at 32×32.
7. Two registries are named `CRITIC_REGISTRY`: the one in `qgan2d_discriminator.py` (one entry, unused seam) and the live one in `src/backends/registry.py`.
8. `src/backends/base.py` and `registry.py` still say the trainer is "not wired to these yet"; it has been since 8 Nov (`--generator`, `--critic`).
9. Marginal W1 cannot see joint structure. The benchmark's headline metrics therefore also report MMD² and energy distances.

---

## 14. How to extend

**A new generator.** Write `src/backends/<name>.py` with a class implementing `GeneratorBackend` (section 4). Register it in `GENERATOR_REGISTRY` and add a toy factory in `registry.toy_factories()` so the contract suite covers it. Add a branch in `trainer_shim.build_generator` and the name to the `--generator` choices and to `VALUE_ABBREV` in `train_2d_qgan.py`. `rescore.py` rebuilds generators through the same builder.

**A new critic.** For a point-wise critic, subclass `PointwiseCritic` and implement `evaluate(points)`; scoring, point sampling and the trainer hook come for free. For a grid critic, implement `CriticBackend` directly. Register it, add a toy factory, a branch in `trainer_shim.build_critic` and the CLI choice.

**A new training objective** (for example an MMD loss for the Born-machine arm). Subclass `_BaseTrainer2D`, implement `discriminator_epoch` (returning the same statistics dictionary, zeros if there is no critic), `_g_real_batch` and `_combine_losses`, and select it in `train_2d_qgan` where `trainer_cls` is chosen. The loop, validation, checkpointing and outputs need no change.

**A new target family.** Subclass `DistributionFamily` with `sample()` and `get_canonical()`, register it in `src/families/registry.py` and add it to the `--family` choices. Densities must follow the `[y, x]` convention and sum to 1.

After any change that touches recorded configuration, the end-to-end goldens need an audited regeneration in their own commit.
