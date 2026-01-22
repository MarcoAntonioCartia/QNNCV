# CV Quantum GAN Training Analysis
## Date: January 23, 2026 - Post-1500 Epoch Training Run

---

## Executive Summary

The 1500-epoch training run revealed **fundamental architectural and training dynamics issues** that prevented the quantum generator from learning the target distribution N(2.0, 0.5²). The model achieved its best Wasserstein distance of 1.745 at epoch 137, then diverged catastrophically to 5.13 by epoch 1500.

**Key Finding**: The generator is severely underpowered compared to the discriminator, suffers from vanishing gradients, and lacks the expressivity to shift from vacuum state (mean≈0) to the target (mean=2.0).

---

## Observed Training Dynamics

### Phase 1: Initial Learning (Epochs 0-137)
- Generator shows some learning, Wasserstein distance decreases to 1.745
- Generated mean hovers around 0, never approaching target of 2.0
- Wigner function shows vacuum-like state centered at origin

### Phase 2: Stagnation (Epochs 137-500)
- No improvement in mean matching
- Standard deviation slowly increases from 0.1 to 0.3
- Generator gradient norm drops to very low values (~0.01-0.1)
- Discriminator maintains strong gradients (~1-10)

### Phase 3: Divergence (Epochs 500-1500)
- Generated mean drifts increasingly negative (-1, -2, -3...)
- Standard deviation explodes (0.5 → 1.2)
- Discriminator loss spikes (2 → 11), indicating confusion
- Wasserstein distance monotonically increases (1.7 → 5.1)

---

## Root Cause Analysis

### Issue 1: Massive Parameter Imbalance

| Component | Parameters | Ratio |
|-----------|------------|-------|
| Generator (QNN weights) | 6 | 1x |
| Generator (decoder) | 1 | - |
| **Total Generator** | **7** | **1x** |
| Discriminator (MLP) | **1153** | **165x** |

The discriminator has **165x more parameters** than the generator. This creates an adversarial imbalance where the discriminator easily overwhelms the generator.

### Issue 2: Initialization Scale Problem

```python
def init_weights(...):
    active_sd: float = 0.0001,  # Squeeze/Displacement magnitude
    passive_sd: float = 0.1     # Beamsplitter/Rotation
```

The displacement magnitudes start at scale 0.0001 - essentially zero. The quantum state begins at vacuum (x-quadrature mean ≈ 0) and the tiny initialization means:
- Initial gradient signals are weak
- Large number of gradient steps needed to shift mean from 0 to 2

### Issue 3: Insufficient Circuit Expressivity

With **1 QNN layer**, the circuit has limited ability to transform states:

```
Input Encoding (Dgate) → Squeeze → Displacement → (no Beamsplitter for 1 mode) → Rotation → Measurement
```

For 1 mode, the beamsplitter is skipped (requires 2+ modes). The single-mode circuit becomes:
```
Dgate(z) → Sgate(θ₁,θ₂) → Dgate(θ₃,θ₄) → Rgate(θ₅)
```

This only has 5 trainable parameters affecting the output. The rotation gate at the end doesn't affect x-quadrature expectation values!

### Issue 4: Cutoff Dimension Limitation

With `cutoff_dim=6`, the Fock space truncation may cause:
- Numerical instability for larger displacements
- Poor representation of coherent states with |α| > 2

For target mean=2.0, we need displacement α ≈ 2.0 (since ⟨x⟩ = √2·Re(α) for coherent states). A cutoff of 6 can represent this but is borderline.

### Issue 5: Learning Rate Imbalance

Both networks use `lr=0.001`, but given the parameter imbalance:
- Discriminator learns 165x faster (more parameters, same LR)
- Generator gradients vanish as D becomes too strong

### Issue 6: Wigner Function Evidence

The Wigner functions across epochs show:

| Epoch | Center Position | Interpretation |
|-------|-----------------|----------------|
| 50 | (0, 0) | Vacuum state |
| 100 | (0, 0) | Still vacuum |
| 500 | (0, 0) | No displacement learned |
| 1000 | (~1, 0) | Slight positive displacement |

The quantum state never achieves the displacement needed (x ≈ 2.8 for mean=2.0).

---

## Quantitative Evidence

### From Training Logs

| Metric | Epoch 100 | Epoch 1000 | Epoch 1500 |
|--------|-----------|------------|------------|
| G Loss | 0.0347 | -0.0643 | -2.6390 |
| D Loss | 2.6031 | 2.3061 | 4.8682 |
| Gen Mean | 0.0137 | -0.0223 | -3.1510 |
| Gen Std | 0.1099 | 0.3409 | 1.2376 |
| W₁ Distance | 1.8263 | 2.0096 | 5.1315 |
| G Grad Norm | 0.091 | 0.441 | 1.585 |
| D Grad Norm | 3.083 | 1.654 | 0.526 |

### Key Observations

1. **Generator loss becoming negative** doesn't mean success - it means the generator found a way to "fool" the discriminator by exploiting its weaknesses, not by matching the real distribution.

2. **Gradient norm ratio** (D/G) goes from 34:1 at epoch 100 to 0.33:1 at epoch 1500 - the training dynamics completely inverted as the system became unstable.

3. **Wasserstein distance** never went below 1.745, indicating the generator never truly approximated the target distribution.

---

## Proposed Solutions

### Immediate Fixes (High Priority)

#### 1. Increase Generator Capacity
```python
# Current
n_layers = 1
n_modes = 1

# Proposed  
n_layers = 3  # More expressivity
n_modes = 2   # Enables beamsplitter entanglement
```

#### 2. Fix Initialization Scale
```python
def init_weights(n_modes, n_layers, active_sd=0.1, passive_sd=0.1):
    # Increase active_sd from 0.0001 to 0.1
    # This gives meaningful initial displacement/squeezing
```

#### 3. Increase Cutoff Dimension
```python
cutoff_dim = 12  # Up from 6, better for displaced states
```

### Training Dynamics Fixes

#### 4. Learning Rate Rebalancing
```python
g_lr = 0.01   # 10x current
d_lr = 0.0001 # 10x slower than current
```

#### 5. Critic Updates
```python
n_critic = 5  # Update D 5 times per G update
```

#### 6. Add Mean-Matching Regularization
```python
def generator_loss(fake_scores, fake_samples, target_mean):
    adv_loss = -tf.reduce_mean(fake_scores)
    mean_reg = tf.square(tf.reduce_mean(fake_samples) - target_mean)
    return adv_loss + 0.5 * mean_reg  # Guide toward correct mean
```

### Architectural Improvements

#### 7. Bias Initialization Toward Target
Initialize the decoder with a bias toward the target mean:
```python
self.decoder = tf.Variable(
    tf.constant([[2.0]]),  # Initialize near target
    name="decoder",
    trainable=True
)
```

#### 8. Use Spectral Normalization on Discriminator
Constrain discriminator Lipschitz constant without gradient penalty:
```python
from tensorflow.keras.layers import SpectralNormalization
```

---

## Recommended Next Steps

1. **Short-term**: Apply fixes #2 (initialization) and #4 (LR balance) - minimal code change
2. **Medium-term**: Increase layers (#1) and add mean regularization (#6)
3. **Long-term**: Consider multi-mode architecture (#1 with n_modes=2+) for more expressivity

---

## Commit Message

```
feat(training): Complete 1500-epoch training run - identify convergence failure

Training Results:
- Best Wasserstein: 1.745 at epoch 137
- Final Wasserstein: 5.13 (diverged)
- Generated mean: -3.15 (target: 2.0)
- Generated std: 1.24 (target: 0.5)

Root Causes Identified:
1. Generator capacity: 7 params vs discriminator's 1153 (165:1 imbalance)
2. Initialization scale: active gates at 0.0001 (too small)
3. Single QNN layer lacks expressivity for mean shift
4. Learning rates equal despite parameter imbalance
5. Cutoff dim=6 borderline for target displacement

Training phases observed:
- Epochs 0-137: Partial learning (W₁ improved to 1.745)
- Epochs 137-500: Stagnation (vanishing generator gradients)
- Epochs 500-1500: Divergence (mean drifted negative, D loss exploded)

Next: Increase n_layers, fix initialization, rebalance learning rates.

See docs/QGAN_TRAINING_ANALYSIS_20260123.md for full analysis.
```

---

## Appendix: Wigner Function Interpretation

The Wigner function W(x, p) provides phase-space visualization of the quantum state:

- **Vacuum state**: Circular Gaussian centered at (0, 0)
- **Coherent state |α⟩**: Circular Gaussian centered at (√2·Re(α), √2·Im(α))
- **Squeezed state**: Elliptical Gaussian

For target N(2.0, 0.5²), we need a coherent state with Re(α) ≈ 1.41 (since ⟨x⟩ = √2·Re(α) = 2.0).

The observed Wigner functions show the state remained near vacuum throughout training, never achieving the required displacement.

---

*Analysis by Claude | QNNCV Project | January 23, 2026*
