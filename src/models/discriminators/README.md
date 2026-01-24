# Discriminators - src/models/discriminators/

Classical neural network discriminators for CV-QNN GANs.

## Active Files

| File | Class | Description | Used By |
|------|-------|-------------|---------|
| `distribution_discriminator.py` | `DistributionDiscriminator` | MLP that takes probability distribution (100 bins) as input. WGAN-GP compatible. | `train_killoran_qgan.py` |
| `classical_discriminator.py` | `ClassicalDiscriminator` | Simple MLP for point samples. | `train_qgan.py` |
| `quantum_sf_discriminator.py` | `QuantumSFDiscriminator` | Discriminator for sample-based training. | Legacy training |

## Key Architecture: DistributionDiscriminator

```python
Input: (batch, 100) probability distribution
  ↓
Dense(128) → LeakyReLU → Dropout(0.3)
  ↓
Dense(64) → LeakyReLU → Dropout(0.3)
  ↓
Dense(32) → LeakyReLU
  ↓
Dense(1) → Linear (no sigmoid for WGAN)
```

## Usage

```python
from src.models.discriminators import DistributionDiscriminator

disc = DistributionDiscriminator(
    input_dim=100,        # Number of bins
    hidden_dims=[128, 64, 32]
)

# Get critic score (not probability)
score = disc(prob_distribution)  # Shape: (batch, 1)
```

## Important Notes

1. **No sigmoid activation** on output (required for WGAN)
2. Input dimension must match generator output bins
3. Gradient penalty applied during training (WGAN-GP)

## WGAN-GP Training

```python
# Gradient penalty calculation
with tf.GradientTape() as gp_tape:
    gp_tape.watch(interpolated)
    pred = discriminator(interpolated)
grads = gp_tape.gradient(pred, interpolated)
gradient_penalty = tf.reduce_mean((tf.norm(grads, axis=1) - 1) ** 2)
```

## Legacy Files (in src/models/legacy/)

- `pure_sf_discriminator.py` - Pure SF approach
- `pure_sf_discriminator_multi.py` - Multi-mode version
- `quantum_discriminator.py` - Old quantum-aware discriminator
- `sf_tutorial_discriminator.py` - Tutorial-based
- `spectral_sf_discriminator.py` - With spectral normalization
