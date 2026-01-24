# Generators - src/models/generators/

Quantum circuit generators for CV-QNN GANs.

## Active Files

| File | Class | Description | Used By |
|------|-------|-------------|---------|
| `killoran_cvqnn.py` | `KilloranCVQNN` | Killoran architecture with Kerr gate. Outputs probability distribution via Hermite transform. | `train_killoran_qgan.py` |
| `quantum_distribution_generator.py` | `QuantumDistributionGenerator` | Alternative distribution-based generator using Hermite polynomials. | `train_distribution_qgan.py` |
| `quantum_sf_generator.py` | `QuantumSFGenerator` | Sample-based generator producing point samples from homodyne measurement. | `train_qgan.py` (legacy) |

## Key Architecture: KilloranCVQNN

```
Layer L := Rotation → Squeeze → Rotation → Displacement → Kerr

Per layer (7 parameters):
- θ₁: First rotation angle
- r, φ: Squeeze magnitude and phase  
- θ₂: Second rotation angle
- α_r, α_φ: Displacement magnitude and phase
- κ: Kerr nonlinearity (CRITICAL for multi-modal)
```

## Usage

```python
from src.models.generators import KilloranCVQNN

gen = KilloranCVQNN(
    n_layers=6,
    cutoff_dim=10,
    use_kerr=True  # Essential for non-Gaussian
)

# Generate probability distribution
prob = gen.generate(z_latent)  # Shape: (batch, 100)
```

## Important Notes

1. **Kerr gate is essential** for multi-modal distributions
2. **cutoff_dim** limits Fock space - higher = more expressive but slower
3. Output is probability distribution (100 bins), not samples

## Legacy Files (in src/models/legacy/)

- `clusterized_quantum_generator.py` - Cluster-conditional approach (abandoned)
- `constellation_sf_generator.py` - Constellation point approach
- `coordinate_quantum_generator.py` - Coordinate-based generator
- `optimal_constellation_generator.py` - Over-engineered constellation
- `pure_sf_generator.py` - Pure SF approach
- `sf_tutorial_generator.py` - Based on SF tutorial
