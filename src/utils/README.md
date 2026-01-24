# Utils - src/utils/

Utility functions for CV-QNN development.

## Active Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `compatibility.py` | NumPy/SciPy/TF version fixes | `apply_all_compatibility_patches()` |
| `scipy_compat.py` | SciPy `simps→simpson` fix | Auto-applied on import |
| `tensorflow_compat.py` | TensorFlow compatibility | TF version patches |
| `warning_suppression.py` | Clean console output | `suppress_warnings()` |
| `visualization.py` | Plotting functions | `plot_comparison()`, `plot_training_curves()` |
| `monitoring.py` | Training metrics | `TrainingMonitor` |
| `killoran_init.py` | Hermite polynomial functions | `hermite_prob_distribution()` |
| `import_checker.py` | Import diagnostics | `check_all_imports()` |
| `data_utils.py` | Data utilities | Data loading helpers |
| `tensor_utils.py` | Tensor operations | Shape utilities |

## Critical: compatibility.py

**Always import at the start of any script:**

```python
from src.utils.compatibility import apply_all_compatibility_patches
apply_all_compatibility_patches()
```

This fixes:
- `scipy.integrate.simps` → `simpson` (deprecated)
- NumPy 2.0 compatibility issues
- TensorFlow eager/graph mode issues

## Hermite Polynomials (killoran_init.py)

Core function for computing position probability from Fock state:

```python
from src.utils.killoran_init import hermite_prob_distribution

# state: (batch, cutoff_dim) complex Fock amplitudes
# xvec: (num_bins,) position values
prob = hermite_prob_distribution(state, xvec)
# prob: (batch, num_bins) probability distribution
```

**Physics:**
```
ψ_n(x) = H_n(x) * exp(-x²/2) / sqrt(2^n * n! * sqrt(π))
P(x) = |Σ c_n * ψ_n(x)|²
```

## Visualization Functions

```python
from src.utils.visualization import plot_comparison, plot_training_curves

# Compare distributions
plot_comparison(target_prob, generated_prob, xvec, save_path='comparison.png')

# Plot training history
plot_training_curves(history, save_path='training.png')
```

## Warning Suppression

For clean training output:

```python
from src.utils.warning_suppression import setup_clean_training

setup_clean_training()  # Suppresses TF, SF, NumPy warnings
```

## Legacy Files (in src/utils/legacy/)

These were useful during development but are no longer actively used:

| File | Original Purpose |
|------|-----------------|
| `batch_processor.py` | Batch processing utilities |
| `bimodal_visualization_tracker.py` | Early bimodal tracking |
| `cluster_analyzer.py` | Cluster analysis (approach abandoned) |
| `enhanced_quantum_circuit_visualizer.py` | Circuit visualization |
| `gpu_memory_manager.py` | GPU memory utilities |
| `gradient_manager.py` | Gradient debugging |
| `quantum_circuit_visualizer.py` | Basic circuit vis |
| `quantum_forensics.py` | Deep debugging tools |
| `quantum_gan_monitor.py` | Old monitoring |
| `quantum_metrics.py` | Old metrics (superseded) |
| `quantum_mode_diagnostics.py` | Mode collapse debugging |
| `quantum_state_visualizer.py` | State visualization |
| `quantum_training_health_checker.py` | Health checks |
| `spectral_normalization.py` | Spectral norm layers |

## Import Order

Recommended import order for new scripts:

```python
# 1. Standard library
import os
import numpy as np

# 2. Compatibility patches (MUST be early)
from src.utils.compatibility import apply_all_compatibility_patches
apply_all_compatibility_patches()

# 3. TensorFlow (after patches)
import tensorflow as tf

# 4. Strawberry Fields
import strawberryfields as sf

# 5. Project modules
from src.models.generators import KilloranCVQNN
from src.training.killoran_trainer import KilloranQGANTrainer
```
