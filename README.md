# Quantum Neural Networks for Continuous Variables (QNNCV)

A research framework for implementing continuous variable quantum generative adversarial networks using Strawberry Fields with full Google Colab compatibility.

## Overview

This project implements quantum generative adversarial networks (QGANs) using continuous variable quantum computing. The framework provides working implementations of quantum generators and discriminators that leverage Strawberry Fields for quantum circuit simulation with automatic compatibility patches for modern environments.

## Current Status

**WORKING IMPLEMENTATION** with full compatibility:

- **Quantum Components**: QuantumSFGenerator and QuantumSFDiscriminator with verified gradient flow
- **Training Framework**: QGANSFTrainer with comprehensive monitoring and quality assessment
- **Google Colab Ready**: Automatic compatibility patches for NumPy 2.0+, SciPy 1.15+, TensorFlow 2.18+
- **Local Development**: Full local setup with CPU/GPU support
- **Tutorial Notebooks**: Step-by-step examples for training quantum GANs

## Recent Breakthroughs

**GRADIENT FLOW PROBLEM SOLVED** (Major Achievement):
- **Individual tf.Variables**: Fixed gradient counting by using individual tf.Variable for each quantum parameter
- **Full Gradient Flow**: Achieved 92%+ gradient flow (50/54 discriminator, 46/50 generator) vs previous 4/32
- **Quantum-First Architecture**: Each quantum parameter now has its own gradient object for optimal training

**DIMENSIONAL ADAPTER SYSTEM** (New Feature):
- **Static Adapters**: Zero-gradient dimensional transformations between N-D data and M quantum modes
- **Flexible Architecture**: Train with any number of quantum modes regardless of data dimensionality
- **Multiple Methods**: Linear projection, padding, truncation, and repeat strategies
- **Backward Compatible**: Maintains all existing functionality while adding flexibility

**EXPERIMENTAL FEATURES** (early research stage):

- **Quantum Encoding Strategies**: 5 experimental approaches for classical-to-quantum data encoding
- **Enhanced Generator Architecture**: Extended SF-based generator with configurable encoding strategies
- **Infrastructure Validation**: Basic testing framework for systematic component validation
- **Configuration Management**: Early-stage config system for research parameter management

*Note: Experimental features are in active development and may require refinement for production use.*

## Installation

### Google Colab (Recommended)

```python
# Clone repository
!git clone https://github.com/your-repo/QNNCV.git
%cd QNNCV

# Run modern setup (handles all compatibility automatically)
!python setup/setup_colab_modern.py

# Import and use
import sys
sys.path.append('/content/QNNCV/src')
import utils  # Auto-applies compatibility patches

from models.generators.quantum_sf_generator import QuantumSFGenerator
from models.discriminators.quantum_sf_discriminator import QuantumSFDiscriminator
from training.qgan_sf_trainer import QGANSFTrainer
```

### Local Development

```bash
git clone https://github.com/your-repo/QNNCV.git
cd QNNCV
python setup/setup_local.py
```

## Compatibility Features

### Automatic Compatibility Patches

The framework includes automatic compatibility fixes for:

- **SciPy 1.15+**: Automatic `simps` → `simpson` function patching
- **NumPy 2.0+**: Restored deprecated aliases (`np.bool`, `np.int`, etc.)
- **TensorFlow 2.18+**: Mixed precision and GPU configuration
- **Strawberry Fields**: Import hooks for seamless integration

### Environment Support

- **Google Colab**: Full GPU support with automatic setup
- **Local Development**: CPU/GPU with conda/pip environments
- **Cross-Platform**: Windows, macOS, Linux compatibility

## Quick Start

### Basic Training Example

```python
# Automatic compatibility (import utils first)
import utils

from models.generators.quantum_sf_generator import QuantumSFGenerator
from models.discriminators.quantum_sf_discriminator import QuantumSFDiscriminator
from training.qgan_sf_trainer import QGANSFTrainer
import numpy as np

# Generate sample data
data = np.random.normal(0, 1, (1000, 2))

# Create quantum components
generator = QuantumSFGenerator(n_modes=2, latent_dim=2, layers=2, cutoff_dim=8)
discriminator = QuantumSFDiscriminator(n_modes=1, input_dim=2, layers=1, cutoff_dim=8)

# Create trainer
trainer = QGANSFTrainer(generator, discriminator, latent_dim=2)

# Train
history = trainer.train(data, epochs=100, batch_size=16)
```

### Tutorial Notebooks

1. **tutorials/minimal_sf_qgan.ipynb**: Basic 20-epoch training example
2. **tutorials/extended_sf_qgan_training.ipynb**: Comprehensive 100+ epoch training with monitoring
3. **tutorials/wide_sf_qgan_training.ipynb**: M-mode quantum circuits with dimensional adapters
4. **tutorials/complete_cv_sf_qgan_template.ipynb**: Google Colab template with full setup

### Dimensional Adapter Example

```python
# Train 4-mode quantum circuits with 2D data using static adapters
import utils
from models.generators.quantum_sf_generator import QuantumSFGenerator
from models.discriminators.quantum_sf_discriminator import QuantumSFDiscriminator

# Static dimensional adapter (no trainable parameters)
class StaticDimensionalAdapter:
    def __init__(self, input_dim, output_dim, method='linear', seed=42):
        self.input_dim = input_dim
        self.output_dim = output_dim
        if method == 'linear' and input_dim != output_dim:
            np.random.seed(seed)
            self.projection_matrix = tf.constant(
                np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / (input_dim + output_dim)),
                dtype=tf.float32
            )
    
    def __call__(self, x):
        return tf.matmul(x, self.projection_matrix)

# Create 4-mode quantum components
quantum_gen = QuantumSFGenerator(n_modes=4, latent_dim=4, layers=1)
quantum_disc = QuantumSFDiscriminator(n_modes=4, input_dim=4, layers=1)

# Create static adapters (zero gradients)
quantum_to_data = StaticDimensionalAdapter(4, 2)  # 4D quantum → 2D data
data_to_quantum = StaticDimensionalAdapter(2, 4)  # 2D data → 4D quantum

# Wrap with adapters
class AdaptedGenerator:
    def __init__(self, quantum_gen, adapter):
        self.quantum_gen = quantum_gen
        self.adapter = adapter
    
    def generate(self, z):
        return self.adapter(self.quantum_gen.generate(z))
    
    @property
    def trainable_variables(self):
        return self.quantum_gen.trainable_variables  # Only quantum gradients!

generator = AdaptedGenerator(quantum_gen, quantum_to_data)
# Result: 4-mode quantum processing with 2D data compatibility
```

## Project Structure

```
QNNCV/
├── setup/                        # Setup and compatibility
│   ├── setup_colab_modern.py     # Modern Colab setup (ACTIVE)
│   ├── setup_local.py            # Local development setup (ACTIVE)
│   └── environment.yml           # Conda environment
├── src/                          # Production code
│   ├── models/
│   │   ├── generators/           # QuantumSFGenerator
│   │   └── discriminators/       # QuantumSFDiscriminator  
│   ├── training/                 # QGANSFTrainer
│   ├── quantum_encodings/        # Experimental encoding strategies (5 types)
│   ├── config/                   # Configuration management (experimental)
│   └── utils/                    # Compatibility patches, utilities
│       ├── __init__.py           # Auto-applies compatibility
│       ├── compatibility.py      # SciPy/NumPy/TF patches
│       ├── quantum_metrics.py    # Quantum-specific metrics (experimental)
│       ├── gpu_memory_manager.py # Resource management (experimental)
│       └── scipy_compat.py       # SciPy integration fixes
├── tutorials/                    # Working examples and notebooks
├── legacy/                       # Deprecated implementations
│   └── setup/                    # Old setup scripts
├── tests/                        # Test suite
├── data/                         # Dataset directory
└── config/                       # Configuration files
```

## Quantum Circuit Implementation

### Continuous Variable Architecture

```
Classical Input → Quantum Encoding → Variational Layers → Measurement → Classical Output
```

**Key Components:**
- **Displacement Gates**: Encode classical data into quantum states
- **Squeezing Gates**: Create quantum correlations and nonlinearity  
- **Interferometers**: Enable quantum mode coupling and entanglement
- **Homodyne Detection**: Extract classical information from quantum states

**Circuit Depth**: Configurable layers with automatic gradient computation

### Hybrid Quantum-Classical Training

- **Quantum Forward Pass**: Strawberry Fields TensorFlow backend
- **Classical Optimization**: Adam optimizer with quantum-aware learning rates
- **Automatic Differentiation**: End-to-end gradient computation through quantum circuits

### Individual Parameter Architecture

**Breakthrough Design:**
- **Individual tf.Variables**: Each quantum parameter (displacement, squeezing, interferometer, Kerr) has its own tf.Variable
- **Gradient Visibility**: TensorFlow optimizer sees each parameter separately for optimal training
- **Backward Compatibility**: Dynamic `quantum_weights` property maintains API compatibility

**Parameter Breakdown (4-mode, 1-layer example):**
```
Interferometer 1: 15 individual parameters → 15 gradients
Squeezing:        4 individual parameters → 4 gradients  
Interferometer 2: 15 individual parameters → 15 gradients
Displacement r:   4 individual parameters → 4 gradients
Displacement φ:   4 individual parameters → 4 gradients
Kerr:            4 individual parameters → 4 gradients
Total:           46 quantum gradients + 4 classical = 50 total gradients
```

**Result**: 92%+ gradient flow vs previous 12% (4/32) with single quantum variable approach.

## Performance Optimization

### GPU Acceleration

- **TensorFlow Operations**: Automatic GPU utilization for classical components
- **Quantum Simulation**: CPU-optimized for Strawberry Fields operations
- **Hybrid Execution**: Optimal device placement for quantum-classical workflows

### Training Efficiency

- **Batch Processing**: Efficient quantum circuit execution
- **Memory Management**: Automatic GPU memory growth
- **Gradient Clipping**: Prevents quantum parameter divergence

## Troubleshooting

### Common Issues

**Import Errors with SciPy/Strawberry Fields:**
```python
# Solution: Import utils first
import utils  # This applies all compatibility patches
import strawberryfields as sf  # Now works correctly
```

**Mixed Precision Tensor Errors:**
```python
# Solution: Disable mixed precision
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('float32')
```

**GPU Not Detected:**
- Check Colab runtime: Runtime → Change runtime type → GPU
- Verify TensorFlow GPU installation: `tf.config.list_physical_devices('GPU')`

### Performance Tips

1. **Start Small**: Use minimal parameters for initial testing
   - `n_modes=1`, `layers=1`, `cutoff_dim=4`
   - Small batch sizes (`batch_size=4`)
   - Few training samples (`data[:50]`)

2. **Scale Gradually**: Increase complexity after verifying basic functionality

3. **Monitor Resources**: Quantum simulations are computationally intensive

## Testing

```bash
# Run all tests
python -m pytest tests/

# Test specific components
python -m pytest tests/unit/test_basic.py
python -m pytest tests/integration/test_hybrid_qgan.py
```

## Configuration

Training parameters via `config/config.yaml`:

```yaml
training:
  epochs: 100
  batch_size: 16
  latent_dim: 2

generator:
  n_modes: 2
  layers: 2
  cutoff_dim: 8

discriminator:
  n_modes: 1
  layers: 1
  cutoff_dim: 8

optimizer:
  learning_rate: 0.001
  beta1: 0.5
  beta2: 0.999
```

## System Requirements

### Minimum Requirements
- **RAM**: 8GB
- **Python**: 3.8+
- **CPU**: Modern processor with AVX support

### Recommended for Training
- **RAM**: 16GB+
- **GPU**: CUDA-compatible for TensorFlow acceleration
- **Environment**: Google Colab Pro for extended training sessions

## Contributing

Focus areas for contributions:

- **Quantum Architectures**: New circuit designs and parameterizations
- **Training Methodologies**: Improved optimization strategies for quantum parameters
- **Evaluation Metrics**: Quantum-specific quality measures
- **Performance Optimization**: Faster quantum simulation techniques
- **Documentation**: Tutorials and examples

## Version History

- **v0.0**: Initial implementation with basic quantum components
- **v0.1**: Added comprehensive training framework
- **v0.2**: Google Colab compatibility and automatic setup

## License

Academic and research use. See LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

## Support

- **Documentation**: Check tutorial notebooks and setup guides
- **Issues**: Submit via GitHub issues
- **Compatibility**: Run setup scripts for automatic environment configuration
