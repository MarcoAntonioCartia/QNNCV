# Quantum Neural Networks for Continuous Variables (QNNCV)

A research framework for implementing continuous variable quantum generative adversarial networks using Strawberry Fields.

## Overview

This project implements quantum generative adversarial networks (QGANs) using continuous variable quantum computing. The framework provides working implementations of quantum generators and discriminators that leverage Strawberry Fields for quantum circuit simulation and automatic differentiation.

## Current Status

The repository contains a functional quantum GAN implementation with:

- **Working quantum components**: QuantumSFGenerator and QuantumSFDiscriminator with verified gradient flow
- **Training framework**: QGANSFTrainer with comprehensive monitoring and quality assessment
- **Tutorial notebooks**: Step-by-step examples for training quantum GANs
- **Google Colab support**: Template for cloud-based training

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.8+
- Strawberry Fields
- NumPy, SciPy, Matplotlib

### Setup

```bash
git clone <repository-url>
cd QNNCV
pip install -r requirements.txt
```

### Key Dependencies

```bash
pip install strawberryfields tensorflow numpy scipy matplotlib
```

## Quick Start

### Basic Usage

```python
from src.models.generators.quantum_sf_generator import QuantumSFGenerator
from src.models.discriminators.quantum_sf_discriminator import QuantumSFDiscriminator
from src.training.qgan_sf_trainer import QGANSFTrainer

# Create quantum components
generator = QuantumSFGenerator(n_modes=2, latent_dim=2, layers=2, cutoff_dim=8)
discriminator = QuantumSFDiscriminator(n_modes=1, input_dim=2, layers=1, cutoff_dim=8)

# Create trainer
trainer = QGANSFTrainer(generator, discriminator, latent_dim=2)

# Train on your data
history = trainer.train(data, epochs=100, batch_size=16)
```

### Tutorial Notebooks

1. **tutorials/minimal_sf_qgan.ipynb**: Basic 20-epoch training example
2. **tutorials/extended_sf_qgan_training.ipynb**: Comprehensive 100+ epoch training with monitoring
3. **tutorials/complete_cv_sf_qgan_template.ipynb**: Google Colab template

## Project Structure

```
QNNCV/
├── src/                          # Production code
│   ├── models/
│   │   ├── generators/           # QuantumSFGenerator
│   │   └── discriminators/       # QuantumSFDiscriminator  
│   ├── training/                 # QGANSFTrainer
│   └── utils/                    # Warning suppression, utilities
├── tutorials/                    # Working examples and notebooks
├── legacy/                       # Deprecated implementations
│   ├── generators/               # Old generators with broken gradients
│   ├── discriminators/           # Old discriminators  
│   ├── utils/                    # Fake gradient implementations
│   └── test_files/               # Historical test files
├── tests/                        # Test suite
├── data/                         # Dataset directory
└── config/                       # Configuration files
```

## Quantum Circuit Implementations

### Continuous Variable Quantum Computing

The framework implements sophisticated CV quantum circuits:

```
Input → Classical Encoding → Displacement Gates → Squeezing Layer → 
Interferometer → Additional Squeezing → Adaptive Measurements → Output
```

Key quantum operations:
- **Displacement gates**: Encode classical information into quantum states
- **Squeezing gates**: Create quantum correlations and nonlinearity
- **Interferometers**: Enable quantum mode coupling and entanglement
- **Homodyne measurements**: Extract classical information from quantum states

### Discrete Variable Quantum Computing

Qubit-based implementations using parameterized quantum circuits:

```
Input Encoding → Parameterized Rotations → Entangling Gates → 
Additional Layers → Measurements → Classical Post-processing
```

## Training Methodology

### Quantum-Aware Training

The framework includes specialized training features for quantum components:

- **Gradient Clipping**: Prevents parameter divergence in quantum circuits
- **Learning Rate Scheduling**: Accommodates quantum parameter optimization
- **Stability Monitoring**: Detects training instabilities specific to quantum systems
- **Adaptive Optimization**: Adjusts training dynamics based on quantum circuit behavior

### Loss Functions

Multiple loss formulations are supported:

1. **Traditional GAN Loss**: Binary cross-entropy with label smoothing
2. **Wasserstein Loss**: With gradient penalty for improved stability

## Evaluation Metrics

The framework provides comprehensive evaluation tools:

- **Wasserstein Distance**: Distribution similarity measurement
- **Maximum Mean Discrepancy (MMD)**: Statistical distance between distributions
- **Coverage and Precision**: Quality and diversity metrics
- **Frechet Distance**: Distribution comparison in feature space

## Testing

Run the complete test suite:

```bash
# All tests
python -m pytest tests/

# Unit tests only
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/
```

## Configuration

Training parameters can be configured via `config/config.yaml`:

```yaml
training:
  epochs: 100
  batch_size: 32
  latent_dim: 10

components:
  generator:
    type: "quantum_continuous"
    n_qumodes: 4
  discriminator:
    type: "classical"
    hidden_units: 64

optimizer:
  learning_rate: 0.0001
  beta_1: 0.5
```

## Research Applications

This framework is designed for research in:

- Quantum machine learning
- Generative modeling with quantum circuits
- Hybrid quantum-classical algorithms
- Quantum advantage investigations
- Variational quantum algorithms

## Contributing

This is a research framework. Contributions should focus on:

- Novel quantum circuit architectures
- Improved training methodologies
- Additional evaluation metrics
- Performance optimizations
- Documentation improvements

## Technical Requirements

### Minimum System Requirements

- 8GB RAM
- Modern CPU with AVX support
- Python 3.8+ environment

### Recommended for Quantum Simulations

- 16GB+ RAM
- Multi-core CPU
- GPU support for TensorFlow (optional)

## Limitations

- Quantum simulations are computationally intensive
- Circuit depth limited by classical simulation capabilities
- Quantum hardware integration requires additional setup
- Some quantum dependencies may have platform-specific requirements

## References

This framework builds upon established quantum computing and machine learning research:

- Quantum computing frameworks: Strawberry Fields, PennyLane
- Generative adversarial networks: Original GAN formulation and variants
- Quantum machine learning: Variational quantum algorithms and quantum neural networks

## License

This project is intended for academic and research use. Please refer to the license file for specific terms and conditions.

## Contact

For questions regarding this research framework, please refer to the project documentation or submit issues through the appropriate channels.
