# Quantum Neural Networks for Computer Vision (QNNCV)

A research framework for implementing and evaluating quantum generative adversarial networks using continuous and discrete variable quantum computing paradigms.

## Overview

This project implements quantum generative adversarial networks (QGANs) that combine classical neural networks with quantum computing components. The framework supports multiple quantum computing paradigms including continuous variable (CV) quantum computing via Strawberry Fields and discrete variable (DV) quantum computing via PennyLane.

## Research Objectives

- Investigate quantum advantages in generative modeling tasks
- Compare classical and quantum GAN architectures
- Develop hybrid quantum-classical training methodologies
- Evaluate quantum circuit expressivity in adversarial learning

## Architecture

### Core Components

The framework implements a modular architecture supporting various generator and discriminator combinations:

- **Classical Components**: Traditional neural network implementations
- **Quantum Continuous Variable**: Photonic quantum computing using Strawberry Fields
- **Quantum Discrete Variable**: Qubit-based quantum computing using PennyLane
- **Hybrid Architectures**: Mixed classical-quantum configurations

### Mathematical Framework

The adversarial training follows the minimax objective:

```
min_G max_D V(D,G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

Where quantum components introduce additional considerations:
- Quantum parameter optimization requires specialized gradient handling
- Circuit depth affects expressivity and trainability
- Measurement strategies impact information extraction

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.13+
- NumPy, SciPy, Matplotlib
- scikit-learn for classical ML utilities

### Quantum Dependencies (Optional)

For full quantum functionality:

```bash
pip install strawberryfields  # Continuous variable quantum computing
pip install pennylane        # Discrete variable quantum computing
```

### Setup

```bash
git clone <repository-url>
cd QNNCV
pip install -r requirements.txt
```

## Usage

### Basic Training

```python
from src.training.qgan_trainer import QGAN
from src.models.generators.classical_generator import ClassicalGenerator
from src.models.discriminators.classical_discriminator import ClassicalDiscriminator
from src.utils.data_utils import load_synthetic_data

# Load or generate data
data = load_synthetic_data(dataset_type="spiral", num_samples=2000)

# Initialize components
generator = ClassicalGenerator(latent_dim=10, output_dim=2)
discriminator = ClassicalDiscriminator(input_dim=2)

# Create and train QGAN
qgan = QGAN(generator, discriminator)
history = qgan.train(data, epochs=100, batch_size=32)
```

### Quantum Components

```python
from src.models.generators.quantum_continuous_generator_enhanced import QuantumContinuousGeneratorEnhanced
from src.models.discriminators.quantum_continuous_discriminator import QuantumContinuousDiscriminator

# Quantum generator with continuous variables
quantum_gen = QuantumContinuousGeneratorEnhanced(n_qumodes=4, latent_dim=10)

# Quantum discriminator
quantum_disc = QuantumContinuousDiscriminator(n_qumodes=4, input_dim=4)

# Hybrid training
qgan = QGAN(quantum_gen, quantum_disc)
```

## Project Structure

```
QNNCV/
├── src/                          # Source code
│   ├── models/                   # Neural network implementations
│   │   ├── generators/           # Generator architectures
│   │   └── discriminators/       # Discriminator architectures
│   ├── training/                 # Training framework
│   └── utils/                    # Utilities and metrics
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── results/                  # Test outputs
├── data/                         # Dataset directory
├── docs/                         # Documentation
├── config/                       # Configuration files
└── requirements.txt              # Dependencies
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
