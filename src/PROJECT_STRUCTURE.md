# Quantum GAN Project Structure - Clean Architecture

## Overview
This project implements a modular Quantum GAN architecture with pure quantum learning and gradient flow through quantum circuits.

## Directory Structure

```
QNNCV/
├── src/                        # Main source code
│   ├── quantum/               # Core quantum modules
│   │   ├── core/             # Quantum circuit implementation
│   │   │   └── quantum_circuit.py
│   │   ├── parameters/       # Gate parameter management
│   │   │   └── gate_parameters.py
│   │   ├── builders/         # Circuit building logic
│   │   │   └── circuit_builder.py
│   │   └── measurements/     # Measurement extraction
│   │       └── measurement_extractor.py
│   │
│   ├── models/               # Model implementations
│   │   ├── quantum_gan.py    # Main QGAN orchestrator
│   │   ├── generators/       # Generator implementations
│   │   │   └── quantum_generator.py
│   │   ├── discriminators/   # Discriminator implementations
│   │   │   └── quantum_discriminator.py
│   │   └── transformations/  # Static transformation matrices
│   │       └── matrix_manager.py
│   │
│   ├── losses/               # Loss functions
│   │   └── quantum_gan_loss.py  # Measurement-based losses
│   │
│   ├── examples/             # Training examples
│   │   ├── train_modular_qgan.py
│   │   ├── train_quantum_gan_measurement_loss.py
│   │   └── train_simple_quantum_gan.py
│   │
│   ├── utils/                # Utility functions
│   │   ├── compatibility.py
│   │   ├── data_utils.py
│   │   ├── visualization.py
│   │   └── warning_suppression.py
│   │
│   └── config/               # Configuration files
│       └── quantum_gan_config.py
│
├── tests/                     # Test directory (currently empty)
│   ├── benchmarks/           # Performance benchmarks
│   ├── logs/                 # Test logs and documentation
│   └── results/              # Test results
│
├── results/                   # Training results and outputs
│   ├── modular_architecture/ # Results from modular implementation
│   └── quantum_gan_results_*/# Training run results
│
├── logs/                      # Project documentation and logs
│   └── *.md                  # Various documentation files
│
├── legacy/                    # Old implementations (archived)
│   ├── generators/           # Old generator implementations
│   ├── discriminators/       # Old discriminator implementations
│   ├── test_files/           # Old test files
│   └── utils/                # Old utility functions
│
├── data/                      # Dataset directory
│   └── qm9/                  # QM9 molecular dataset
│
├── tutorials/                 # Jupyter notebook tutorials
├── scripts/                   # Utility scripts
├── setup/                     # Setup and installation files
└── config/                    # Configuration files
```

## Key Components

### 1. Quantum Core (`src/quantum/`)
- **PureQuantumCircuit**: Base quantum circuit with gradient support
- **GateParameterManager**: Manages trainable quantum parameters
- **CircuitBuilder**: Builds quantum circuits with proper structure
- **MeasurementExtractor**: Extracts quantum measurements (raw/holistic)

### 2. Models (`src/models/`)
- **QuantumGAN**: Main orchestrator combining generator and discriminator
- **PureQuantumGenerator**: Generates samples using quantum circuits
- **PureQuantumDiscriminator**: Discriminates using quantum circuits
- **TransformationPair**: Static matrices for input/output mapping

### 3. Losses (`src/losses/`)
- **QuantumMeasurementLoss**: Direct loss on quantum measurements
- **QuantumWassersteinLoss**: Wasserstein loss for quantum GANs

### 4. Training Examples (`src/examples/`)
- Complete training scripts demonstrating the modular architecture
- Measurement-based loss implementation
- Simple quantum GAN training

## Key Features

1. **Pure Quantum Learning**: No classical neural networks in quantum components
2. **Gradient Flow**: Confirmed gradient flow through all quantum parameters
3. **Modular Design**: Clean separation of concerns
4. **Static Transformations**: Fixed matrices ensure quantum-only learning
5. **Measurement-Based Loss**: Direct optimization on quantum measurements

## Usage

```python
from src.models.quantum_gan import QuantumGAN

# Configure generator and discriminator
generator_config = {
    'latent_dim': 6,
    'output_dim': 2,
    'n_modes': 4,
    'layers': 2,
    'cutoff_dim': 6,
    'measurement_type': 'raw'
}

discriminator_config = {
    'input_dim': 2,
    'n_modes': 2,
    'layers': 2,
    'cutoff_dim': 6,
    'measurement_type': 'raw'
}

# Create and train QGAN
qgan = QuantumGAN(
    generator_config=generator_config,
    discriminator_config=discriminator_config,
    loss_type='wasserstein'
)
```

## Recent Changes

- Moved all old implementations to `legacy/`
- Cleaned up test files and moved to appropriate locations
- Organized documentation in `logs/`
- Created clean modular architecture in `src/`
- Confirmed gradient flow through quantum circuits
