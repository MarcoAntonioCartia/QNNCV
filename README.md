# Quantum Neural Networks for Continuous Variables (QNNCV)

A research framework for implementing continuous variable quantum generative adversarial networks using Strawberry Fields with comprehensive training infrastructure and modular quantum circuit architectures.

## Overview

This project implements quantum generative adversarial networks (QGANs) using continuous variable quantum computing. The framework provides modular quantum circuit implementations, complete training infrastructure with gradient flow optimization, and comprehensive evaluation tools for quantum machine learning research.

## Current Implementation Status

**Phase 3: Complete Training Framework**

- **Pure SF Components**: PureSFGenerator and PureSFDiscriminator with individual sample processing
- **Training Infrastructure**: Complete QuantumGANTrainer with Wasserstein loss and gradient penalty
- **Gradient Management**: Robust handling of Strawberry Fields gradient computation with NaN detection and backup strategies
- **Data Generation**: Synthetic data generators for bimodal and multi-distribution testing
- **Evaluation Framework**: Comprehensive monitoring, visualization, and checkpoint management

**Phase 2: Pure SF Architecture** (Completed)

- **Individual Parameter Mapping**: Each quantum parameter as separate tf.Variable for optimal gradient flow
- **Native SF Programs**: Direct Program-Engine execution without classical wrappers
- **30-Parameter Models**: Generator (4 modes, 2 layers) and discriminator (3 modes, 2 layers) configurations

**Phase 1: Gradient Flow Optimization** (Completed)

- **100% Gradient Flow**: Achieved complete gradient flow through quantum parameters
- **Individual Variables**: Fixed gradient counting by using individual tf.Variable for each quantum parameter
- **Quantum-First Architecture**: Optimized parameter structure for quantum circuit training

**Compatibility Layer** (Maintained)

- **Google Colab Ready**: Automatic compatibility patches for NumPy 2.0+, SciPy 1.15+, TensorFlow 2.18+
- **Cross-Platform Support**: Full local setup with CPU/GPU support
- **Legacy Support**: Backward compatibility with earlier implementations

## Installation

### Google Colab (Recommended for Initial Testing)

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

# Import Pure SF components
from models.generators.pure_sf_generator import PureSFGenerator
from models.discriminators.pure_sf_discriminator import PureSFDiscriminator
from training.quantum_gan_trainer import QuantumGANTrainer
```

### Local Development

```bash
git clone https://github.com/your-repo/QNNCV.git
cd QNNCV
python setup/setup_local.py
```

## Quick Start

### Complete Training Example (Phase 3)

```python
# Automatic compatibility (import utils first)
import utils

# Import Pure SF training framework
from models.generators.pure_sf_generator import PureSFGenerator
from models.discriminators.pure_sf_discriminator import PureSFDiscriminator
from training.quantum_gan_trainer import QuantumGANTrainer
from training.data_generators import BimodalDataGenerator

# Create Pure SF quantum components
generator = PureSFGenerator(
    latent_dim=4,
    output_dim=2,
    n_modes=4,
    layers=2,
    cutoff_dim=6
)

discriminator = PureSFDiscriminator(
    input_dim=2,
    n_modes=3,  # Smaller than generator for balanced training
    layers=2,
    cutoff_dim=6
)

# Create data generator
data_generator = BimodalDataGenerator(
    batch_size=8,
    n_features=2,
    mode1_center=(-2.0, -2.0),
    mode2_center=(2.0, 2.0)
)

# Create trainer with Wasserstein loss
trainer = QuantumGANTrainer(
    generator=generator,
    discriminator=discriminator,
    loss_type='wasserstein',
    n_critic=5,
    learning_rate_g=1e-3,
    learning_rate_d=1e-3,
    verbose=True
)

# Train with comprehensive monitoring
trainer.train(
    data_generator=data_generator,
    epochs=30,
    steps_per_epoch=20,
    latent_dim=4,
    validation_data=data_generator.generate_dataset(50),
    plot_interval=10,
    save_interval=20
)
```

### Minimal Training Script

For quick testing, use the provided training script:

```bash
python src/examples/train_pure_sf_qgan.py
```

This script provides:
- Model creation and validation
- Gradient flow verification
- Complete training loop with monitoring
- Automatic result saving and visualization

### Testing Individual Components

```python
import utils
from models.generators.pure_sf_generator import PureSFGenerator
import tensorflow as tf

# Test generator
generator = PureSFGenerator(latent_dim=4, output_dim=2, n_modes=4, layers=2)
z = tf.random.normal([8, 4])
samples = generator.generate(z)

print(f"Generator output shape: {samples.shape}")
print(f"Trainable parameters: {len(generator.trainable_variables)}")

# Verify gradient flow
with tf.GradientTape() as tape:
    output = generator.generate(z)
    loss = tf.reduce_mean(output)

gradients = tape.gradient(loss, generator.trainable_variables)
gradient_flow = sum(1 for g in gradients if g is not None)
print(f"Gradient flow: {gradient_flow}/{len(generator.trainable_variables)}")
```

## Architecture Overview

### Pure SF Implementation (Current)

The framework implements native Strawberry Fields quantum circuits with individual parameter processing:

```
Classical Input → SF Program → Individual Processing → Quantum Output
```

**Key Design Principles:**
- **Individual Sample Processing**: Each data sample processed through separate quantum circuit instance
- **Native SF Programs**: Direct Program-Engine execution without classical wrappers
- **Parameter Isolation**: Each quantum parameter as individual tf.Variable for optimal gradient computation
- **Gradient Management**: Robust handling of SF computational limitations with backup strategies

### Training Framework Components

**QuantumGANTrainer**: Complete training orchestration
- Wasserstein loss with gradient penalty
- Individual sample adversarial training
- Comprehensive gradient flow monitoring
- Automatic evaluation and checkpointing

**Data Generators**: Synthetic data creation
- BimodalDataGenerator: Two-mode distributions for mode coverage testing
- SyntheticDataGenerator: Multi-distribution support (circular, Swiss roll)
- Validation data generation and visualization

**Gradient Management**: Robust SF integration
- NaN gradient detection and handling
- Finite difference backup strategies
- Parameter bounds enforcement
- Comprehensive training monitoring

### Quantum Circuit Architecture

**Pure SF Generator (30 parameters, 4 modes, 2 layers):**
```
Input Encoding → Variational Layer 1 → Variational Layer 2 → Measurement → Output Decoding
```

**Pure SF Discriminator (22 parameters, 3 modes, 2 layers):**
```
Input Encoding → Variational Layer 1 → Variational Layer 2 → Measurement → Classification
```

**Parameter Structure (per layer):**
- Interferometer parameters: Systematic unitary transformations
- Squeezing parameters: Quantum correlation creation
- Displacement parameters: Classical data encoding
- Rotation parameters: Measurement basis selection

## Project Structure

```
QNNCV/
├── src/                              # Core implementation
│   ├── models/
│   │   ├── generators/
│   │   │   ├── pure_sf_generator.py       # Pure SF generator (Phase 2)
│   │   │   └── quantum_sf_generator.py    # Legacy hybrid implementation
│   │   └── discriminators/
│   │       ├── pure_sf_discriminator.py   # Pure SF discriminator (Phase 2)
│   │       └── quantum_sf_discriminator.py # Legacy hybrid implementation
│   ├── training/                     # Training framework (Phase 3)
│   │   ├── quantum_gan_trainer.py         # Complete training orchestration
│   │   ├── data_generators.py             # Synthetic data generation
│   │   └── __init__.py
│   ├── quantum/                      # Quantum circuit infrastructure
│   │   ├── core/
│   │   │   └── pure_sf_circuit.py         # Pure SF circuit implementation
│   │   ├── measurements/
│   │   │   └── measurement_extractor.py   # Quantum measurement handling
│   │   └── parameters/
│   │       └── gate_parameters.py         # Parameter management
│   ├── losses/
│   │   └── quantum_gan_loss.py            # Quantum-aware loss functions
│   ├── utils/
│   │   ├── gradient_manager.py            # Gradient flow management
│   │   ├── compatibility.py               # Environment compatibility
│   │   └── quantum_metrics.py             # Quantum-specific evaluation
│   └── examples/
│       └── train_pure_sf_qgan.py          # Complete training example
├── tutorials/                        # Working examples and notebooks
├── legacy/                          # Previous implementations and research
├── setup/                           # Environment setup
├── tests/                           # Validation and testing
└── docs/                           # Documentation and research notes
```

## Training Configuration

### Recommended Parameters for Initial Testing

```python
# Model configuration
config = {
    'latent_dim': 4,           # Latent space dimension
    'output_dim': 2,           # Output data dimension
    'generator_modes': 4,      # Quantum modes for generator
    'discriminator_modes': 3,  # Quantum modes for discriminator (smaller)
    'layers': 2,               # Variational layers
    'cutoff_dim': 6,           # Fock space cutoff
    
    # Training parameters
    'batch_size': 8,           # Small batches for testing
    'epochs': 30,              # Sufficient for convergence observation
    'steps_per_epoch': 20,     # Quick epochs
    'n_critic': 5,             # Discriminator:generator training ratio
    'learning_rate': 1e-3,     # Conservative learning rate
}
```

### Performance Considerations

**Memory Requirements:**
- 8GB RAM minimum for basic testing
- 16GB+ recommended for extended training
- GPU recommended but not required

**Training Time:**
- Individual sample processing: ~3-4 seconds per step
- 30 epochs (600 steps): ~30-45 minutes
- Scales linearly with batch size and circuit complexity

**Gradient Flow Monitoring:**
- Target: 100% gradient flow (30/30 generator, 22/22 discriminator)
- Monitor gradient clipping frequency
- Watch for NaN gradient detection

## Validation and Testing

### Required Pre-Training Validation

Before training, verify:

```python
# 1. Component creation
generator = PureSFGenerator(latent_dim=4, output_dim=2, n_modes=4, layers=2)
discriminator = PureSFDiscriminator(input_dim=2, n_modes=3, layers=2)

# 2. Forward pass functionality
z_test = tf.random.normal([8, 4])
x_test = tf.random.normal([8, 2])
gen_output = generator.generate(z_test)
disc_output = discriminator.discriminate(x_test)

# 3. Gradient flow verification
# (See gradient flow test in examples above)

# 4. Training step execution
trainer = QuantumGANTrainer(generator, discriminator)
# Verify trainer creation succeeds
```

### Training Monitoring

Monitor these key metrics during training:

```python
# Gradient flow (should remain near 100%)
g_gradients = trainer.metrics_history['g_gradients'][-1]  # Target: 30
d_gradients = trainer.metrics_history['d_gradients'][-1]  # Target: 22

# Loss progression
g_loss = trainer.metrics_history['g_loss'][-1]
d_loss = trainer.metrics_history['d_loss'][-1]
w_distance = trainer.metrics_history['w_distance'][-1]

# Sample diversity (should remain > 1e-4)
z_test = tf.random.normal([100, 4])
samples = generator.generate(z_test)
sample_variance = tf.math.reduce_variance(samples)
```

### Common Issues and Diagnostics

**Issue**: Zero gradient flow
- **Diagnosis**: Check individual component forward passes
- **Solution**: Verify tf.Variable creation in quantum parameters

**Issue**: NaN gradients during training
- **Diagnosis**: Monitor QuantumGradientManager logs
- **Solution**: Automatic backup gradient generation

**Issue**: Mode collapse (low sample variance)
- **Diagnosis**: Monitor sample variance metrics
- **Solution**: Adjust entropy regularization in loss function

**Issue**: Training instability
- **Diagnosis**: Check Wasserstein distance convergence
- **Solution**: Reduce learning rates or increase gradient penalty weight

## Research Applications

### Quantum Machine Learning Research

- **Circuit Architecture**: Novel variational quantum circuit designs
- **Optimization**: Quantum parameter optimization strategies
- **Evaluation**: Quantum-specific quality metrics development

### Continuous Variable Quantum Computing

- **CV Gate Sets**: Implementation of different continuous variable operations
- **Measurement Strategies**: Homodyne vs. heterodyne detection comparison
- **Noise Models**: Integration of realistic quantum noise

### Generative Modeling

- **Distribution Learning**: Complex probability distribution approximation
- **Mode Coverage**: Multi-modal distribution generation
- **Quantum Advantage**: Classical vs. quantum generative model comparison

## Way Forward

### Immediate Next Steps

1. **Extended Validation**: Train on multiple synthetic distributions
2. **Real Dataset Integration**: Implement loaders for standard ML datasets
3. **Performance Optimization**: Batch processing for quantum circuits
4. **Advanced Metrics**: Quantum fidelity and entanglement measures

### Medium-term Development

1. **Hybrid Architectures**: Classical-quantum hybrid discriminators
2. **Advanced Loss Functions**: Quantum-specific adversarial losses
3. **Multi-scale Training**: Progressive training strategies
4. **Hardware Integration**: Real quantum device compatibility

### Long-term Research Directions

1. **Theoretical Analysis**: Quantum GAN convergence guarantees
2. **Scalability**: Large-scale quantum circuit training
3. **Applications**: Domain-specific quantum generative models
4. **Benchmarking**: Standardized evaluation protocols

## System Requirements

### Minimum Requirements
- **RAM**: 8GB
- **Python**: 3.8+
- **Dependencies**: TensorFlow 2.x, Strawberry Fields, NumPy, SciPy

### Recommended for Research
- **RAM**: 16GB+
- **GPU**: CUDA-compatible for classical components
- **Environment**: Dedicated Python environment with fixed dependency versions

## Contributing

### Development Guidelines

1. **Testing**: All new components require unit tests
2. **Documentation**: Code documentation and usage examples
3. **Validation**: Gradient flow verification for quantum components
4. **Compatibility**: Maintain backward compatibility where possible

### Research Areas

- **Quantum Circuit Design**: Novel CV quantum architectures
- **Training Methodologies**: Advanced optimization for quantum parameters
- **Evaluation Metrics**: Quantum-specific quality measures
- **Performance**: Computational efficiency improvements

## License

Academic and research use. See LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```
[Citation format to be determined based on publication]
```

## Support

- **Documentation**: Comprehensive examples in tutorials/ and src/examples/
- **Issues**: Submit detailed bug reports via GitHub issues
- **Setup**: Run setup scripts for automatic environment configuration
- **Validation**: Use provided test scripts to verify installation
