# Modular Pure Quantum Neural Network Architecture

## Overview

This modular architecture implements a pure quantum approach for neural networks with the following key principles:

1. **Gradient Flow Preservation**: Single SF program per model to maintain gradient flow
2. **Pure Quantum Learning**: All learning through individual quantum gate parameters
3. **Modular Components**: Separable but cohesive modules for flexibility
4. **Raw Measurement Optimization**: Direct optimization on quantum measurements

## Architecture Components

### 1. Quantum Core (`src/quantum/`)

#### Core Circuit (`quantum/core/quantum_circuit.py`)
- **QuantumCircuitBase**: Abstract base ensuring single SF program
- **PureQuantumCircuit**: Implementation with individual gate parameters
- Maintains gradient flow through single execution point

#### Parameter Management (`quantum/parameters/gate_parameters.py`)
- **GateParameterManager**: Manages individual tf.Variables for each gate
- **ParameterModulator**: Converts encodings to parameter modulations
- Ensures all quantum parameters are trainable

#### Circuit Building (`quantum/builders/circuit_builder.py`)
- **CircuitBuilder**: Builds layers within existing SF program
- **LayerBuilder**: Templates for common quantum patterns
- **InterferometerBuilder**: Specialized interferometer architectures
- Never creates new programs - only adds to existing

#### Measurement Extraction (`quantum/measurements/measurement_extractor.py`)
- **RawMeasurementExtractor**: X/P quadratures, photon numbers
- **HolisticMeasurementExtractor**: Includes mode correlations
- **AdaptiveMeasurementExtractor**: Learnable measurement selection
- No statistical processing - raw quantum information

### 2. Models (`src/models/`)

#### Transformations (`models/transformations/matrix_manager.py`)
- **StaticTransformationMatrix**: Non-trainable for pure quantum learning
- **TrainableTransformationMatrix**: When classical parameters needed
- **TransformationPair**: Manages encoder/decoder pairs
- **AdaptiveTransformationMatrix**: Can switch between modes

#### Generators (`models/generators/`)
```python
class QuantumGenerator:
    def __init__(self, config):
        # Single quantum circuit
        self.circuit = PureQuantumCircuit(...)
        
        # Transformation matrices
        self.transforms = TransformationPair(...)
        
        # Measurement extractor
        self.measurements = RawMeasurementExtractor(...)
```

#### Discriminators (`models/discriminators/`)
```python
class QuantumDiscriminator:
    def __init__(self, config):
        # Single quantum circuit
        self.circuit = PureQuantumCircuit(...)
        
        # Input transformation
        self.transform = TransformationMatrix(...)
        
        # Measurement extractor
        self.measurements = RawMeasurementExtractor(...)
        
        # Minimal classifier
        self.classifier = tf.keras.Sequential([...])
```

### 3. QGAN System (`src/qgan/`)

#### QGAN Orchestrator
```python
class PureQuantumQGAN:
    def __init__(self, config):
        self.generator = QuantumGenerator(config.generator)
        self.discriminator = QuantumDiscriminator(config.discriminator)
        self.loss_fn = QuantumWassersteinLoss()
    
    def train_step(self, real_data):
        # Single execution preserves gradients
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            # Generate and discriminate
            # Compute losses on raw measurements
            # Apply gradients
```

### 4. Loss Functions (`src/losses/`)

#### Quantum Wasserstein Loss
- Operates on raw measurement space
- Gradient penalty on measurements
- Quantum-specific regularizations

### 5. Training (`src/training/`)

#### Trainer with Gradient Verification
```python
class QuantumGANTrainer:
    def verify_gradients(self):
        # Ensure all quantum parameters have gradients
        # Check gradient flow through measurements
        # Verify no gradient breaking
```

## Implementation Roadmap

### Phase 1: Core Quantum Components (Week 1)
1. Implement base quantum circuit class
2. Create parameter management system
3. Build circuit construction helpers
4. Test gradient flow preservation

### Phase 2: Model Components (Week 2)
1. Implement transformation matrices
2. Create quantum generator
3. Create quantum discriminator
4. Test individual components

### Phase 3: QGAN Integration (Week 3)
1. Build QGAN orchestrator
2. Implement loss functions
3. Create training loop
4. Add monitoring/callbacks

### Phase 4: Testing & Validation (Week 4)
1. Unit tests for all components
2. Integration tests
3. Performance benchmarks
4. Quantum advantage analysis

## Usage Example

```python
from src.quantum import PureQuantumCircuit
from src.models.generators import QuantumGenerator
from src.models.discriminators import QuantumDiscriminator
from src.qgan import PureQuantumQGAN
from src.training import QuantumGANTrainer

# Configuration
config = {
    'generator': {
        'latent_dim': 6,
        'output_dim': 2,
        'n_modes': 4,
        'layers': 2,
        'transformation': 'static'  # or 'trainable'
    },
    'discriminator': {
        'input_dim': 2,
        'n_modes': 2,
        'layers': 2,
        'transformation': 'static'
    },
    'training': {
        'batch_size': 8,
        'learning_rate': 0.001,
        'epochs': 100
    }
}

# Create QGAN
qgan = PureQuantumQGAN(config)

# Create trainer
trainer = QuantumGANTrainer(qgan, config['training'])

# Train
trainer.train(dataset)

# Evaluate
results = trainer.evaluate()
```

## Key Design Principles

### 1. Gradient Safety
- Single SF program per model
- No program recreation during execution
- Modular components work within existing program

### 2. Pure Quantum Learning
- Option for static transformation matrices
- All learning through quantum gate parameters
- Raw measurement optimization

### 3. Flexibility
- Easy to switch between static/trainable transformations
- Different measurement strategies
- Various loss functions

### 4. Error Resilience
- Graceful handling of quantum circuit failures
- Gradient verification at each step
- Comprehensive logging

## Testing Strategy

### Unit Tests
- Each module independently testable
- Mock quantum states for measurement tests
- Gradient flow verification

### Integration Tests
- Full pipeline execution
- Multi-batch training
- Performance metrics

### Quantum Tests
- Verify quantum properties preserved
- Check entanglement generation
- Measure quantum advantage

## Next Steps

1. **Implement Base Classes**: Start with quantum circuit base
2. **Build Components**: Create modular pieces
3. **Integrate**: Combine into working QGAN
4. **Test**: Comprehensive testing suite
5. **Optimize**: Performance tuning
6. **Document**: API documentation

## Notes on Gradient Flow

The most critical aspect is maintaining gradient flow through the quantum circuit. This is achieved by:

1. **Single Program**: One SF program per model
2. **Single Engine**: One SF engine per model
3. **Single Execution**: One run() call per forward pass
4. **Modular Building**: Components add to program, not create new ones

Example of gradient-safe pattern:
```python
class GradientSafeQuantumModel:
    def __init__(self):
        self.prog = sf.Program(n_modes)
        self.eng = sf.Engine(backend="tf", ...)
        self._build_circuit()  # Builds into self.prog
    
    def forward(self, x):
        # Single execution preserves gradients
        state = self.eng.run(self.prog, args=mapping).state
        return self.extract_measurements(state)
```

## Troubleshooting

### Common Issues

1. **Gradient Breaking**
   - Check for multiple SF programs
   - Verify single execution point
   - Ensure tf.Variables used for parameters

2. **Measurement Errors**
   - Verify state.ket() returns valid tensor
   - Check cutoff dimension sufficient
   - Ensure measurement extraction differentiable

3. **Performance Issues**
   - Batch processing limitations in SF
   - Consider measurement caching
   - Profile quantum circuit execution

### Debug Tools

```python
def verify_gradient_flow(model, input_data):
    with tf.GradientTape() as tape:
        output = model(input_data)
        loss = tf.reduce_mean(output)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    for i, (var, grad) in enumerate(zip(model.trainable_variables, gradients)):
        if grad is None:
            print(f"WARNING: No gradient for {var.name}")
        else:
            print(f"✓ Gradient OK for {var.name}")
```

This modular architecture provides a solid foundation for implementing and experimenting with pure quantum neural networks while maintaining the critical gradient flow through Strawberry Fields.
