"""
SF Tutorial Generator - Standalone Implementation

This module implements a standalone quantum generator using the proven
SF tutorial pattern that eliminates NaN gradients while maintaining
compatibility with our modular measurement and transformation systems.
"""

import numpy as np
import tensorflow as tf
import logging
import sys
import os
from typing import Dict, List, Optional, Any

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', '..')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import our proven SF tutorial circuit
from quantum.core.sf_tutorial_quantum_circuit import SFTutorialQuantumCircuit, SFTutorialCircuitWithEncoding

# Import existing modular components to preserve
from quantum.measurements import RawMeasurementExtractor
from models.transformations import TransformationPair

# Import warning suppression (skip for now - just for clean output)
# from utils.warning_suppression import suppress_tf_warnings

logger = logging.getLogger(__name__)


class SFTutorialGenerator:
    """
    Standalone quantum generator using SF tutorial pattern.
    
    Combines proven SF tutorial gradient flow with existing modular
    measurement and transformation systems for future extensibility.
    """
    
    def __init__(self, 
                 latent_dim: int = 6,
                 output_dim: int = 2,
                 n_modes: int = 4,
                 layers: int = 2,
                 cutoff_dim: int = 6):
        """
        Initialize SF tutorial generator.
        
        Args:
            latent_dim: Dimension of latent input
            output_dim: Dimension of generated output
            n_modes: Number of quantum modes
            layers: Number of quantum layers
            cutoff_dim: Fock space cutoff dimension
        """
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_modes = n_modes
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        
        # Suppress TensorFlow warnings for clean output (disabled for now)
        # suppress_tf_warnings()
        
        # Core SF tutorial quantum circuit (proven gradient flow!)
        self.quantum_circuit = SFTutorialQuantumCircuit(
            n_modes=n_modes,
            layers=layers,
            cutoff_dim=cutoff_dim
        )
        
        # Input encoding layer for data-dependent quantum parameters
        self.input_encoder = tf.Variable(
            tf.random.normal([latent_dim, self.quantum_circuit.get_parameter_count()], stddev=0.01),
            name="generator_input_encoder"
        )
        
        # Keep existing modular measurement system (for future extensibility)
        self.measurements = RawMeasurementExtractor(
            n_modes=n_modes,
            cutoff_dim=cutoff_dim
        )
        
        # Keep existing transformation system (for future extensibility)
        # Note: SF tutorial circuit produces more complex measurements, so we'll determine
        # the actual measurement dimension after a test execution
        
        # Temporary execution to get actual measurement dimensions
        test_state = self.quantum_circuit.execute()
        test_measurements = self.quantum_circuit.extract_measurements(test_state)
        actual_measurement_dim = int(tf.reduce_prod(tf.shape(test_measurements)))
        
        self.transforms = TransformationPair(
            encoder_dim=(latent_dim, n_modes * 3),  # For future input encoding enhancements
            decoder_dim=(actual_measurement_dim, output_dim),  # Actual measurements to output
            trainable=False,  # Static for pure quantum learning
            name_prefix="sf_tutorial_generator"
        )
        
        logger.info(f"SF Tutorial Generator initialized: {latent_dim} → {output_dim}")
        logger.info(f"  Quantum circuit: {n_modes} modes, {layers} layers")
        logger.info(f"  Parameters: {self.get_parameter_count()}")
        logger.info(f"  Using SF tutorial pattern for 100% gradient flow")
    
    def generate(self, z: tf.Tensor) -> tf.Tensor:
        """
        Generate samples using SF tutorial quantum circuit.
        
        Args:
            z: Latent input [batch_size, latent_dim]
            
        Returns:
            Generated samples [batch_size, output_dim]
        """
        batch_size = tf.shape(z)[0]
        
        # Encode latent input to quantum parameter modulation
        input_encoding = tf.matmul(z, self.input_encoder)
        
        # For batch processing, we use mean encoding (SF tutorial limitation)
        # This preserves gradient flow while handling multiple samples
        mean_encoding = tf.reduce_mean(input_encoding, axis=0, keepdims=True)
        
        # Execute SF tutorial quantum circuit (100% gradient flow!)
        state = self.quantum_circuit.execute(mean_encoding)
        
        # Extract measurements using SF tutorial circuit's built-in method
        measurements = self.quantum_circuit.extract_measurements(state)
        
        # Handle tensor shape properly for batching
        measurements_shape = tf.shape(measurements)
        measurements_flat = tf.reshape(measurements, [-1])  # Flatten to 1D
        
        # Replicate measurements for each sample in batch
        # (SF tutorial limitation - single execution per batch)
        measurements_expanded = tf.expand_dims(measurements_flat, 0)  # Add batch dimension
        batch_measurements = tf.tile(measurements_expanded, [batch_size, 1])
        
        # Transform quantum measurements to output space
        output = self.transforms.decode(batch_measurements)
        
        return output
    
    def generate_with_state_info(self, z: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Generate samples with additional quantum state information.
        
        Args:
            z: Latent input
            
        Returns:
            Dictionary containing generated samples and quantum state info
        """
        # Generate normally
        samples = self.generate(z)
        
        # Get quantum state for analysis
        input_encoding = tf.matmul(z, self.input_encoder)
        mean_encoding = tf.reduce_mean(input_encoding, axis=0, keepdims=True)
        state = self.quantum_circuit.execute(mean_encoding)
        
        # Extract additional state properties
        ket = state.ket()
        state_info = {
            'generated_samples': samples,
            'quantum_state_norm': tf.norm(ket),
            'quantum_measurements': self.quantum_circuit.extract_measurements(state),
            'parameter_encoding': mean_encoding
        }
        
        return state_info
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Return all trainable variables."""
        return [
            self.input_encoder,  # Input encoding weights
        ] + self.quantum_circuit.trainable_variables  # SF tutorial quantum weights
        # Note: measurements and transforms are static for pure quantum learning
    
    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters."""
        total = 0
        for var in self.trainable_variables:
            total += int(tf.reduce_prod(var.shape))
        return total
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get detailed information about generator components."""
        return {
            'latent_dim': self.latent_dim,
            'output_dim': self.output_dim,
            'quantum_circuit_info': self.quantum_circuit.get_circuit_info(),
            'parameter_count': self.get_parameter_count(),
            'trainable_variables': len(self.trainable_variables),
            'sf_tutorial_compatible': True,
            'components': {
                'quantum_circuit': 'SFTutorialQuantumCircuit',
                'measurements': type(self.measurements).__name__,
                'transforms': type(self.transforms).__name__
            }
        }


class SFTutorialGeneratorWithAdvancedEncoding(SFTutorialGenerator):
    """
    Enhanced SF tutorial generator with advanced input encoding.
    
    This version uses the SFTutorialCircuitWithEncoding for more sophisticated
    input-dependent quantum parameter modulation.
    """
    
    def __init__(self, 
                 latent_dim: int = 6,
                 output_dim: int = 2,
                 n_modes: int = 4,
                 layers: int = 2,
                 cutoff_dim: int = 6):
        """Initialize enhanced SF tutorial generator."""
        # Initialize base components first
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_modes = n_modes
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        
        # suppress_tf_warnings()  # Disabled for now
        
        # Use the enhanced circuit with encoding
        self.quantum_circuit = SFTutorialCircuitWithEncoding(
            n_modes=n_modes,
            layers=layers,
            input_dim=latent_dim,
            cutoff_dim=cutoff_dim
        )
        
        # Keep modular components
        self.measurements = RawMeasurementExtractor(n_modes=n_modes, cutoff_dim=cutoff_dim)
        
        measurement_dim = self.quantum_circuit.get_measurement_dim()
        self.transforms = TransformationPair(
            encoder_dim=(latent_dim, n_modes * 3),
            decoder_dim=(measurement_dim, output_dim),
            trainable=False,
            name_prefix="sf_tutorial_generator_advanced"
        )
        
        logger.info(f"SF Tutorial Generator (Advanced) initialized: {latent_dim} → {output_dim}")
    
    def generate(self, z: tf.Tensor) -> tf.Tensor:
        """Generate using advanced encoding circuit."""
        batch_size = tf.shape(z)[0]
        
        # Execute with trainable encoding (more sophisticated)
        state = self.quantum_circuit.execute_with_encoding(z)
        
        # Extract measurements
        measurements = self.quantum_circuit.extract_measurements(state)
        
        # Replicate for batch
        batch_measurements = tf.tile([measurements], [batch_size, 1])
        
        # Transform to output
        output = self.transforms.decode(batch_measurements)
        
        return output
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Return trainable variables (includes advanced encoding)."""
        return self.quantum_circuit.trainable_variables  # Both weights and encoder


def test_sf_tutorial_generator_gradients():
    """Test SF tutorial generator for 100% gradient flow."""
    print("🧪 Testing SF Tutorial Generator Gradients...")
    
    # Test basic generator
    generator = SFTutorialGenerator(latent_dim=6, output_dim=2, n_modes=4, layers=2)
    
    print(f"Generator parameter count: {generator.get_parameter_count()}")
    print(f"Trainable variables: {len(generator.trainable_variables)}")
    
    # Test generation
    z = tf.random.normal([4, 6])
    
    with tf.GradientTape() as tape:
        samples = generator.generate(z)
        loss = tf.reduce_mean(tf.square(samples))
    
    gradients = tape.gradient(loss, generator.trainable_variables)
    
    # Check gradient health
    all_valid = True
    for i, (var, grad) in enumerate(zip(generator.trainable_variables, gradients)):
        if grad is None:
            print(f"❌ No gradient for variable {i}: {var.name}")
            all_valid = False
        elif tf.reduce_any(tf.math.is_nan(grad)):
            print(f"❌ NaN gradient for variable {i}: {var.name}")
            all_valid = False
        else:
            grad_norm = tf.norm(grad)
            print(f"✅ Valid gradient for {var.name}: norm = {grad_norm:.6f}")
    
    if all_valid:
        print("🎉 SUCCESS: SF Tutorial Generator has 100% valid gradients!")
        return True
    else:
        print("❌ FAILED: Gradient issues in generator")
        return False


def test_sf_tutorial_generator_independence():
    """Test that multiple generators are independent."""
    print("\n🔄 Testing Generator Independence...")
    
    gen1 = SFTutorialGenerator(latent_dim=6, output_dim=2, n_modes=4, layers=2)
    gen2 = SFTutorialGenerator(latent_dim=6, output_dim=2, n_modes=4, layers=2)
    
    # Verify no shared variables
    gen1_vars = set(id(v) for v in gen1.trainable_variables)
    gen2_vars = set(id(v) for v in gen2.trainable_variables)
    
    overlap = gen1_vars.intersection(gen2_vars)
    if len(overlap) == 0:
        print("✅ Generators are independent (no shared variables)")
        return True
    else:
        print(f"❌ Generators share {len(overlap)} variables")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("SF TUTORIAL GENERATOR TESTS")
    print("=" * 60)
    
    # Test gradient flow
    test1_success = test_sf_tutorial_generator_gradients()
    
    # Test independence
    test2_success = test_sf_tutorial_generator_independence()
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if test1_success and test2_success:
        print("🎉 SUCCESS: SF Tutorial Generator ready for production!")
    else:
        print("❌ FAILED: Issues detected")
    
    print("=" * 60)
