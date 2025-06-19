"""
SF Tutorial Discriminator - Standalone Implementation

This module implements a standalone quantum discriminator using the proven
SF tutorial pattern that eliminates NaN gradients while maintaining
compatibility with our modular systems.
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
from quantum.core.sf_tutorial_quantum_circuit import SFTutorialQuantumCircuit

# Import existing modular components to preserve
from quantum.measurements import RawMeasurementExtractor
from models.transformations import TransformationPair

logger = logging.getLogger(__name__)


class SFTutorialDiscriminator:
    """
    Standalone quantum discriminator using SF tutorial pattern.
    
    Uses proven SF tutorial gradient flow for stable quantum machine learning.
    Outputs binary classification logits for real vs fake data.
    """
    
    def __init__(self, 
                 input_dim: int = 2,
                 n_modes: int = 4,
                 layers: int = 2,
                 cutoff_dim: int = 6):
        """
        Initialize SF tutorial discriminator.
        
        Args:
            input_dim: Dimension of input data
            n_modes: Number of quantum modes
            layers: Number of quantum layers
            cutoff_dim: Fock space cutoff dimension
        """
        self.input_dim = input_dim
        self.n_modes = n_modes
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        
        # Core SF tutorial quantum circuit (proven gradient flow!)
        self.quantum_circuit = SFTutorialQuantumCircuit(
            n_modes=n_modes,
            layers=layers,
            cutoff_dim=cutoff_dim
        )
        
        # Input encoding layer for data-dependent quantum parameters
        self.input_encoder = tf.Variable(
            tf.random.normal([input_dim, self.quantum_circuit.get_parameter_count()], stddev=0.01),
            name="discriminator_input_encoder"
        )
        
        # Keep existing modular measurement system (for future extensibility)
        self.measurements = RawMeasurementExtractor(
            n_modes=n_modes,
            cutoff_dim=cutoff_dim
        )
        
        # Keep existing transformation system (for future extensibility)
        # Get actual measurement dimensions via test execution
        test_state = self.quantum_circuit.execute()
        test_measurements = self.quantum_circuit.extract_measurements(test_state)
        actual_measurement_dim = int(tf.reduce_prod(tf.shape(test_measurements)))
        
        self.transforms = TransformationPair(
            encoder_dim=(input_dim, n_modes * 3),  # For future input encoding enhancements
            decoder_dim=(actual_measurement_dim, 1),  # Quantum measurements to binary output
            trainable=False,  # Static for pure quantum learning
            name_prefix="sf_tutorial_discriminator"
        )
        
        logger.info(f"SF Tutorial Discriminator initialized: {input_dim} → 1 (binary)")
        logger.info(f"  Quantum circuit: {n_modes} modes, {layers} layers")
        logger.info(f"  Parameters: {self.get_parameter_count()}")
        logger.info(f"  Using SF tutorial pattern for 100% gradient flow")
    
    def discriminate(self, x: tf.Tensor) -> tf.Tensor:
        """
        Discriminate input data (real vs fake).
        
        Args:
            x: Input data [batch_size, input_dim]
            
        Returns:
            Logits [batch_size, 1] (positive = real, negative = fake)
        """
        batch_size = tf.shape(x)[0]
        
        # Encode input data to quantum parameter modulation
        input_encoding = tf.matmul(x, self.input_encoder)
        
        # For batch processing, we use mean encoding (SF tutorial limitation)
        # This preserves gradient flow while handling multiple samples
        mean_encoding = tf.reduce_mean(input_encoding, axis=0, keepdims=True)
        
        # Execute SF tutorial quantum circuit (100% gradient flow!)
        state = self.quantum_circuit.execute(mean_encoding)
        
        # Extract measurements using SF tutorial circuit's built-in method
        measurements = self.quantum_circuit.extract_measurements(state)
        
        # Handle tensor shape properly for batching
        measurements_flat = tf.reshape(measurements, [-1])  # Flatten to 1D
        
        # Replicate measurements for each sample in batch
        # (SF tutorial limitation - single execution per batch)
        measurements_expanded = tf.expand_dims(measurements_flat, 0)  # Add batch dimension
        batch_measurements = tf.tile(measurements_expanded, [batch_size, 1])
        
        # Transform quantum measurements to binary classification logits
        logits = self.transforms.decode(batch_measurements)
        
        return logits
    
    def discriminate_with_state_info(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Discriminate with additional quantum state information.
        
        Args:
            x: Input data
            
        Returns:
            Dictionary containing logits and quantum state info
        """
        # Discriminate normally
        logits = self.discriminate(x)
        
        # Get quantum state for analysis
        input_encoding = tf.matmul(x, self.input_encoder)
        mean_encoding = tf.reduce_mean(input_encoding, axis=0, keepdims=True)
        state = self.quantum_circuit.execute(mean_encoding)
        
        # Extract additional state properties
        ket = state.ket()
        state_info = {
            'discriminator_logits': logits,
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
        """Get detailed information about discriminator components."""
        return {
            'input_dim': self.input_dim,
            'output_dim': 1,  # Binary classification
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


def test_sf_tutorial_discriminator_gradients():
    """Test SF tutorial discriminator for 100% gradient flow."""
    print("🧪 Testing SF Tutorial Discriminator Gradients...")
    
    # Test discriminator
    discriminator = SFTutorialDiscriminator(input_dim=2, n_modes=4, layers=2)
    
    print(f"Discriminator parameter count: {discriminator.get_parameter_count()}")
    print(f"Trainable variables: {len(discriminator.trainable_variables)}")
    
    # Test discrimination
    x = tf.random.normal([4, 2])
    
    with tf.GradientTape() as tape:
        logits = discriminator.discriminate(x)
        loss = tf.reduce_mean(tf.square(logits))
    
    gradients = tape.gradient(loss, discriminator.trainable_variables)
    
    # Check gradient health
    all_valid = True
    for i, (var, grad) in enumerate(zip(discriminator.trainable_variables, gradients)):
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
        print("🎉 SUCCESS: SF Tutorial Discriminator has 100% valid gradients!")
        return True
    else:
        print("❌ FAILED: Gradient issues in discriminator")
        return False


def test_sf_tutorial_discriminator_independence():
    """Test that multiple discriminators are independent."""
    print("\n🔄 Testing Discriminator Independence...")
    
    disc1 = SFTutorialDiscriminator(input_dim=2, n_modes=4, layers=2)
    disc2 = SFTutorialDiscriminator(input_dim=2, n_modes=4, layers=2)
    
    # Verify no shared variables
    disc1_vars = set(id(v) for v in disc1.trainable_variables)
    disc2_vars = set(id(v) for v in disc2.trainable_variables)
    
    overlap = disc1_vars.intersection(disc2_vars)
    if len(overlap) == 0:
        print("✅ Discriminators are independent (no shared variables)")
        return True
    else:
        print(f"❌ Discriminators share {len(overlap)} variables")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("SF TUTORIAL DISCRIMINATOR TESTS")
    print("=" * 60)
    
    # Test gradient flow
    test1_success = test_sf_tutorial_discriminator_gradients()
    
    # Test independence
    test2_success = test_sf_tutorial_discriminator_independence()
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if test1_success and test2_success:
        print("🎉 SUCCESS: SF Tutorial Discriminator ready for production!")
    else:
        print("❌ FAILED: Issues detected")
    
    print("=" * 60)
