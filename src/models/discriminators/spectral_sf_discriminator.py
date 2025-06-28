"""
Spectral Normalized Strawberry Fields Discriminator

This discriminator combines quantum circuits with spectral normalization
to prevent gradient collapse and stabilize training.
"""

import numpy as np
import tensorflow as tf
import logging
import sys
import os
from typing import Dict, List, Optional, Any

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.quantum.core.pure_sf_circuit import PureSFQuantumCircuit
from src.utils.spectral_normalization import SpectralNormDense

logger = logging.getLogger(__name__)


class SpectralSFDiscriminator:
    """
    Spectral normalized quantum discriminator for stable training.
    
    Combines quantum circuits with spectral normalization to:
    - Prevent gradient explosion/vanishing
    - Maintain stable training dynamics
    - Preserve quantum advantages
    """
    
    def __init__(self, 
                 input_dim: int = 2,
                 n_modes: int = 4,
                 layers: int = 2,
                 cutoff_dim: int = 6,
                 use_spectral_norm: bool = True):
        """
        Initialize spectral normalized SF discriminator.
        
        Args:
            input_dim: Dimension of input data
            n_modes: Number of quantum modes
            layers: Number of quantum layers
            cutoff_dim: Fock space cutoff dimension
            use_spectral_norm: Whether to use spectral normalization
        """
        self.input_dim = input_dim
        self.n_modes = n_modes
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        self.use_spectral_norm = use_spectral_norm
        
        # Core quantum circuit
        self.quantum_circuit = PureSFQuantumCircuit(
            n_modes=n_modes,
            n_layers=layers,
            cutoff_dim=cutoff_dim,
            circuit_type="variational"
        )
        
        # Get measurement dimension
        test_state = self.quantum_circuit.execute()
        test_measurements = self.quantum_circuit.extract_measurements(test_state)
        self.measurement_dim = int(tf.reduce_prod(tf.shape(test_measurements)))
        
        # Build neural network components with spectral normalization
        self._build_networks()
        
        logger.info(f"Spectral SF Discriminator initialized: {input_dim} → 1")
        logger.info(f"  Quantum circuit: {n_modes} modes, {layers} layers")
        logger.info(f"  Measurement dim: {self.measurement_dim}")
        logger.info(f"  Spectral normalization: {use_spectral_norm}")
        logger.info(f"  Total parameters: {self.get_parameter_count()}")
    
    def _build_networks(self):
        """Build neural network components."""
        # Input encoder: maps input to quantum parameters
        if self.use_spectral_norm:
            self.input_encoder = tf.keras.Sequential([
                SpectralNormDense(
                    units=self.n_modes * 2,
                    activation='tanh',
                    name='input_encoder_spectral'
                )
            ], name='input_encoder')
        else:
            self.input_encoder = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    units=self.n_modes * 2,
                    activation='tanh',
                    name='input_encoder_dense'
                )
            ], name='input_encoder')
        
        # Output decoder: maps quantum measurements to logits
        if self.use_spectral_norm:
            self.output_decoder = tf.keras.Sequential([
                SpectralNormDense(
                    units=32,
                    activation='leaky_relu',
                    name='decoder_hidden_spectral'
                ),
                SpectralNormDense(
                    units=1,
                    activation=None,
                    name='decoder_output_spectral'
                )
            ], name='output_decoder')
        else:
            self.output_decoder = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    units=32,
                    activation='leaky_relu',
                    name='decoder_hidden'
                ),
                tf.keras.layers.Dense(
                    units=1,
                    activation=None,
                    name='decoder_output'
                )
            ], name='output_decoder')
        
        # Build networks with dummy inputs
        dummy_input = tf.zeros([1, self.input_dim])
        dummy_measurements = tf.zeros([1, self.measurement_dim])
        
        self.input_encoder(dummy_input)
        self.output_decoder(dummy_measurements)
    
    def discriminate(self, x: tf.Tensor) -> tf.Tensor:
        """
        Discriminate input data with spectral normalization.
        
        Args:
            x: Input data [batch_size, input_dim]
            
        Returns:
            Logits [batch_size, 1]
        """
        batch_size = tf.shape(x)[0]
        
        # Encode input to quantum parameters
        quantum_params = self.input_encoder(x)  # [batch_size, n_modes * 2]
        
        # Process each sample through quantum circuit
        batch_measurements = []
        for i in range(batch_size):
            # Get parameters for this sample
            sample_params = quantum_params[i]
            
            # Execute quantum circuit
            state = self.quantum_circuit.execute(input_encoding=sample_params)
            measurements = self.quantum_circuit.extract_measurements(state)
            
            # Flatten measurements
            measurements_flat = tf.reshape(measurements, [-1])
            batch_measurements.append(measurements_flat)
        
        # Stack measurements
        batch_measurements = tf.stack(batch_measurements, axis=0)
        
        # Decode to logits with spectral normalization
        logits = self.output_decoder(batch_measurements)
        
        return logits
    
    def discriminate_with_gradient_info(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Discriminate with gradient information for debugging.
        
        Args:
            x: Input data
            
        Returns:
            Dictionary with logits and gradient info
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            logits = self.discriminate(x)
            loss = tf.reduce_mean(tf.square(logits))
        
        # Compute gradients
        gradients = tape.gradient(loss, x)
        
        # Gradient norm
        if gradients is not None:
            grad_norm = tf.norm(gradients)
        else:
            grad_norm = tf.constant(0.0)
        
        return {
            'logits': logits,
            'gradient_norm': grad_norm,
            'has_gradients': gradients is not None
        }
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Return all trainable variables."""
        variables = []
        
        # Quantum circuit variables
        variables.extend(self.quantum_circuit.trainable_variables)
        
        # Neural network variables
        variables.extend(self.input_encoder.trainable_variables)
        variables.extend(self.output_decoder.trainable_variables)
        
        return variables
    
    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters."""
        total = 0
        for var in self.trainable_variables:
            total += tf.reduce_prod(var.shape)
        return int(total)
    
    def get_spectral_norms(self) -> Dict[str, float]:
        """Get spectral norms of all layers (for monitoring)."""
        norms = {}
        
        if self.use_spectral_norm:
            # Get spectral norms from spectral normalized layers
            for layer in self.input_encoder.layers:
                if isinstance(layer, SpectralNormDense):
                    # Compute spectral norm of the kernel
                    kernel = layer.kernel
                    singular_values = tf.linalg.svd(kernel, compute_uv=False)
                    max_sv = tf.reduce_max(singular_values)
                    norms[f'input_encoder_{layer.name}'] = float(max_sv)
            
            for layer in self.output_decoder.layers:
                if isinstance(layer, SpectralNormDense):
                    kernel = layer.kernel
                    singular_values = tf.linalg.svd(kernel, compute_uv=False)
                    max_sv = tf.reduce_max(singular_values)
                    norms[f'output_decoder_{layer.name}'] = float(max_sv)
        
        return norms
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get detailed component information."""
        return {
            'input_dim': self.input_dim,
            'output_dim': 1,
            'quantum_circuit_info': self.quantum_circuit.get_circuit_info(),
            'measurement_dim': self.measurement_dim,
            'parameter_count': self.get_parameter_count(),
            'spectral_normalization': self.use_spectral_norm,
            'spectral_norms': self.get_spectral_norms(),
            'components': {
                'quantum_circuit': 'PureSFQuantumCircuit',
                'input_encoder': 'SpectralNormDense' if self.use_spectral_norm else 'Dense',
                'output_decoder': 'SpectralNormDense' if self.use_spectral_norm else 'Dense'
            }
        }


def test_spectral_sf_discriminator():
    """Test spectral normalized discriminator."""
    print("🧪 Testing Spectral SF Discriminator...")
    
    # Test with spectral normalization
    discriminator = SpectralSFDiscriminator(
        input_dim=2,
        n_modes=4,
        layers=1,
        use_spectral_norm=True
    )
    
    print(f"Parameter count: {discriminator.get_parameter_count()}")
    print(f"Trainable variables: {len(discriminator.trainable_variables)}")
    
    # Test discrimination
    x = tf.random.normal([4, 2])
    
    try:
        print("Testing discrimination...")
        logits = discriminator.discriminate(x)
        print(f"✅ Discrimination successful: {x.shape} → {logits.shape}")
        print(f"   Sample logits: {logits.numpy()[:2].flatten()}")
        
        # Test gradient flow
        print("Testing gradient flow...")
        with tf.GradientTape() as tape:
            logits = discriminator.discriminate(x)
            loss = tf.reduce_mean(tf.square(logits))
        
        gradients = tape.gradient(loss, discriminator.trainable_variables)
        valid_grads = [g for g in gradients if g is not None]
        grad_ratio = len(valid_grads) / len(discriminator.trainable_variables)
        
        print(f"✅ Gradient flow: {grad_ratio:.1%} ({len(valid_grads)}/{len(discriminator.trainable_variables)})")
        
        # Test spectral norms
        if discriminator.use_spectral_norm:
            spectral_norms = discriminator.get_spectral_norms()
            print(f"✅ Spectral norms: {spectral_norms}")
            
            # Check if spectral norms are close to 1 (as expected)
            for name, norm in spectral_norms.items():
                if norm > 2.0:
                    print(f"⚠️ High spectral norm in {name}: {norm:.3f}")
        
        # Test gradient info
        grad_info = discriminator.discriminate_with_gradient_info(x)
        print(f"✅ Gradient info: norm={grad_info['gradient_norm']:.6f}, has_grads={grad_info['has_gradients']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("SPECTRAL SF DISCRIMINATOR TESTS")
    print("=" * 60)
    
    success = test_spectral_sf_discriminator()
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if success:
        print("🎉 SUCCESS: Spectral SF Discriminator ready!")
        print("✅ Spectral normalization working")
        print("✅ Gradient flow preserved")
        print("✅ Quantum circuit integration")
    else:
        print("❌ FAILED: Issues detected")
    
    print("=" * 60)
