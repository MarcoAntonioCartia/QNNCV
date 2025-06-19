"""
Quantum encoding strategies for continuous variable quantum GANs.

This module provides various encoding schemes to map classical data into quantum
parameters for continuous variable quantum circuits. Each encoding strategy offers
different advantages for quantum advantage and expressivity.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class QuantumEncodingStrategy:
    """Base class for quantum encoding strategies."""
    
    def __init__(self, name: str):
        self.name = name
        
    def encode(self, z: tf.Tensor, n_modes: int, **kwargs) -> tf.Tensor:
        """
        Encode classical data into quantum parameters.
        
        Args:
            z: Classical latent vector [batch_size, latent_dim]
            n_modes: Number of quantum modes
            **kwargs: Additional encoding-specific parameters
            
        Returns:
            Encoded quantum parameters
        """
        raise NotImplementedError("Subclasses must implement encode method")
    
    def get_parameter_count(self, n_modes: int, **kwargs) -> int:
        """Get number of parameters produced by this encoding."""
        raise NotImplementedError("Subclasses must implement get_parameter_count method")

class CoherentStateEncoding(QuantumEncodingStrategy):
    """
    FIXED: Coherent state encoding with ZERO trainable parameters.
    
    Encodes classical data as complex amplitudes for coherent states using
    static mathematical transformations only. No tf.keras layers are used.
    """
    
    def __init__(self):
        super().__init__("coherent_state")
        self.static_weights_real = None
        self.static_weights_imag = None
        
    def encode(self, z: tf.Tensor, n_modes: int, **kwargs) -> tf.Tensor:
        """
        FIXED: Encode data as coherent state amplitudes with ZERO trainable parameters.
        
        Args:
            z: latent vector [batch_size, latent_dim]
            n_modes: number of quantum modes
            
        Returns:
            Complex amplitudes for coherent states [batch_size, 2*n_modes]
        """
        latent_dim = z.shape[-1]
        
        # Use simple linear transformation instead of complex slicing/padding
        if latent_dim >= 2 * n_modes:
            # Direct slicing - this preserves gradients
            real_parts = z[..., :n_modes]
            imag_parts = z[..., n_modes:2*n_modes]
        else:
            # Use STATIC transformation matrices (NO trainable parameters)
            if self.static_weights_real is None:
                # Create fixed random transformation matrices (Xavier initialization)
                import numpy as np
                np.random.seed(42)  # Fixed seed for reproducibility
                self.static_weights_real = tf.constant(
                    np.random.randn(latent_dim, n_modes) * np.sqrt(2.0 / (latent_dim + n_modes)),
                    dtype=tf.float32,
                    name='static_coherent_real'
                )
                self.static_weights_imag = tf.constant(
                    np.random.randn(latent_dim, n_modes) * np.sqrt(2.0 / (latent_dim + n_modes)),
                    dtype=tf.float32,
                    name='static_coherent_imag'
                )
            
            # Apply static transformation (NO trainable variables)
            real_parts = tf.matmul(z, self.static_weights_real)
            imag_parts = tf.matmul(z, self.static_weights_imag)
        
        # Scale to reasonable range for quantum stability
        real_parts = real_parts * 0.3
        imag_parts = imag_parts * 0.3
        
        # Return concatenated real and imaginary parts
        return tf.concat([real_parts, imag_parts], axis=-1)
    
    def get_parameter_count(self, n_modes: int, **kwargs) -> int:
        """Get number of parameters (2 per mode for complex amplitudes)."""
        return 2 * n_modes

class DirectDisplacementEncoding(QuantumEncodingStrategy):
    """
    Direct displacement encoding - simplest approach.
    
    Directly uses latent values as displacement parameters for displacement gates.
    This is the most straightforward encoding but may limit quantum expressivity.
    """
    
    def __init__(self):
        super().__init__("direct_displacement")
        
    def encode(self, z: tf.Tensor, n_modes: int, **kwargs) -> tf.Tensor:
        """
        Directly use latent values as displacement parameters.
        
        Args:
            z: latent vector [batch_size, latent_dim] 
            n_modes: number of quantum modes
            
        Returns:
            Displacement parameters for each mode [batch_size, n_modes]
        """
        latent_dim = z.shape[-1]
        
        # Repeat or truncate to match n_modes
        if latent_dim >= n_modes:
            # Take first n_modes values
            displacements = z[..., :n_modes]
        else:
            # Repeat pattern to fill n_modes
            repeats = n_modes // latent_dim + 1
            repeated = tf.tile(z, [1, repeats])
            displacements = repeated[..., :n_modes]
        
        # Scale to reasonable range for quantum stability
        displacements = displacements * 0.5  # Keep displacements small
        
        return displacements
    
    def get_parameter_count(self, n_modes: int, **kwargs) -> int:
        """Get number of parameters (1 per mode for displacement)."""
        return n_modes

class AngleEncoding(QuantumEncodingStrategy):
    """
    FIXED: Angle encoding with learnable transformations that preserves gradients.
    
    Encodes classical data as rotation angles in quantum gates using learnable
    linear transformations. This allows for more complex mappings and can be
    beneficial for complex data distributions.
    """
    
    def __init__(self):
        super().__init__("angle_encoding")
        self.weights_initialized = False
        self.transformation_weights = None
        
    def encode(self, z: tf.Tensor, n_modes: int, n_layers: int = 1, **kwargs) -> tf.Tensor:
        """
        FIXED: Encode classical data as rotation angles with gradient preservation.
        
        Args:
            z: latent vector [batch_size, latent_dim]
            n_modes: number of quantum modes  
            n_layers: number of quantum layers
            
        Returns:
            Angles for rotation gates [batch_size, n_layers * n_modes]
        """
        latent_dim = z.shape[-1]
        total_angles = n_modes * n_layers
        
        # Initialize weights outside of gradient tape
        if not self.weights_initialized:
            self.transformation_weights = tf.Variable(
                tf.random.normal([latent_dim, total_angles], stddev=0.1),
                name='fixed_angle_weights'
            )
            self.weights_initialized = True
        
        # Simple linear transformation
        angles = tf.matmul(z, self.transformation_weights)
        
        # Use tanh instead of sigmoid for better gradients
        angles = tf.nn.tanh(angles) * np.pi  # Range [-π, π]
        
        return angles
    
    def get_parameter_count(self, n_modes: int, n_layers: int = 1, **kwargs) -> int:
        """Get number of parameters (n_modes per layer)."""
        return n_modes * n_layers

class SparseParameterEncoding(QuantumEncodingStrategy):
    """
    FIXED: Sparse parameter encoding that preserves gradients.
    
    Only modulates a subset of quantum parameters while most remain as learnable
    constants. This can be more efficient and may help with training stability
    while still allowing input-dependent quantum circuits.
    """
    
    def __init__(self, sparsity_ratio: float = 0.3):
        super().__init__("sparse_parameter")
        self.sparsity_ratio = sparsity_ratio
        self.weights_initialized = False
        self.transformation_weights = None
        self.modulation_indices = None
        
    def encode(self, z: tf.Tensor, n_modes: int, n_layers: int = 1, 
               total_quantum_params: Optional[int] = None, **kwargs) -> tf.Tensor:
        """
        FIXED: Sparsely modulate quantum parameters with gradient preservation.
        
        Args:
            z: latent vector [batch_size, latent_dim]
            n_modes: number of quantum modes
            n_layers: number of quantum layers
            total_quantum_params: total number of quantum circuit parameters
            
        Returns:
            Parameter modulations (mostly zeros) [batch_size, total_quantum_params]
        """
        if total_quantum_params is None:
            M = int(n_modes * (n_modes - 1)) + max(1, n_modes - 1)
            total_quantum_params = n_layers * (2 * M + 4 * n_modes)
        
        latent_dim = z.shape[-1]
        batch_size = tf.shape(z)[0]
        n_modulated = int(total_quantum_params * self.sparsity_ratio)
        
        # Initialize weights and indices outside gradient tape
        if not self.weights_initialized:
            self.transformation_weights = tf.Variable(
                tf.random.normal([latent_dim, n_modulated], stddev=0.1),
                name='fixed_sparse_weights'
            )
            # Use fixed indices instead of random shuffle
            self.modulation_indices = tf.constant(
                np.random.choice(total_quantum_params, n_modulated, replace=False),
                name='fixed_modulation_indices'
            )
            self.weights_initialized = True
        
        # Linear transformation to modulated parameters
        modulated_params = tf.matmul(z, self.transformation_weights)
        
        # Use one_hot instead of scatter for better gradients
        indices_one_hot = tf.one_hot(self.modulation_indices, total_quantum_params)
        
        # Broadcast and multiply
        modulated_expanded = tf.expand_dims(modulated_params, -1)  # [batch, n_modulated, 1]
        indices_expanded = tf.expand_dims(indices_one_hot, 0)      # [1, n_modulated, total_params]
        
        # Element-wise multiplication and sum
        full_modulation = tf.reduce_sum(modulated_expanded * indices_expanded, axis=1)
        
        return full_modulation * 0.1
    
    def get_parameter_count(self, n_modes: int, n_layers: int = 1, 
                          total_quantum_params: Optional[int] = None, **kwargs) -> int:
        """Get number of modulated parameters."""
        if total_quantum_params is None:
            M = int(n_modes * (n_modes - 1)) + max(1, n_modes - 1)
            total_quantum_params = n_layers * (2 * M + 4 * n_modes)
        return int(total_quantum_params * self.sparsity_ratio)

class ClassicalNeuralEncoding(QuantumEncodingStrategy):
    """
    Classical neural network encoding (backward compatibility).
    
    Uses a classical neural network to map latent vectors to quantum parameters.
    This maintains compatibility with existing implementations while providing
    a flexible encoding strategy.
    """
    
    def __init__(self, hidden_units: int = 16):
        super().__init__("classical_neural")
        self.hidden_units = hidden_units
        self.encoder_network = None
        
    def encode(self, z: tf.Tensor, n_modes: int, n_layers: int = 1, 
               total_quantum_params: Optional[int] = None, **kwargs) -> tf.Tensor:
        """
        Encode using classical neural network.
        
        Args:
            z: latent vector [batch_size, latent_dim]
            n_modes: number of quantum modes
            n_layers: number of quantum layers
            total_quantum_params: total number of quantum circuit parameters
            
        Returns:
            Quantum parameters [batch_size, total_quantum_params]
        """
        if total_quantum_params is None:
            # Estimate based on typical SF circuit structure
            M = int(n_modes * (n_modes - 1)) + max(1, n_modes - 1)
            total_quantum_params = n_layers * (2 * M + 4 * n_modes)
        
        latent_dim = z.shape[-1]
        
        # Create encoder network if not exists
        if self.encoder_network is None:
            self.encoder_network = tf.keras.Sequential([
                tf.keras.layers.Dense(self.hidden_units, activation='tanh', 
                                    name='classical_encoder_hidden'),
                tf.keras.layers.Dense(total_quantum_params, activation='tanh', 
                                    name='classical_encoder_output')
            ], name='classical_neural_encoder')
            
            # Build the network
            dummy_input = tf.zeros((1, latent_dim))
            _ = self.encoder_network(dummy_input)
        
        # Encode latent to quantum parameters
        encoded_params = self.encoder_network(z)
        
        return encoded_params
    
    def get_parameter_count(self, n_modes: int, n_layers: int = 1, 
                          total_quantum_params: Optional[int] = None, **kwargs) -> int:
        """Get number of quantum parameters produced."""
        if total_quantum_params is None:
            M = int(n_modes * (n_modes - 1)) + max(1, n_modes - 1)
            total_quantum_params = n_layers * (2 * M + 4 * n_modes)
        return total_quantum_params

class QuantumEncodingFactory:
    """Factory class for creating quantum encoding strategies."""
    
    _strategies = {
        'coherent_state': CoherentStateEncoding,
        'direct_displacement': DirectDisplacementEncoding,
        'angle_encoding': AngleEncoding,
        'sparse_parameter': SparseParameterEncoding,
        'classical_neural': ClassicalNeuralEncoding
    }
    
    @classmethod
    def create_encoding(cls, strategy_name: str, **kwargs) -> QuantumEncodingStrategy:
        """
        Create a quantum encoding strategy.
        
        Args:
            strategy_name: Name of the encoding strategy
            **kwargs: Strategy-specific parameters
            
        Returns:
            Quantum encoding strategy instance
        """
        if strategy_name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(f"Unknown encoding strategy: {strategy_name}. "
                           f"Available strategies: {available}")
        
        strategy_class = cls._strategies[strategy_name]
        
        # Handle strategy-specific parameters
        if strategy_name == 'sparse_parameter':
            sparsity_ratio = kwargs.get('sparsity_ratio', 0.3)
            return strategy_class(sparsity_ratio=sparsity_ratio)
        elif strategy_name == 'classical_neural':
            hidden_units = kwargs.get('hidden_units', 16)
            return strategy_class(hidden_units=hidden_units)
        else:
            return strategy_class()
    
    @classmethod
    def get_available_strategies(cls) -> list:
        """Get list of available encoding strategies."""
        return list(cls._strategies.keys())
    
    @classmethod
    def get_strategy_info(cls) -> Dict[str, str]:
        """Get information about available strategies."""
        return {
            'coherent_state': 'Natural encoding for CV quantum computing using complex amplitudes',
            'direct_displacement': 'Simple direct mapping to displacement parameters',
            'angle_encoding': 'Learnable transformation to rotation angles',
            'sparse_parameter': 'Efficient encoding modulating only subset of parameters',
            'classical_neural': 'Neural network encoding for backward compatibility'
        }

def test_quantum_encodings():
    """Test all quantum encoding strategies."""
    print("Testing Quantum Encoding Strategies...")
    print("=" * 50)
    
    # Test parameters
    batch_size = 4
    latent_dim = 6
    n_modes = 2
    n_layers = 2
    
    # Create test data
    z_test = tf.random.normal([batch_size, latent_dim])
    print(f"Test input shape: {z_test.shape}")
    
    # Test all encoding strategies
    strategies = QuantumEncodingFactory.get_available_strategies()
    
    for strategy_name in strategies:
        print(f"\nTesting {strategy_name} encoding:")
        print("-" * 30)
        
        try:
            # Create encoding strategy
            encoding = QuantumEncodingFactory.create_encoding(strategy_name)
            
            # Test encoding
            if strategy_name in ['angle_encoding', 'sparse_parameter', 'classical_neural']:
                encoded = encoding.encode(z_test, n_modes, n_layers=n_layers)
            else:
                encoded = encoding.encode(z_test, n_modes)
            
            print(f"✓ Encoding successful")
            print(f"  Output shape: {encoded.shape}")
            print(f"  Output range: [{tf.reduce_min(encoded):.3f}, {tf.reduce_max(encoded):.3f}]")
            
            # Test parameter count
            if strategy_name in ['angle_encoding', 'sparse_parameter', 'classical_neural']:
                param_count = encoding.get_parameter_count(n_modes, n_layers=n_layers)
            else:
                param_count = encoding.get_parameter_count(n_modes)
            
            print(f"  Parameter count: {param_count}")
            
        except Exception as e:
            print(f"✗ Encoding failed: {e}")
    
    # Test factory methods
    print(f"\n\nFactory Methods Test:")
    print("-" * 30)
    print(f"Available strategies: {QuantumEncodingFactory.get_available_strategies()}")
    
    strategy_info = QuantumEncodingFactory.get_strategy_info()
    print(f"\nStrategy descriptions:")
    for name, description in strategy_info.items():
        print(f"  {name}: {description}")
    
    print(f"\n✓ All quantum encoding tests completed!")

if __name__ == "__main__":
    test_quantum_encodings()
