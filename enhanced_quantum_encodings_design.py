"""
Enhanced Quantum Encoding Factory Design - More General Architecture

This design addresses the limitations of the current QuantumEncodingFactory and
provides a more flexible, context-aware, and extensible architecture for quantum
encoding strategies across different quantum ML components.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional, Tuple, Union, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class EncodingContext(Enum):
    """Defines the context in which encoding is being used."""
    GENERATOR = "generator"           # Latent → Quantum parameters for generation
    DISCRIMINATOR = "discriminator"   # Data → Quantum features for classification
    CLASSIFIER = "classifier"        # Data → Quantum features for classification
    AUTOENCODER = "autoencoder"      # Data → Quantum representation
    CUSTOM = "custom"                # User-defined context

class OutputType(Enum):
    """Defines the type of output expected from encoding."""
    QUANTUM_PARAMS = "quantum_params"     # Parameters for quantum circuit
    QUANTUM_FEATURES = "quantum_features" # Features extracted from quantum state
    COHERENT_AMPLITUDES = "coherent_amplitudes"  # Complex amplitudes for coherent states
    DISPLACEMENT_PARAMS = "displacement_params"   # Displacement gate parameters
    ROTATION_ANGLES = "rotation_angles"           # Rotation gate angles
    CUSTOM = "custom"                             # User-defined output type

class QuantumEncodingConfig:
    """Configuration class for quantum encoding strategies."""
    
    def __init__(self, 
                 context: EncodingContext,
                 output_type: OutputType,
                 n_modes: int,
                 n_layers: int = 1,
                 input_dim: Optional[int] = None,
                 target_dim: Optional[int] = None,
                 scaling_factor: float = 1.0,
                 stability_constraints: bool = True,
                 **kwargs):
        """
        Initialize encoding configuration.
        
        Args:
            context: The context in which encoding is used
            output_type: The type of output expected
            n_modes: Number of quantum modes
            n_layers: Number of quantum layers
            input_dim: Input dimension (auto-detected if None)
            target_dim: Target output dimension (auto-calculated if None)
            scaling_factor: Scaling factor for output values
            stability_constraints: Whether to apply stability constraints
            **kwargs: Additional context-specific parameters
        """
        self.context = context
        self.output_type = output_type
        self.n_modes = n_modes
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.scaling_factor = scaling_factor
        self.stability_constraints = stability_constraints
        self.extra_params = kwargs
    
    def auto_configure(self, input_shape: Tuple[int, ...]) -> 'QuantumEncodingConfig':
        """Auto-configure parameters based on input shape and context."""
        if self.input_dim is None:
            self.input_dim = input_shape[-1]
        
        if self.target_dim is None:
            self.target_dim = self._calculate_target_dim()
        
        return self
    
    def _calculate_target_dim(self) -> int:
        """Calculate target dimension based on context and output type."""
        if self.output_type == OutputType.COHERENT_AMPLITUDES:
            return 2 * self.n_modes  # Real + imaginary parts
        elif self.output_type == OutputType.DISPLACEMENT_PARAMS:
            return self.n_modes
        elif self.output_type == OutputType.ROTATION_ANGLES:
            return self.n_modes * self.n_layers
        elif self.output_type == OutputType.QUANTUM_PARAMS:
            # Standard SF circuit parameter count
            M = int(self.n_modes * (self.n_modes - 1)) + max(1, self.n_modes - 1)
            return self.n_layers * (2 * M + 4 * self.n_modes)
        elif self.output_type == OutputType.QUANTUM_FEATURES:
            # Context-dependent feature count
            if self.context == EncodingContext.DISCRIMINATOR:
                return self.n_modes * 3  # Position, momentum, photon number per mode
            else:
                return self.n_modes
        else:
            return self.n_modes  # Default fallback

class EnhancedQuantumEncodingStrategy:
    """Enhanced base class for quantum encoding strategies with context awareness."""
    
    def __init__(self, name: str):
        self.name = name
        self._cached_networks = {}  # Cache for different configurations
        
    def encode(self, 
               data: tf.Tensor, 
               config: QuantumEncodingConfig,
               **kwargs) -> tf.Tensor:
        """
        Context-aware encoding of classical data.
        
        Args:
            data: Input data tensor
            config: Encoding configuration
            **kwargs: Additional parameters
            
        Returns:
            Encoded output according to configuration
        """
        # Auto-configure if needed
        config = config.auto_configure(data.shape)
        
        # Apply context-specific preprocessing
        preprocessed_data = self._preprocess_data(data, config)
        
        # Perform encoding
        encoded = self._encode_implementation(preprocessed_data, config, **kwargs)
        
        # Apply context-specific postprocessing
        postprocessed = self._postprocess_output(encoded, config)
        
        return postprocessed
    
    def _preprocess_data(self, data: tf.Tensor, config: QuantumEncodingConfig) -> tf.Tensor:
        """Context-specific data preprocessing."""
        if config.context == EncodingContext.GENERATOR:
            # Generator typically uses noise, may need normalization
            return tf.nn.tanh(data)  # Bound to [-1, 1]
        elif config.context == EncodingContext.DISCRIMINATOR:
            # Discriminator uses real data, may need different scaling
            return data * config.scaling_factor
        else:
            return data
    
    def _postprocess_output(self, output: tf.Tensor, config: QuantumEncodingConfig) -> tf.Tensor:
        """Context-specific output postprocessing."""
        if config.stability_constraints:
            if config.output_type == OutputType.COHERENT_AMPLITUDES:
                # Limit amplitude magnitude for stability
                return output * 0.5
            elif config.output_type == OutputType.DISPLACEMENT_PARAMS:
                # Limit displacement for stability
                return tf.nn.tanh(output) * 0.3
            elif config.output_type == OutputType.ROTATION_ANGLES:
                # Map to [0, 2π] range
                return tf.nn.sigmoid(output) * 2 * np.pi
        
        return output
    
    def _encode_implementation(self, 
                             data: tf.Tensor, 
                             config: QuantumEncodingConfig,
                             **kwargs) -> tf.Tensor:
        """Subclasses implement the actual encoding logic."""
        raise NotImplementedError("Subclasses must implement _encode_implementation")
    
    def get_parameter_count(self, config: QuantumEncodingConfig) -> int:
        """Get number of parameters for given configuration."""
        return config.target_dim
    
    def supports_context(self, context: EncodingContext) -> bool:
        """Check if this strategy supports the given context."""
        return True  # Base implementation supports all contexts
    
    def supports_output_type(self, output_type: OutputType) -> bool:
        """Check if this strategy supports the given output type."""
        return True  # Base implementation supports all output types

class AdaptiveCoherentStateEncoding(EnhancedQuantumEncodingStrategy):
    """
    Adaptive coherent state encoding that adjusts behavior based on context.
    """
    
    def __init__(self):
        super().__init__("adaptive_coherent_state")
    
    def _encode_implementation(self, 
                             data: tf.Tensor, 
                             config: QuantumEncodingConfig,
                             **kwargs) -> tf.Tensor:
        """Context-aware coherent state encoding."""
        batch_size = tf.shape(data)[0]
        input_dim = data.shape[-1]
        
        if config.context == EncodingContext.GENERATOR:
            # Generator: Use latent noise to create diverse coherent states
            return self._encode_for_generation(data, config)
        elif config.context == EncodingContext.DISCRIMINATOR:
            # Discriminator: Extract features from real/fake data
            return self._encode_for_discrimination(data, config)
        else:
            # Default behavior
            return self._encode_default(data, config)
    
    def _encode_for_generation(self, data: tf.Tensor, config: QuantumEncodingConfig) -> tf.Tensor:
        """Encoding optimized for generation tasks."""
        input_dim = data.shape[-1]
        
        # Split into real and imaginary parts
        if input_dim >= 2 * config.n_modes:
            real_parts = data[..., :config.n_modes]
            imag_parts = data[..., config.n_modes:2*config.n_modes]
        else:
            # Use learnable transformation for insufficient dimensions
            network_key = f"gen_{input_dim}_{config.n_modes}"
            if network_key not in self._cached_networks:
                self._cached_networks[network_key] = tf.keras.Sequential([
                    tf.keras.layers.Dense(config.n_modes * 2, activation='tanh'),
                    tf.keras.layers.Reshape([2, config.n_modes])
                ])
            
            transformed = self._cached_networks[network_key](data)
            real_parts = transformed[:, 0, :]
            imag_parts = transformed[:, 1, :]
        
        # Create complex amplitudes
        amplitudes = tf.complex(real_parts, imag_parts)
        
        if config.output_type == OutputType.COHERENT_AMPLITUDES:
            return tf.concat([tf.math.real(amplitudes), tf.math.imag(amplitudes)], axis=-1)
        else:
            # Convert to requested output type
            return self._convert_amplitudes(amplitudes, config)
    
    def _encode_for_discrimination(self, data: tf.Tensor, config: QuantumEncodingConfig) -> tf.Tensor:
        """Encoding optimized for discrimination tasks."""
        input_dim = data.shape[-1]
        
        # Use learnable transformation to extract quantum features
        network_key = f"disc_{input_dim}_{config.target_dim}"
        if network_key not in self._cached_networks:
            self._cached_networks[network_key] = tf.keras.Sequential([
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(config.target_dim, activation='tanh')
            ])
        
        features = self._cached_networks[network_key](data)
        
        if config.output_type == OutputType.QUANTUM_FEATURES:
            # Return multiple quantum observables
            return self._extract_quantum_features(features, config)
        else:
            return features
    
    def _encode_default(self, data: tf.Tensor, config: QuantumEncodingConfig) -> tf.Tensor:
        """Default encoding behavior."""
        # Simple linear transformation
        input_dim = data.shape[-1]
        
        if input_dim >= config.target_dim:
            return data[..., :config.target_dim]
        else:
            # Pad or repeat
            repeats = config.target_dim // input_dim + 1
            repeated = tf.tile(data, [1, repeats])
            return repeated[..., :config.target_dim]
    
    def _convert_amplitudes(self, amplitudes: tf.Tensor, config: QuantumEncodingConfig) -> tf.Tensor:
        """Convert coherent amplitudes to other output types."""
        if config.output_type == OutputType.DISPLACEMENT_PARAMS:
            return tf.math.real(amplitudes)  # Use real part as displacement
        elif config.output_type == OutputType.ROTATION_ANGLES:
            phases = tf.math.angle(amplitudes)
            return tf.tile(phases, [1, config.n_layers])
        else:
            return tf.concat([tf.math.real(amplitudes), tf.math.imag(amplitudes)], axis=-1)
    
    def _extract_quantum_features(self, features: tf.Tensor, config: QuantumEncodingConfig) -> tf.Tensor:
        """Extract multiple quantum observables for discrimination."""
        # Reshape features to represent different observables per mode
        n_observables = config.target_dim // config.n_modes
        reshaped = tf.reshape(features, [-1, config.n_modes, n_observables])
        
        # Apply different transformations for different observables
        position = tf.nn.tanh(reshaped[..., 0])  # Position quadrature
        momentum = tf.nn.tanh(reshaped[..., 1]) if n_observables > 1 else tf.zeros_like(position)  # Momentum quadrature
        photon_num = tf.nn.softplus(reshaped[..., 2]) if n_observables > 2 else tf.zeros_like(position)  # Photon number
        
        return tf.concat([position, momentum, photon_num], axis=-1)
    
    def supports_output_type(self, output_type: OutputType) -> bool:
        """This strategy supports multiple output types."""
        supported = [
            OutputType.COHERENT_AMPLITUDES,
            OutputType.DISPLACEMENT_PARAMS,
            OutputType.ROTATION_ANGLES,
            OutputType.QUANTUM_FEATURES
        ]
        return output_type in supported

class EnhancedQuantumEncodingFactory:
    """
    Enhanced factory with context awareness and automatic strategy selection.
    """
    
    _strategies = {
        'adaptive_coherent_state': AdaptiveCoherentStateEncoding,
        # Add more enhanced strategies here
    }
    
    _context_recommendations = {
        EncodingContext.GENERATOR: ['adaptive_coherent_state'],
        EncodingContext.DISCRIMINATOR: ['adaptive_coherent_state'],
        EncodingContext.CLASSIFIER: ['adaptive_coherent_state'],
    }
    
    @classmethod
    def create_encoding(cls, 
                       strategy_name: str = None,
                       context: EncodingContext = None,
                       output_type: OutputType = None,
                       auto_select: bool = False,
                       **kwargs) -> EnhancedQuantumEncodingStrategy:
        """
        Create an enhanced quantum encoding strategy.
        
        Args:
            strategy_name: Specific strategy name (optional if auto_select=True)
            context: Encoding context for auto-selection
            output_type: Desired output type
            auto_select: Whether to automatically select best strategy
            **kwargs: Strategy-specific parameters
            
        Returns:
            Enhanced quantum encoding strategy instance
        """
        if auto_select and context is not None:
            strategy_name = cls._recommend_strategy(context, output_type)
        
        if strategy_name is None:
            raise ValueError("Must provide strategy_name or enable auto_select with context")
        
        if strategy_name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {available}")
        
        strategy_class = cls._strategies[strategy_name]
        return strategy_class()
    
    @classmethod
    def _recommend_strategy(cls, 
                          context: EncodingContext, 
                          output_type: OutputType = None) -> str:
        """Recommend best strategy for given context and output type."""
        recommendations = cls._context_recommendations.get(context, [])
        
        if not recommendations:
            return list(cls._strategies.keys())[0]  # Default to first available
        
        # For now, return first recommendation
        # In future, could implement more sophisticated selection logic
        return recommendations[0]
    
    @classmethod
    def create_config(cls,
                     context: EncodingContext,
                     output_type: OutputType,
                     n_modes: int,
                     **kwargs) -> QuantumEncodingConfig:
        """Create a configuration for quantum encoding."""
        return QuantumEncodingConfig(
            context=context,
            output_type=output_type,
            n_modes=n_modes,
            **kwargs
        )
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available enhanced strategies."""
        return list(cls._strategies.keys())
    
    @classmethod
    def get_context_recommendations(cls, context: EncodingContext) -> List[str]:
        """Get recommended strategies for a specific context."""
        return cls._context_recommendations.get(context, [])

# Example usage demonstrating the enhanced flexibility
def example_usage():
    """Demonstrate the enhanced quantum encoding factory."""
    
    # Generator use case
    generator_config = EnhancedQuantumEncodingFactory.create_config(
        context=EncodingContext.GENERATOR,
        output_type=OutputType.COHERENT_AMPLITUDES,
        n_modes=2,
        n_layers=3
    )
    
    generator_encoding = EnhancedQuantumEncodingFactory.create_encoding(
        auto_select=True,
        context=EncodingContext.GENERATOR
    )
    
    # Discriminator use case
    discriminator_config = EnhancedQuantumEncodingFactory.create_config(
        context=EncodingContext.DISCRIMINATOR,
        output_type=OutputType.QUANTUM_FEATURES,
        n_modes=2,
        scaling_factor=0.5
    )
    
    discriminator_encoding = EnhancedQuantumEncodingFactory.create_encoding(
        auto_select=True,
        context=EncodingContext.DISCRIMINATOR
    )
    
    # Test data
    latent_data = tf.random.normal([4, 6])  # Generator input
    real_data = tf.random.normal([4, 2])    # Discriminator input
    
    # Encode for generator
    gen_output = generator_encoding.encode(latent_data, generator_config)
    print(f"Generator output shape: {gen_output.shape}")
    
    # Encode for discriminator
    disc_output = discriminator_encoding.encode(real_data, discriminator_config)
    print(f"Discriminator output shape: {disc_output.shape}")

if __name__ == "__main__":
    example_usage()
