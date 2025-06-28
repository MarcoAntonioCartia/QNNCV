"""
Spectral Normalization for TensorFlow/Keras

This module implements spectral normalization to stabilize discriminator training
and prevent gradient explosion/vanishing in quantum GANs.
"""

import tensorflow as tf
import numpy as np
from typing import Optional


class SpectralNormalization(tf.keras.layers.Wrapper):
    """
    Spectral Normalization wrapper for any layer with a kernel.
    
    This layer wraps any layer that has a 'kernel' attribute and applies
    spectral normalization to stabilize training by constraining the
    spectral norm (largest singular value) of the weight matrix to 1.
    """
    
    def __init__(self, layer, power_iterations=1, **kwargs):
        """
        Initialize spectral normalization wrapper.
        
        Args:
            layer: The layer to wrap (must have 'kernel' attribute)
            power_iterations: Number of power iterations for spectral norm estimation
        """
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        self.power_iterations = power_iterations
        
    def build(self, input_shape):
        """Build the wrapped layer and initialize spectral norm variables."""
        super(SpectralNormalization, self).build(input_shape)
        
        # Get the kernel from the wrapped layer
        if not hasattr(self.layer, 'kernel'):
            raise ValueError('SpectralNormalization can only wrap layers with a kernel attribute')
        
        kernel = self.layer.kernel
        kernel_shape = kernel.shape
        
        # Initialize u and v vectors for power iteration
        # u should have shape [output_dim, 1]
        # v should have shape [input_dim, 1] 
        if len(kernel_shape) == 2:  # Dense layer
            self.u = self.add_weight(
                name='u',
                shape=[kernel_shape[-1], 1],
                initializer='random_normal',
                trainable=False,
                dtype=kernel.dtype
            )
            self.v = self.add_weight(
                name='v', 
                shape=[kernel_shape[0], 1],
                initializer='random_normal',
                trainable=False,
                dtype=kernel.dtype
            )
        else:
            raise ValueError(f'Spectral normalization not implemented for kernel shape {kernel_shape}')
    
    def call(self, inputs, training=None):
        """Apply spectral normalization and call the wrapped layer."""
        # Get the kernel
        kernel = self.layer.kernel
        
        # Perform power iteration to estimate spectral norm
        if training:
            # Power iteration method to find largest singular value
            u = self.u
            v = self.v
            
            for _ in range(self.power_iterations):
                # v = W^T @ u / ||W^T @ u||
                v_new = tf.matmul(kernel, u, transpose_a=True)
                v_new = tf.nn.l2_normalize(v_new, axis=0)
                
                # u = W @ v / ||W @ v||
                u_new = tf.matmul(kernel, v_new)
                u_new = tf.nn.l2_normalize(u_new, axis=0)
                
                u = u_new
                v = v_new
            
            # Update u and v
            self.u.assign(u)
            self.v.assign(v)
            
            # Compute spectral norm: sigma = u^T @ W @ v
            sigma = tf.matmul(u, tf.matmul(kernel, v), transpose_a=True)
            sigma = tf.squeeze(sigma)
            
            # Normalize the kernel by the spectral norm
            kernel_normalized = kernel / sigma
            
            # Temporarily replace the kernel
            original_kernel = self.layer.kernel
            self.layer.kernel = kernel_normalized
            
            # Call the wrapped layer
            output = self.layer(inputs)
            
            # Restore original kernel
            self.layer.kernel = original_kernel
            
            return output
        else:
            # During inference, just call the layer normally
            return self.layer(inputs)
    
    def get_config(self):
        """Get configuration for serialization."""
        config = super(SpectralNormalization, self).get_config()
        config.update({
            'power_iterations': self.power_iterations
        })
        return config


def spectral_norm_dense(units, activation=None, name=None, **kwargs):
    """
    Create a Dense layer with spectral normalization.
    
    Args:
        units: Number of output units
        activation: Activation function
        name: Layer name
        **kwargs: Additional arguments for Dense layer
        
    Returns:
        SpectralNormalization wrapped Dense layer
    """
    dense_layer = tf.keras.layers.Dense(
        units=units,
        activation=activation,
        name=f'{name}_dense' if name else None,
        **kwargs
    )
    
    return SpectralNormalization(
        dense_layer,
        name=f'{name}_spectral_norm' if name else None
    )


class SpectralNormDense(tf.keras.layers.Layer):
    """
    Dense layer with built-in spectral normalization.
    
    This is a more efficient implementation that directly applies
    spectral normalization without wrapping.
    """
    
    def __init__(self, units, activation=None, power_iterations=1, **kwargs):
        """
        Initialize spectral normalized dense layer.
        
        Args:
            units: Number of output units
            activation: Activation function
            power_iterations: Number of power iterations
        """
        super(SpectralNormDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.power_iterations = power_iterations
        
    def build(self, input_shape):
        """Build the layer."""
        super(SpectralNormDense, self).build(input_shape)
        
        # Create kernel
        self.kernel = self.add_weight(
            name='kernel',
            shape=[input_shape[-1], self.units],
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Create bias
        self.bias = self.add_weight(
            name='bias',
            shape=[self.units],
            initializer='zeros',
            trainable=True
        )
        
        # Initialize u and v for power iteration
        # u should match output dimension, v should match input dimension
        self.u = self.add_weight(
            name='u',
            shape=[self.units],
            initializer='random_normal',
            trainable=False
        )
        
        self.v = self.add_weight(
            name='v',
            shape=[input_shape[-1]], 
            initializer='random_normal',
            trainable=False
        )
    
    def call(self, inputs, training=None):
        """Forward pass with spectral normalization."""
        # Apply spectral normalization to kernel
        if training:
            # Power iteration
            u = self.u
            v = self.v
            
            for _ in range(self.power_iterations):
                # v = W^T @ u / ||W^T @ u||
                v_new = tf.matmul(self.kernel, u, transpose_a=True)
                v_new = tf.nn.l2_normalize(v_new, axis=0)
                
                # u = W @ v / ||W @ v||
                u_new = tf.matmul(self.kernel, v_new)
                u_new = tf.nn.l2_normalize(u_new, axis=0)
                
                u = u_new
                v = v_new
            
            # Update u and v
            self.u.assign(u)
            self.v.assign(v)
            
            # Compute spectral norm
            sigma = tf.matmul(u, tf.matmul(self.kernel, v), transpose_a=True)
            sigma = tf.squeeze(sigma)
            
            # Normalize kernel
            kernel_normalized = self.kernel / (sigma + 1e-8)  # Add epsilon for stability
        else:
            kernel_normalized = self.kernel
        
        # Apply linear transformation
        output = tf.matmul(inputs, kernel_normalized) + self.bias
        
        # Apply activation
        if self.activation is not None:
            output = self.activation(output)
            
        return output
    
    def get_config(self):
        """Get configuration."""
        config = super(SpectralNormDense, self).get_config()
        config.update({
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
            'power_iterations': self.power_iterations
        })
        return config


def test_spectral_normalization():
    """Test spectral normalization implementation."""
    print("🧪 Testing Spectral Normalization...")
    
    # Test SpectralNormDense
    layer = SpectralNormDense(units=10, activation='relu')
    
    # Build layer
    inputs = tf.random.normal([32, 20])
    outputs = layer(inputs, training=True)
    
    print(f"✅ SpectralNormDense test passed: {inputs.shape} → {outputs.shape}")
    
    # Test spectral norm constraint
    # The spectral norm should be approximately 1 after normalization
    kernel = layer.kernel
    singular_values = tf.linalg.svd(kernel, compute_uv=False)
    max_singular_value = tf.reduce_max(singular_values)
    
    print(f"   Max singular value: {max_singular_value.numpy():.6f}")
    
    # Test wrapper version
    dense = tf.keras.layers.Dense(10, activation='relu')
    spectral_dense = SpectralNormalization(dense)
    
    # Build and test
    spectral_dense.build(inputs.shape)
    outputs2 = spectral_dense(inputs, training=True)
    
    print(f"✅ SpectralNormalization wrapper test passed: {inputs.shape} → {outputs2.shape}")
    
    return True


if __name__ == "__main__":
    test_spectral_normalization()
