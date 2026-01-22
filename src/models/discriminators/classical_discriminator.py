"""
Classical Discriminator
=======================

Simple MLP discriminator for QGAN testing.

This provides a baseline classical discriminator that can be used
with the quantum generator. For a true quantum GAN comparison,
replace this with QuantumSFDiscriminator.
"""

import tensorflow as tf
from typing import List


class ClassicalDiscriminator:
    """
    Simple MLP Discriminator.
    
    Architecture:
    - Input layer
    - Hidden layers with LeakyReLU
    - Output layer (no activation for WGAN)
    
    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension (usually 1)
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: List[int] = None,
        output_dim: int = 1
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [32, 32]
        self.output_dim = output_dim
        
        # Build model
        self.model = self._build_model()
    
    def _build_model(self) -> tf.keras.Model:
        """Build the discriminator model."""
        layers = []
        
        # Input layer
        layers.append(tf.keras.layers.InputLayer(input_shape=(self.input_dim,)))
        
        # Hidden layers
        for dim in self.hidden_dims:
            layers.append(tf.keras.layers.Dense(dim))
            layers.append(tf.keras.layers.LeakyReLU(alpha=0.2))
        
        # Output layer (no activation for WGAN)
        layers.append(tf.keras.layers.Dense(self.output_dim))
        
        return tf.keras.Sequential(layers)
    
    def discriminate(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute discriminator scores.
        
        Args:
            x: Input samples [batch_size, input_dim]
            
        Returns:
            Scores [batch_size, output_dim]
        """
        return self.model(x)
    
    @property
    def trainable_variables(self):
        """Return trainable variables."""
        return self.model.trainable_variables
    
    @property
    def num_params(self) -> int:
        """Total number of parameters."""
        return sum(tf.reduce_prod(v.shape).numpy() for v in self.trainable_variables)
    
    def get_config(self) -> dict:
        """Return configuration."""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'num_params': self.num_params
        }


if __name__ == "__main__":
    print("Testing ClassicalDiscriminator...")
    
    disc = ClassicalDiscriminator(input_dim=1, hidden_dims=[32, 32])
    print(f"Config: {disc.get_config()}")
    
    x = tf.random.normal([4, 1])
    scores = disc.discriminate(x)
    print(f"Input shape: {x.shape}, Output shape: {scores.shape}")
    
    # Test gradients
    with tf.GradientTape() as tape:
        scores = disc.discriminate(x)
        loss = tf.reduce_mean(tf.square(scores))
    
    grads = tape.gradient(loss, disc.trainable_variables)
    print(f"Gradients computed: {all(g is not None for g in grads)}")
    
    print("✓ ClassicalDiscriminator test passed!")
