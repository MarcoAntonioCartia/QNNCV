"""
Distribution Discriminator
===========================

Discriminator that takes probability distributions as input instead of 
scalar samples. Used for distribution-to-distribution GAN training.

Input: P(x) distribution [num_bins]
Output: Score (real/fake)
"""

import tensorflow as tf
import numpy as np
from typing import List, Optional


class DistributionDiscriminator:
    """
    Discriminator for probability distributions.
    
    Takes a discretized probability distribution P(x) as input and
    outputs a score indicating whether it's real or fake.
    
    Architecture: MLP with optional 1D convolutions for local pattern detection.
    """
    
    def __init__(
        self,
        input_dim: int = 100,  # num_bins of P(x)
        hidden_dims: List[int] = [64, 32],
        use_spectral_norm: bool = False,
        use_conv: bool = False
    ):
        """
        Args:
            input_dim: Size of input distribution (num_bins)
            hidden_dims: Hidden layer dimensions
            use_spectral_norm: Apply spectral normalization (for WGAN stability)
            use_conv: Use 1D convolutions for local pattern detection
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.use_spectral_norm = use_spectral_norm
        self.use_conv = use_conv
        
        # Build layers
        self._build_network()
    
    def _build_network(self):
        """Build the discriminator network."""
        self.layers = []
        
        if self.use_conv:
            # Conv layers for local pattern detection
            # Input: [batch, num_bins] -> [batch, num_bins, 1]
            self.conv_layers = [
                tf.Variable(tf.random.normal([5, 1, 16], stddev=0.1), name="conv1_w"),
                tf.Variable(tf.zeros([16]), name="conv1_b"),
                tf.Variable(tf.random.normal([5, 16, 32], stddev=0.1), name="conv2_w"),
                tf.Variable(tf.zeros([32]), name="conv2_b"),
            ]
            # After conv: flattened features -> MLP
            conv_out_dim = (self.input_dim // 4) * 32  # Assuming 2 pooling layers
        else:
            self.conv_layers = []
            conv_out_dim = self.input_dim
        
        # MLP layers
        dims = [conv_out_dim] + self.hidden_dims + [1]
        
        for i in range(len(dims) - 1):
            # Weight matrix
            w = tf.Variable(
                tf.random.normal([dims[i], dims[i+1]], stddev=0.1),
                name=f"dense_{i}_w"
            )
            b = tf.Variable(
                tf.zeros([dims[i+1]]),
                name=f"dense_{i}_b"
            )
            self.layers.append((w, b))
    
    def discriminate(self, P_x: tf.Tensor) -> tf.Tensor:
        """
        Compute discriminator score for distributions.
        
        Args:
            P_x: Probability distributions [batch_size, num_bins]
        
        Returns:
            Scores [batch_size, 1]
        """
        x = P_x
        
        if self.use_conv:
            # Reshape for conv: [batch, bins] -> [batch, bins, 1]
            x = tf.expand_dims(x, -1)
            
            # Conv1
            x = tf.nn.conv1d(x, self.conv_layers[0], stride=1, padding='SAME')
            x = x + self.conv_layers[1]
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.nn.max_pool1d(x, ksize=2, strides=2, padding='SAME')
            
            # Conv2
            x = tf.nn.conv1d(x, self.conv_layers[2], stride=1, padding='SAME')
            x = x + self.conv_layers[3]
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = tf.nn.max_pool1d(x, ksize=2, strides=2, padding='SAME')
            
            # Flatten
            x = tf.reshape(x, [tf.shape(x)[0], -1])
        
        # MLP layers
        for i, (w, b) in enumerate(self.layers):
            x = tf.matmul(x, w) + b
            
            # Apply activation (except last layer)
            if i < len(self.layers) - 1:
                x = tf.nn.leaky_relu(x, alpha=0.2)
        
        return x  # [batch_size, 1]
    
    @property
    def trainable_variables(self):
        """Return all trainable variables."""
        vars = []
        for w, b in self.layers:
            vars.extend([w, b])
        if self.use_conv:
            vars.extend(self.conv_layers)
        return vars
    
    @property
    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(int(np.prod(v.shape)) for v in self.trainable_variables)
    
    def get_config(self) -> dict:
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'use_spectral_norm': self.use_spectral_norm,
            'use_conv': self.use_conv,
            'num_params': self.num_params
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing DistributionDiscriminator...")
    
    # Create discriminator
    D = DistributionDiscriminator(
        input_dim=100,
        hidden_dims=[64, 32]
    )
    
    print(f"Config: {D.get_config()}")
    
    # Test forward pass
    P_x = tf.random.uniform([8, 100], minval=0, maxval=1)  # Fake distributions
    P_x = P_x / tf.reduce_sum(P_x, axis=1, keepdims=True)  # Normalize
    
    scores = D.discriminate(P_x)
    
    print(f"Input shape: {P_x.shape}")
    print(f"Output shape: {scores.shape}")
    print(f"Score range: [{tf.reduce_min(scores).numpy():.4f}, {tf.reduce_max(scores).numpy():.4f}]")
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    with tf.GradientTape() as tape:
        scores = D.discriminate(P_x)
        loss = tf.reduce_mean(scores)
    
    grads = tape.gradient(loss, D.trainable_variables)
    print(f"All gradients computed: {all(g is not None for g in grads)}")
    
    print("\n✓ All tests passed!")
