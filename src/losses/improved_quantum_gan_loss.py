"""
Improved Quantum GAN Loss with Zero-Centered Gradient Penalty

This module implements an improved loss function that addresses discriminator
gradient collapse through zero-centered gradient penalty and better regularization.
"""

import tensorflow as tf
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class ImprovedQuantumWassersteinLoss:
    """
    Improved Wasserstein loss with zero-centered gradient penalty.
    
    This loss function addresses discriminator gradient collapse by:
    - Using zero-centered gradient penalty instead of 1-centered
    - Adding noise regularization to prevent overconfidence
    - Better balancing of loss components
    """
    
    def __init__(self, 
                 lambda_gp=1.0,  # Reduced from 10.0
                 lambda_entropy=1.0,
                 lambda_physics=1.0,
                 gp_center=0.0,  # Zero-centered instead of 1.0
                 noise_std=0.01):  # Add noise to inputs
        """
        Initialize improved quantum Wasserstein loss.
        
        Args:
            lambda_gp: Gradient penalty weight (reduced)
            lambda_entropy: Quantum entropy regularization weight
            lambda_physics: Physics constraint weight
            gp_center: Target gradient norm (0.0 for zero-centered GP)
            noise_std: Standard deviation of noise added to discriminator inputs
        """
        self.lambda_gp = lambda_gp
        self.lambda_entropy = lambda_entropy
        self.lambda_physics = lambda_physics
        self.gp_center = gp_center
        self.noise_std = noise_std
        
        logger.info(f"Improved Quantum Wasserstein Loss initialized:")
        logger.info(f"  - Gradient penalty: {lambda_gp} (zero-centered: {gp_center})")
        logger.info(f"  - Entropy regularization: {lambda_entropy}")
        logger.info(f"  - Physics constraints: {lambda_physics}")
        logger.info(f"  - Input noise std: {noise_std}")
    
    def __call__(self, real_samples, fake_samples, generator, discriminator):
        """
        Compute improved quantum-enhanced Wasserstein loss.
        
        Args:
            real_samples: Real training data
            fake_samples: Generated samples
            generator: Quantum generator
            discriminator: Quantum discriminator
            
        Returns:
            tuple: (discriminator_loss, generator_loss, metrics_dict)
        """
        batch_size = tf.shape(real_samples)[0]
        
        # Add noise to inputs to prevent discriminator overconfidence
        if self.noise_std > 0:
            real_noisy = real_samples + tf.random.normal(tf.shape(real_samples), stddev=self.noise_std)
            fake_noisy = fake_samples + tf.random.normal(tf.shape(fake_samples), stddev=self.noise_std)
        else:
            real_noisy = real_samples
            fake_noisy = fake_samples
        
        # 1. Wasserstein distance
        real_output = discriminator.discriminate(real_noisy)
        fake_output = discriminator.discriminate(fake_noisy)
        w_distance = tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)
        
        # 2. Zero-centered gradient penalty
        gradient_penalty = self._compute_zero_centered_gp(
            real_samples, fake_samples, discriminator
        )
        
        # 3. Quantum regularization
        try:
            quantum_cost = generator.compute_quantum_cost()
            entropy_bonus = self.lambda_entropy * quantum_cost
        except Exception as e:
            logger.warning(f"Quantum metrics computation failed: {e}")
            entropy_bonus = tf.constant(0.0)
        
        # 4. Discriminator regularization (prevent overconfidence)
        real_output_penalty = 0.001 * tf.reduce_mean(tf.square(real_output))
        fake_output_penalty = 0.001 * tf.reduce_mean(tf.square(fake_output))
        output_penalty = real_output_penalty + fake_output_penalty
        
        # Final losses
        d_loss = -w_distance + gradient_penalty + output_penalty
        g_loss = -tf.reduce_mean(fake_output) - entropy_bonus
        
        # Metrics for monitoring
        metrics = {
            'w_distance': w_distance,
            'gradient_penalty': gradient_penalty,
            'entropy_bonus': entropy_bonus,
            'output_penalty': output_penalty,
            'real_output_mean': tf.reduce_mean(real_output),
            'fake_output_mean': tf.reduce_mean(fake_output),
            'real_output_std': tf.math.reduce_std(real_output),
            'fake_output_std': tf.math.reduce_std(fake_output),
            'total_d_loss': d_loss,
            'total_g_loss': g_loss
        }
        
        return d_loss, g_loss, metrics
    
    def _compute_zero_centered_gp(self, real_samples, fake_samples, discriminator):
        """
        Compute zero-centered gradient penalty.
        
        This encourages gradients to be close to zero rather than 1,
        which can help prevent gradient explosion/vanishing.
        """
        # Ensure same batch size
        real_batch_size = tf.shape(real_samples)[0]
        fake_batch_size = tf.shape(fake_samples)[0]
        min_batch_size = tf.minimum(real_batch_size, fake_batch_size)
        
        real_truncated = real_samples[:min_batch_size]
        fake_truncated = fake_samples[:min_batch_size]
        
        # Random interpolation
        alpha = tf.random.uniform([min_batch_size, 1], 0.0, 1.0)
        interpolated = alpha * real_truncated + (1 - alpha) * fake_truncated
        
        # Compute gradients
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            interp_output = discriminator.discriminate(interpolated)
        
        gradients = gp_tape.gradient(interp_output, interpolated)
        
        if gradients is None:
            logger.warning("Zero-centered GP: gradients are None")
            return tf.constant(0.0)
        
        # Zero-centered gradient penalty
        gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        gradient_penalty = self.lambda_gp * tf.reduce_mean(tf.square(gradient_norm - self.gp_center))
        
        return gradient_penalty


class SpectralNormalizedLoss:
    """
    Loss function specifically designed for spectral normalized discriminators.
    
    This loss takes advantage of spectral normalization properties to
    provide more stable training.
    """
    
    def __init__(self, 
                 lambda_gp=0.1,  # Much reduced since spectral norm handles this
                 lambda_entropy=1.0,
                 spectral_reg=0.01):  # Regularization on spectral norms
        """
        Initialize spectral normalized loss.
        
        Args:
            lambda_gp: Gradient penalty weight (reduced due to spectral norm)
            lambda_entropy: Entropy regularization weight
            spectral_reg: Spectral norm regularization weight
        """
        self.lambda_gp = lambda_gp
        self.lambda_entropy = lambda_entropy
        self.spectral_reg = spectral_reg
        
        logger.info(f"Spectral Normalized Loss initialized:")
        logger.info(f"  - Gradient penalty: {lambda_gp} (reduced)")
        logger.info(f"  - Entropy regularization: {lambda_entropy}")
        logger.info(f"  - Spectral regularization: {spectral_reg}")
    
    def __call__(self, real_samples, fake_samples, generator, discriminator):
        """
        Compute loss for spectral normalized discriminator.
        
        Args:
            real_samples: Real training data
            fake_samples: Generated samples
            generator: Quantum generator
            discriminator: Spectral normalized discriminator
            
        Returns:
            tuple: (discriminator_loss, generator_loss, metrics_dict)
        """
        # 1. Basic Wasserstein distance
        real_output = discriminator.discriminate(real_samples)
        fake_output = discriminator.discriminate(fake_samples)
        w_distance = tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)
        
        # 2. Minimal gradient penalty (spectral norm handles most of this)
        if self.lambda_gp > 0:
            gradient_penalty = self._compute_minimal_gp(real_samples, fake_samples, discriminator)
        else:
            gradient_penalty = tf.constant(0.0)
        
        # 3. Spectral norm regularization
        spectral_penalty = self._compute_spectral_penalty(discriminator)
        
        # 4. Quantum regularization
        try:
            quantum_cost = generator.compute_quantum_cost()
            entropy_bonus = self.lambda_entropy * quantum_cost
        except:
            entropy_bonus = tf.constant(0.0)
        
        # Final losses
        d_loss = -w_distance + gradient_penalty + spectral_penalty
        g_loss = -tf.reduce_mean(fake_output) - entropy_bonus
        
        # Metrics
        metrics = {
            'w_distance': w_distance,
            'gradient_penalty': gradient_penalty,
            'spectral_penalty': spectral_penalty,
            'entropy_bonus': entropy_bonus,
            'real_output_mean': tf.reduce_mean(real_output),
            'fake_output_mean': tf.reduce_mean(fake_output),
            'total_d_loss': d_loss,
            'total_g_loss': g_loss
        }
        
        return d_loss, g_loss, metrics
    
    def _compute_minimal_gp(self, real_samples, fake_samples, discriminator):
        """Compute minimal gradient penalty for spectral normalized discriminator."""
        batch_size = tf.minimum(tf.shape(real_samples)[0], tf.shape(fake_samples)[0])
        
        real_truncated = real_samples[:batch_size]
        fake_truncated = fake_samples[:batch_size]
        
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        interpolated = alpha * real_truncated + (1 - alpha) * fake_truncated
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            interp_output = discriminator.discriminate(interpolated)
        
        gradients = tape.gradient(interp_output, interpolated)
        
        if gradients is None:
            return tf.constant(0.0)
        
        gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        # Very mild penalty - spectral norm does the heavy lifting
        gradient_penalty = self.lambda_gp * tf.reduce_mean(tf.square(gradient_norm))
        
        return gradient_penalty
    
    def _compute_spectral_penalty(self, discriminator):
        """Compute penalty on spectral norms to encourage them to stay near 1."""
        if hasattr(discriminator, 'get_spectral_norms'):
            spectral_norms = discriminator.get_spectral_norms()
            
            penalty = 0.0
            for name, norm in spectral_norms.items():
                # Encourage spectral norms to stay close to 1
                penalty += tf.square(norm - 1.0)
            
            return self.spectral_reg * penalty
        else:
            return tf.constant(0.0)


def create_improved_loss(loss_type: str = 'zero_centered_gp', **kwargs):
    """
    Factory function for improved loss functions.
    
    Args:
        loss_type: Type of improved loss
        **kwargs: Additional arguments
        
    Returns:
        Loss function instance
    """
    if loss_type == 'zero_centered_gp':
        return ImprovedQuantumWassersteinLoss(**kwargs)
    elif loss_type == 'spectral_normalized':
        return SpectralNormalizedLoss(**kwargs)
    else:
        raise ValueError(f"Unknown improved loss type: {loss_type}")


def test_improved_losses():
    """Test improved loss functions."""
    print("🧪 Testing Improved Loss Functions...")
    
    # Create dummy data
    real_samples = tf.random.normal([8, 2])
    fake_samples = tf.random.normal([8, 2])
    
    # Mock generator and discriminator
    class MockGenerator:
        def compute_quantum_cost(self):
            return tf.constant(0.1)
    
    class MockDiscriminator:
        def discriminate(self, x):
            return tf.random.normal([tf.shape(x)[0], 1])
        
        def get_spectral_norms(self):
            return {'layer1': 1.2, 'layer2': 0.9}
    
    generator = MockGenerator()
    discriminator = MockDiscriminator()
    
    # Test zero-centered GP loss
    print("Testing zero-centered GP loss...")
    loss_fn = ImprovedQuantumWassersteinLoss(lambda_gp=1.0, gp_center=0.0)
    d_loss, g_loss, metrics = loss_fn(real_samples, fake_samples, generator, discriminator)
    
    print(f"✅ Zero-centered GP: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}")
    print(f"   Metrics: {list(metrics.keys())}")
    
    # Test spectral normalized loss
    print("Testing spectral normalized loss...")
    loss_fn = SpectralNormalizedLoss(lambda_gp=0.1, spectral_reg=0.01)
    d_loss, g_loss, metrics = loss_fn(real_samples, fake_samples, generator, discriminator)
    
    print(f"✅ Spectral normalized: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}")
    print(f"   Metrics: {list(metrics.keys())}")
    
    return True


if __name__ == "__main__":
    test_improved_losses()
