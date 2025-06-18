"""
Quantum-aware loss functions for GAN training.

This module implements loss functions that maintain gradient flow
through quantum circuits by using measurement probabilities and
quantum state properties.
"""

import tensorflow as tf
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class QuantumGANLoss:
    """
    Quantum-aware GAN loss that ensures gradient flow.
    
    Uses measurement statistics and quantum properties to compute
    differentiable losses for both generator and discriminator.
    """
    
    def __init__(self, loss_type: str = 'wasserstein', 
                 gradient_penalty_weight: float = 10.0):
        """
        Initialize quantum GAN loss.
        
        Args:
            loss_type: Type of loss ('wasserstein', 'standard')
            gradient_penalty_weight: Weight for gradient penalty
        """
        self.loss_type = loss_type
        self.gradient_penalty_weight = gradient_penalty_weight
        
        logger.info(f"Quantum GAN loss initialized: {loss_type}")
    
    def generator_loss(self, 
                      fake_output: tf.Tensor,
                      fake_measurements: Optional[tf.Tensor] = None,
                      real_measurements: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Compute generator loss with quantum awareness.
        
        Args:
            fake_output: Discriminator output for fake samples
            fake_measurements: Raw quantum measurements from generator
            real_measurements: Target measurements (if available)
            
        Returns:
            Generator loss
        """
        if self.loss_type == 'wasserstein':
            # Wasserstein loss: maximize discriminator output
            base_loss = -tf.reduce_mean(fake_output)
        else:
            # Standard GAN loss
            base_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(fake_output),
                    logits=fake_output
                )
            )
        
        # Add quantum-specific regularization if measurements provided
        if fake_measurements is not None and real_measurements is not None:
            # Measurement matching loss
            measurement_loss = self._measurement_matching_loss(
                fake_measurements, real_measurements
            )
            
            # Quantum state regularization
            quantum_reg = self._quantum_regularization(fake_measurements)
            
            total_loss = base_loss + 0.1 * measurement_loss + 0.01 * quantum_reg
        else:
            total_loss = base_loss
        
        return total_loss
    
    def discriminator_loss(self,
                          real_output: tf.Tensor,
                          fake_output: tf.Tensor,
                          interpolated_output: Optional[tf.Tensor] = None,
                          interpolated_samples: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Compute discriminator loss with gradient penalty.
        
        Args:
            real_output: Discriminator output for real samples
            fake_output: Discriminator output for fake samples
            interpolated_output: Output for interpolated samples (for GP)
            interpolated_samples: Interpolated samples (for GP)
            
        Returns:
            Discriminator loss
        """
        if self.loss_type == 'wasserstein':
            # Wasserstein loss
            base_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
            
            # Add gradient penalty if provided
            if interpolated_output is not None and interpolated_samples is not None:
                gp = self._gradient_penalty(interpolated_output, interpolated_samples)
                total_loss = base_loss + self.gradient_penalty_weight * gp
            else:
                total_loss = base_loss
        else:
            # Standard GAN loss
            real_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(real_output),
                    logits=real_output
                )
            )
            fake_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(fake_output),
                    logits=fake_output
                )
            )
            total_loss = real_loss + fake_loss
        
        return total_loss
    
    def _measurement_matching_loss(self, 
                                  fake_measurements: tf.Tensor,
                                  real_measurements: tf.Tensor) -> tf.Tensor:
        """
        Compute loss between measurement distributions.
        
        Args:
            fake_measurements: Generated measurements
            real_measurements: Target measurements
            
        Returns:
            Measurement matching loss
        """
        # Compute statistics of measurements
        fake_mean = tf.reduce_mean(fake_measurements, axis=0)
        real_mean = tf.reduce_mean(real_measurements, axis=0)
        
        fake_var = tf.math.reduce_variance(fake_measurements, axis=0)
        real_var = tf.math.reduce_variance(real_measurements, axis=0)
        
        # Match first and second moments
        mean_loss = tf.reduce_mean(tf.square(fake_mean - real_mean))
        var_loss = tf.reduce_mean(tf.square(fake_var - real_var))
        
        return mean_loss + 0.5 * var_loss
    
    def _quantum_regularization(self, measurements: tf.Tensor) -> tf.Tensor:
        """
        Add quantum-specific regularization.
        
        Args:
            measurements: Quantum measurements
            
        Returns:
            Regularization term
        """
        # Ensure physical constraints
        # For photon number measurements, ensure non-negative
        photon_reg = tf.reduce_mean(tf.nn.relu(-measurements))
        
        # Encourage diversity in measurements
        diversity_reg = -tf.math.reduce_variance(measurements)
        
        return photon_reg + 0.1 * diversity_reg
    
    def _gradient_penalty(self, 
                         interpolated_output: tf.Tensor,
                         interpolated_samples: tf.Tensor) -> tf.Tensor:
        """
        Compute gradient penalty for Wasserstein GAN.
        
        Args:
            interpolated_output: Discriminator output for interpolated samples
            interpolated_samples: Interpolated samples
            
        Returns:
            Gradient penalty
        """
        gradients = tf.gradients(interpolated_output, interpolated_samples)[0]
        
        if gradients is None:
            logger.warning("Gradient penalty: gradients are None")
            return tf.constant(0.0)
        
        # Compute L2 norm of gradients
        gradients_norm = tf.sqrt(
            tf.reduce_sum(tf.square(gradients), axis=list(range(1, len(gradients.shape))))
        )
        
        # Penalize deviation from unit norm
        gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)
        
        return gradient_penalty


class QuantumMeasurementLoss:
    """
    Direct loss on quantum measurements for gradient flow.
    
    This loss function operates directly on the quantum measurements
    to ensure gradients flow through the quantum circuit.
    """
    
    def __init__(self, target_distribution: Optional[str] = None):
        """
        Initialize quantum measurement loss.
        
        Args:
            target_distribution: Target distribution type
        """
        self.target_distribution = target_distribution
        logger.info("Quantum measurement loss initialized")
    
    def compute_loss(self,
                    generated_measurements: tf.Tensor,
                    real_data: tf.Tensor,
                    generator_params: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Compute loss directly from quantum measurements.
        
        Args:
            generated_measurements: Raw quantum measurements from generator
            real_data: Real data samples
            generator_params: Generator parameters (for regularization)
            
        Returns:
            Loss value
        """
        # Transform measurements to match data dimension
        # This is a simple linear projection - could be more sophisticated
        data_dim = real_data.shape[-1]
        measurement_dim = generated_measurements.shape[-1]
        
        if measurement_dim != data_dim:
            # Project measurements to data space
            projection = tf.Variable(
                tf.random.normal([measurement_dim, data_dim], stddev=0.1),
                trainable=False,
                name="measurement_projection"
            )
            generated_samples = tf.matmul(generated_measurements, projection)
        else:
            generated_samples = generated_measurements
        
        # Compute distribution matching loss
        # Using MMD (Maximum Mean Discrepancy) for distribution matching
        mmd_loss = self._compute_mmd(generated_samples, real_data)
        
        # Add regularization on generator parameters if provided
        if generator_params is not None:
            param_reg = 0.001 * tf.reduce_mean(tf.square(generator_params))
            total_loss = mmd_loss + param_reg
        else:
            total_loss = mmd_loss
        
        return total_loss
    
    def _compute_mmd(self, samples1: tf.Tensor, samples2: tf.Tensor) -> tf.Tensor:
        """
        Compute Maximum Mean Discrepancy between two sets of samples.
        
        Args:
            samples1: First set of samples
            samples2: Second set of samples
            
        Returns:
            MMD loss
        """
        # Use Gaussian kernel
        def gaussian_kernel(x, y, sigma=1.0):
            """Gaussian RBF kernel."""
            diff = x[:, None, :] - y[None, :, :]
            return tf.exp(-tf.reduce_sum(tf.square(diff), axis=-1) / (2 * sigma**2))
        
        # Compute kernel matrices
        k_xx = gaussian_kernel(samples1, samples1)
        k_yy = gaussian_kernel(samples2, samples2)
        k_xy = gaussian_kernel(samples1, samples2)
        
        # MMD = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
        mmd = (tf.reduce_mean(k_xx) + tf.reduce_mean(k_yy) - 
               2 * tf.reduce_mean(k_xy))
        
        return mmd


def create_quantum_loss(loss_type: str = 'measurement_based',
                       **kwargs) -> Any:
    """
    Factory function to create quantum loss functions.
    
    Args:
        loss_type: Type of loss function
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function instance
    """
    if loss_type == 'measurement_based':
        return QuantumMeasurementLoss(**kwargs)
    elif loss_type == 'gan':
        return QuantumGANLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
