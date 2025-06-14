# src/utils/quantum_losses.py
"""
Quantum-specific loss functions for GAN training.

This module contains loss functions that leverage quantum information
and are specifically designed for quantum generative models.
"""

import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger(__name__)

class QuantumWassersteinLoss:
    """
    Wasserstein loss enhanced with quantum regularization terms.
    
    This loss function combines the stability of Wasserstein GAN with
    quantum-specific regularization to encourage diverse quantum states
    and maintain physical validity.
    """
    
    def __init__(self, lambda_gp=10.0, lambda_entropy=0.5, lambda_physics=1.0):
        """
        Initialize quantum Wasserstein loss.
        
        Args:
            lambda_gp (float): Gradient penalty weight
            lambda_entropy (float): Quantum entropy regularization weight
            lambda_physics (float): Physics constraint weight
        """
        self.lambda_gp = lambda_gp
        self.lambda_entropy = lambda_entropy  
        self.lambda_physics = lambda_physics
        
        logger.info(f"Quantum Wasserstein Loss initialized:")
        logger.info(f"  - Gradient penalty: {lambda_gp}")
        logger.info(f"  - Entropy regularization: {lambda_entropy}")
        logger.info(f"  - Physics constraints: {lambda_physics}")
    
    def __call__(self, real_samples, fake_samples, generator, discriminator):
        """
        Compute quantum-enhanced Wasserstein loss.
        
        Args:
            real_samples: Real training data
            fake_samples: Generated samples
            generator: Quantum generator with compute_quantum_cost method
            discriminator: Quantum discriminator
            
        Returns:
            tuple: (discriminator_loss, generator_loss, metrics_dict)
        """
        batch_size = tf.shape(real_samples)[0]
        
        # 1. Wasserstein distance
        real_output = discriminator.discriminate(real_samples)
        fake_output = discriminator.discriminate(fake_samples)
        w_distance = tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)
        
        # 2. Gradient penalty for Lipschitz constraint
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            interp_output = discriminator.discriminate(interpolated)
        
        gradients = gp_tape.gradient(interp_output, interpolated)
        gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        gradient_penalty = self.lambda_gp * tf.reduce_mean(tf.square(gradient_norm - 1.0))
        
        # 3. Quantum regularization
        try:
            quantum_metrics = generator.compute_quantum_cost()
            
            # Entropy regularization (encourages diverse quantum states)
            entropy_bonus = self.lambda_entropy * quantum_metrics['entropy']
            
            # Physics constraints (ensure valid quantum states)
            trace_penalty = self.lambda_physics * tf.square(quantum_metrics['trace'] - 1.0)
            norm_penalty = self.lambda_physics * tf.square(quantum_metrics['norm'] - 1.0)
            
        except Exception as e:
            logger.warning(f"Quantum metrics computation failed: {e}")
            entropy_bonus = tf.constant(0.0)
            trace_penalty = tf.constant(0.0)
            norm_penalty = tf.constant(0.0)
        
        # Final losses
        d_loss = -w_distance + gradient_penalty + trace_penalty + norm_penalty
        g_loss = -tf.reduce_mean(fake_output) - entropy_bonus + trace_penalty + norm_penalty
        
        # Metrics for monitoring
        metrics = {
            'w_distance': w_distance,
            'gradient_penalty': gradient_penalty,
            'entropy_bonus': entropy_bonus,
            'trace_penalty': trace_penalty,
            'norm_penalty': norm_penalty,
            'total_d_loss': d_loss,
            'total_g_loss': g_loss
        }
        
        return d_loss, g_loss, metrics

class QuantumMMDLoss:
    """Alternative: Maximum Mean Discrepancy loss with quantum regularization."""
    
    def __init__(self, sigma=1.0, lambda_entropy=0.1):
        self.sigma = sigma
        self.lambda_entropy = lambda_entropy
    
    def __call__(self, real_samples, fake_samples, generator, discriminator=None):
        """Compute MMD loss with quantum entropy regularization."""
        
        # Gaussian kernel for MMD
        def gaussian_kernel(x, y):
            return tf.exp(-tf.reduce_sum(tf.square(x - y), axis=1) / (2 * self.sigma**2))
        
        # MMD computation
        real_real = tf.reduce_mean(gaussian_kernel(real_samples[:, None], real_samples[None, :]))
        fake_fake = tf.reduce_mean(gaussian_kernel(fake_samples[:, None], fake_samples[None, :]))
        real_fake = tf.reduce_mean(gaussian_kernel(real_samples[:, None], fake_samples[None, :]))
        
        mmd_loss = real_real + fake_fake - 2 * real_fake
        
        # Quantum entropy regularization
        try:
            quantum_metrics = generator.compute_quantum_cost()
            entropy_reg = -self.lambda_entropy * quantum_metrics['entropy']
        except:
            entropy_reg = tf.constant(0.0)
        
        total_loss = mmd_loss + entropy_reg
        
        metrics = {
            'mmd_loss': mmd_loss,
            'entropy_reg': entropy_reg,
            'total_loss': total_loss
        }
        
        return total_loss, total_loss, metrics  # Same loss for both G and D

# Utility functions
def compute_gradient_penalty(real_samples, fake_samples, discriminator, lambda_gp=10.0):
    """Standalone gradient penalty computation."""
    batch_size = tf.shape(real_samples)[0]
    alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        interp_output = discriminator.discriminate(interpolated)
    
    gradients = tape.gradient(interp_output, interpolated)
    gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
    gradient_penalty = lambda_gp * tf.reduce_mean(tf.square(gradient_norm - 1.0))
    
    return gradient_penalty

def compute_quantum_regularization(generator, lambda_entropy=0.5, lambda_physics=1.0):
    """Standalone quantum regularization computation."""
    try:
        quantum_metrics = generator.compute_quantum_cost()
        
        entropy_bonus = lambda_entropy * quantum_metrics['entropy']
        trace_penalty = lambda_physics * tf.square(quantum_metrics['trace'] - 1.0)
        norm_penalty = lambda_physics * tf.square(quantum_metrics['norm'] - 1.0)
        
        return entropy_bonus, trace_penalty, norm_penalty, quantum_metrics
    
    except Exception as e:
        logger.warning(f"Quantum regularization failed: {e}")
        return (tf.constant(0.0), tf.constant(0.0), tf.constant(0.0), 
                {'entropy': 0.0, 'trace': 1.0, 'norm': 1.0})