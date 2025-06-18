"""
Quantum GAN Orchestrator using modular architecture.

This module combines the quantum generator and discriminator into a complete
QGAN system with training capabilities and monitoring.
"""

import tensorflow as tf
import numpy as np
import logging
import os
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime

from .generators import PureQuantumGenerator
from .discriminators import PureQuantumDiscriminator, QuantumWassersteinDiscriminator
from losses.quantum_gan_loss import QuantumWassersteinLoss, compute_gradient_penalty
from utils.quantum_metrics import QuantumMetrics
from utils.visualization import plot_results, plot_training_history

logger = logging.getLogger(__name__)


class QuantumGAN:
    """
    Complete Quantum GAN system orchestrating generator and discriminator.
    
    This class provides:
    - Training loop with configurable loss functions
    - Metric tracking and visualization
    - Checkpointing and model saving
    - Pure quantum learning (no classical neural networks)
    """
    
    def __init__(self,
                 generator_config: Dict[str, Any],
                 discriminator_config: Dict[str, Any],
                 loss_type: str = "wasserstein",
                 learning_rate_g: float = 1e-3,
                 learning_rate_d: float = 1e-3,
                 n_critic: int = 5):
        """
        Initialize Quantum GAN.
        
        Args:
            generator_config: Configuration for quantum generator
            discriminator_config: Configuration for quantum discriminator
            loss_type: Type of loss function ('wasserstein', 'standard')
            learning_rate_g: Learning rate for generator
            learning_rate_d: Learning rate for discriminator
            n_critic: Number of discriminator updates per generator update
        """
        # Create generator
        self.generator = PureQuantumGenerator(**generator_config)
        
        # Create discriminator based on loss type
        if loss_type == "wasserstein":
            self.discriminator = QuantumWassersteinDiscriminator(**discriminator_config)
        else:
            self.discriminator = PureQuantumDiscriminator(**discriminator_config)
        
        self.loss_type = loss_type
        self.n_critic = n_critic
        
        # Initialize optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate_g)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate_d)
        
        # Initialize loss function
        if loss_type == "wasserstein":
            self.loss_fn = QuantumWassersteinLoss(
                lambda_gp=10.0,
                lambda_entropy=0.0,  # Disable entropy for pure quantum
                lambda_physics=0.0   # Disable physics constraints
            )
        
        # Initialize metrics
        self.metrics = QuantumMetrics()
        
        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'w_distance': [],
            'gradient_penalty': [],
            'mean_difference': [],
            'std_difference': []
        }
        
        logger.info(f"Quantum GAN initialized:")
        logger.info(f"  Loss type: {loss_type}")
        logger.info(f"  Generator params: {len(self.generator.trainable_variables)}")
        logger.info(f"  Discriminator params: {len(self.discriminator.trainable_variables)}")
    
    @tf.function
    def train_discriminator_step(self, real_batch: tf.Tensor, z_batch: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Single discriminator training step."""
        with tf.GradientTape() as tape:
            # Generate fake samples
            fake_batch = self.generator.generate(z_batch)
            
            # Discriminator outputs
            real_output = self.discriminator.discriminate(real_batch)
            fake_output = self.discriminator.discriminate(fake_batch)
            
            if self.loss_type == "wasserstein":
                # Wasserstein loss
                d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
                
                # Gradient penalty
                gp = compute_gradient_penalty(real_batch, fake_batch, self.discriminator)
                d_loss = d_loss + gp
                
                metrics = {
                    'd_loss': d_loss,
                    'w_distance': -d_loss + gp,
                    'gradient_penalty': gp
                }
            else:
                # Standard GAN loss
                d_loss_real = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        tf.ones_like(real_output), real_output
                    )
                )
                d_loss_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        tf.zeros_like(fake_output), fake_output
                    )
                )
                d_loss = d_loss_real + d_loss_fake
                
                metrics = {
                    'd_loss': d_loss,
                    'd_loss_real': d_loss_real,
                    'd_loss_fake': d_loss_fake
                }
        
        # Apply gradients
        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_gradients, self.discriminator.trainable_variables)
        )
        
        return metrics
    
    @tf.function
    def train_generator_step(self, z_batch: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Single generator training step."""
        with tf.GradientTape() as tape:
            # Generate fake samples
            fake_batch = self.generator.generate(z_batch)
            
            # Discriminator output for fake samples
            fake_output = self.discriminator.discriminate(fake_batch)
            
            if self.loss_type == "wasserstein":
                # Wasserstein generator loss
                g_loss = -tf.reduce_mean(fake_output)
            else:
                # Standard GAN generator loss
                g_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        tf.ones_like(fake_output), fake_output
                    )
                )
            
            metrics = {'g_loss': g_loss}
        
        # Apply gradients
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )
        
        return metrics
    
    def train(self,
              data_generator: Callable,
              epochs: int,
              batch_size: int,
              steps_per_epoch: int,
              latent_dim: int,
              validation_data: Optional[tf.Tensor] = None,
              save_interval: int = 10,
              plot_interval: int = 5,
              checkpoint_dir: Optional[str] = None):
        """
        Train the Quantum GAN.
        
        Args:
            data_generator: Function that yields batches of real data
            epochs: Number of training epochs
            batch_size: Batch size
            steps_per_epoch: Number of steps per epoch
            latent_dim: Dimension of latent space
            validation_data: Optional validation data for metrics
            save_interval: Interval for saving checkpoints
            plot_interval: Interval for plotting results
            checkpoint_dir: Directory for saving checkpoints
        """
        logger.info(f"Starting Quantum GAN training for {epochs} epochs")
        
        # Create checkpoint directory
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training loop
        for epoch in range(epochs):
            epoch_d_loss = []
            epoch_g_loss = []
            epoch_metrics = {}
            
            # Train for one epoch
            for step in range(steps_per_epoch):
                # Get real data batch
                real_batch = next(data_generator())
                
                # Train discriminator
                for _ in range(self.n_critic):
                    z_batch = tf.random.normal([batch_size, latent_dim])
                    d_metrics = self.train_discriminator_step(real_batch, z_batch)
                    epoch_d_loss.append(d_metrics['d_loss'])
                
                # Train generator
                z_batch = tf.random.normal([batch_size, latent_dim])
                g_metrics = self.train_generator_step(z_batch)
                epoch_g_loss.append(g_metrics['g_loss'])
            
            # Compute epoch metrics
            avg_d_loss = tf.reduce_mean(epoch_d_loss)
            avg_g_loss = tf.reduce_mean(epoch_g_loss)
            
            # Update history
            self.history['d_loss'].append(float(avg_d_loss))
            self.history['g_loss'].append(float(avg_g_loss))
            
            # Compute additional metrics if validation data provided
            if validation_data is not None:
                z_val = tf.random.normal([tf.shape(validation_data)[0], latent_dim])
                generated_val = self.generator.generate(z_val)
                
                # Compute distribution metrics
                val_metrics = self.metrics.classical_distribution_metrics(
                    validation_data, generated_val
                )
                
                self.history['mean_difference'].append(float(val_metrics['mean_difference']))
                self.history['std_difference'].append(float(val_metrics['std_difference']))
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")
            
            # Plot results
            if (epoch + 1) % plot_interval == 0 and validation_data is not None:
                self._plot_current_results(validation_data, epoch + 1, checkpoint_dir)
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0 and checkpoint_dir:
                self._save_checkpoint(epoch + 1, checkpoint_dir)
        
        # Final plots
        if checkpoint_dir:
            plot_training_history(self.history, 
                                os.path.join(checkpoint_dir, 'training_history.png'))
        
        logger.info("Training completed!")
    
    def generate(self, n_samples: int) -> tf.Tensor:
        """
        Generate samples using the trained generator.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        z = tf.random.normal([n_samples, self.generator.latent_dim])
        return self.generator.generate(z)
    
    def evaluate(self, real_data: tf.Tensor, n_samples: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate the GAN performance.
        
        Args:
            real_data: Real data for comparison
            n_samples: Number of samples to generate (default: same as real data)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if n_samples is None:
            n_samples = tf.shape(real_data)[0]
        
        # Generate samples
        generated_data = self.generate(n_samples)
        
        # Compute comprehensive metrics
        metrics = self.metrics.compute_comprehensive_metrics(
            real_data, 
            generated_data,
            discriminator=self.discriminator
        )
        
        # Add GAN-specific metrics
        real_scores = self.discriminator.discriminate(real_data)
        fake_scores = self.discriminator.discriminate(generated_data)
        
        metrics['real_score_mean'] = float(tf.reduce_mean(real_scores))
        metrics['fake_score_mean'] = float(tf.reduce_mean(fake_scores))
        metrics['score_difference'] = metrics['real_score_mean'] - metrics['fake_score_mean']
        
        return metrics
    
    def _plot_current_results(self, real_data: tf.Tensor, epoch: int, save_dir: Optional[str]):
        """Plot current generation results."""
        n_samples = min(500, tf.shape(real_data)[0])
        generated_data = self.generate(n_samples)
        
        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f'results_epoch_{epoch}.png')
        
        plot_results(real_data[:n_samples], generated_data, epoch, save_path)
    
    def _save_checkpoint(self, epoch: int, checkpoint_dir: str):
        """Save model checkpoint."""
        # Save generator
        gen_path = os.path.join(checkpoint_dir, f'generator_epoch_{epoch}.h5')
        gen_weights = {var.name: var.numpy() for var in self.generator.trainable_variables}
        np.savez(gen_path, **gen_weights)
        
        # Save discriminator
        disc_path = os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch}.h5')
        disc_weights = {var.name: var.numpy() for var in self.discriminator.trainable_variables}
        np.savez(disc_path, **disc_weights)
        
        logger.info(f"Checkpoint saved for epoch {epoch}")
    
    def load_checkpoint(self, epoch: int, checkpoint_dir: str):
        """Load model checkpoint."""
        # Load generator
        gen_path = os.path.join(checkpoint_dir, f'generator_epoch_{epoch}.h5.npz')
        gen_weights = np.load(gen_path)
        for var in self.generator.trainable_variables:
            if var.name in gen_weights:
                var.assign(gen_weights[var.name])
        
        # Load discriminator
        disc_path = os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch}.h5.npz')
        disc_weights = np.load(disc_path)
        for var in self.discriminator.trainable_variables:
            if var.name in disc_weights:
                var.assign(disc_weights[var.name])
        
        logger.info(f"Checkpoint loaded from epoch {epoch}")


def create_quantum_gan(gan_type: str = "wasserstein", **kwargs) -> QuantumGAN:
    """
    Factory function to create Quantum GAN with default configurations.
    
    Args:
        gan_type: Type of GAN ('wasserstein' or 'standard')
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured QuantumGAN instance
    """
    # Default generator configuration
    generator_config = {
        'latent_dim': kwargs.get('latent_dim', 6),
        'output_dim': kwargs.get('output_dim', 2),
        'n_modes': kwargs.get('gen_n_modes', 4),
        'layers': kwargs.get('gen_layers', 2),
        'cutoff_dim': kwargs.get('cutoff_dim', 6),
        'measurement_type': kwargs.get('measurement_type', 'raw')
    }
    
    # Default discriminator configuration
    discriminator_config = {
        'input_dim': kwargs.get('output_dim', 2),
        'n_modes': kwargs.get('disc_n_modes', 2),
        'layers': kwargs.get('disc_layers', 2),
        'cutoff_dim': kwargs.get('cutoff_dim', 6),
        'measurement_type': kwargs.get('measurement_type', 'raw')
    }
    
    # Create and return GAN
    return QuantumGAN(
        generator_config=generator_config,
        discriminator_config=discriminator_config,
        loss_type=gan_type,
        learning_rate_g=kwargs.get('learning_rate_g', 1e-3),
        learning_rate_d=kwargs.get('learning_rate_d', 1e-3),
        n_critic=kwargs.get('n_critic', 5)
    )
