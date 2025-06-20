"""
Complete Quantum GAN Training Framework for Pure SF Implementation

This module implements the full training framework from your training guide,
adapted to work with Pure SF components with guaranteed gradient flow.
"""

import numpy as np
import tensorflow as tf
import logging
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import os
import json

# Import Pure SF components
from ..models.generators.pure_sf_generator import PureSFGenerator
from ..models.discriminators.pure_sf_discriminator import PureSFDiscriminator

# Import training infrastructure
from ..losses.quantum_gan_loss import QuantumWassersteinLoss
from ..utils.gradient_manager import QuantumGradientManager
from .data_generators import BimodalDataGenerator

logger = logging.getLogger(__name__)


class QuantumGANTrainer:
    """
    Complete training framework for Pure SF Quantum GANs.
    
    Implements the full training strategy from your guide with:
    - QuantumWassersteinLoss with gradient penalty
    - QuantumGradientManager for SF NaN handling
    - Individual sample processing for diversity preservation
    - Comprehensive monitoring and visualization
    """
    
    def __init__(self, 
                 generator: PureSFGenerator,
                 discriminator: PureSFDiscriminator,
                 loss_type: str = 'wasserstein',
                 n_critic: int = 5,
                 learning_rate_g: float = 1e-3,
                 learning_rate_d: float = 1e-3,
                 entropy_weight: float = 0.01,
                 verbose: bool = True):
        """
        Initialize quantum GAN trainer.
        
        Args:
            generator: Pure SF quantum generator
            discriminator: Pure SF quantum discriminator  
            loss_type: Type of loss function ('wasserstein')
            n_critic: Train discriminator n_critic times per generator step
            learning_rate_g: Generator learning rate
            learning_rate_d: Discriminator learning rate
            entropy_weight: Weight for entropy regularization (diversity)
            verbose: Enable detailed logging
        """
        self.generator = generator
        self.discriminator = discriminator
        self.loss_type = loss_type
        self.n_critic = n_critic
        self.entropy_weight = entropy_weight
        self.verbose = verbose
        
        # Initialize loss function
        if loss_type == 'wasserstein':
            self.loss_fn = QuantumWassersteinLoss(
                lambda_gp=10.0,        # Gradient penalty weight
                lambda_entropy=0.5,    # Quantum entropy regularization
                lambda_physics=1.0     # Physics constraints
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Initialize gradient managers (critical for SF)
        self.g_gradient_manager = QuantumGradientManager(
            verbose=verbose,
            max_gradient_norm=1.0,      # Conservative for quantum
            parameter_bounds=(-5.0, 5.0),  # Reasonable quantum bounds
            clip_gradients=True
        )
        
        self.d_gradient_manager = QuantumGradientManager(
            verbose=verbose,
            max_gradient_norm=1.0,      # Conservative for quantum
            parameter_bounds=(-5.0, 5.0),  # Reasonable quantum bounds
            clip_gradients=True
        )
        
        # Initialize optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate_g,
            beta_1=0.5  # Recommended for GANs
        )
        self.d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate_d,
            beta_1=0.5  # Recommended for GANs
        )
        
        # Training metrics tracking
        self.metrics_history = {
            'g_loss': [], 'd_loss': [], 'w_distance': [],
            'gradient_penalty': [], 'g_gradients': [], 'd_gradients': [],
            'entropy_bonus': [], 'physics_penalty': []
        }
        
        # Training state
        self.step_counter = 0
        self.epoch_counter = 0
        
        if verbose:
            logger.info("QuantumGANTrainer initialized")
            logger.info(f"  Generator parameters: {len(generator.trainable_variables)}")
            logger.info(f"  Discriminator parameters: {len(discriminator.trainable_variables)}")
            logger.info(f"  Loss type: {loss_type}")
            logger.info(f"  n_critic: {n_critic}")
            logger.info(f"  entropy_weight: {entropy_weight}")
    
    def train_discriminator_step(self, real_batch: tf.Tensor, z_batch: tf.Tensor) -> Dict[str, Any]:
        """
        Single discriminator training step with gradient penalty.
        
        Args:
            real_batch: Real data samples
            z_batch: Latent input for generator
            
        Returns:
            Dictionary of discriminator metrics
        """
        with self.d_gradient_manager.managed_computation(self.discriminator.trainable_variables) as tape:
            # Generate fake samples
            fake_batch = self.generator.generate(z_batch)
            
            # Discriminator outputs
            real_output = self.discriminator.discriminate(real_batch)
            fake_output = self.discriminator.discriminate(fake_batch)
            
            # Wasserstein distance
            w_distance = tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)
            
            # Gradient penalty for Lipschitz constraint
            batch_size = tf.shape(real_batch)[0]
            alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
            interpolated = alpha * real_batch + (1 - alpha) * fake_batch
            
            # Compute gradient penalty
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                interp_output = self.discriminator.discriminate(interpolated)
            
            gradients = gp_tape.gradient(interp_output, interpolated)
            
            if gradients is None:
                gradient_penalty = tf.constant(0.0)
                if self.verbose:
                    logger.warning("Gradient penalty computation failed")
            else:
                gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
                gradient_penalty = 10.0 * tf.reduce_mean(tf.square(gradient_norm - 1.0))
            
            # Total discriminator loss (minimize negative Wasserstein distance)
            d_loss = -w_distance + gradient_penalty
        
        # Apply gradients using gradient manager
        d_gradients = self.d_gradient_manager.safe_gradient(
            tape, d_loss, self.discriminator.trainable_variables
        )
        
        success = self.d_gradient_manager.apply_gradients_safely(
            self.d_optimizer, d_gradients, self.discriminator.trainable_variables
        )
        
        # Collect metrics
        metrics = {
            'total_loss': d_loss,
            'w_distance': w_distance,
            'gradient_penalty': gradient_penalty,
            'gradient_success': success,
            'gradient_count': sum(1 for g in d_gradients if g is not None)
        }
        
        return metrics
    
    def train_generator_step(self, z_batch: tf.Tensor) -> Dict[str, Any]:
        """
        Single generator training step with quantum regularization.
        
        Args:
            z_batch: Latent input for generator
            
        Returns:
            Dictionary of generator metrics
        """
        with self.g_gradient_manager.managed_computation(self.generator.trainable_variables) as tape:
            # Generate fake samples
            fake_batch = self.generator.generate(z_batch)
            
            # Discriminator output for fake samples
            fake_output = self.discriminator.discriminate(fake_batch)
            
            # Wasserstein generator loss (maximize discriminator output)
            base_loss = -tf.reduce_mean(fake_output)
            
            # Quantum regularization terms
            # Simple physics regularization (keep parameters bounded)
            physics_penalty = 0.0
            for var in self.generator.trainable_variables:
                physics_penalty += tf.reduce_mean(tf.nn.relu(tf.abs(var) - 5.0))
            physics_penalty *= 0.1
            
            # Entropy regularization (encourage diverse quantum states)
            # Use variance of generated samples as diversity proxy
            sample_variance = tf.math.reduce_variance(fake_batch)
            entropy_bonus = -self.entropy_weight * sample_variance  # Negative because we want to maximize variance
            
            # Total generator loss
            g_loss = base_loss + physics_penalty - entropy_bonus
        
        # Apply gradients using gradient manager
        g_gradients = self.g_gradient_manager.safe_gradient(
            tape, g_loss, self.generator.trainable_variables
        )
        
        success = self.g_gradient_manager.apply_gradients_safely(
            self.g_optimizer, g_gradients, self.generator.trainable_variables
        )
        
        # Collect metrics
        metrics = {
            'total_loss': g_loss,
            'base_loss': base_loss,
            'physics_penalty': physics_penalty,
            'entropy_bonus': entropy_bonus,
            'gradient_success': success,
            'gradient_count': sum(1 for g in g_gradients if g is not None)
        }
        
        return metrics
    
    def train_epoch(self, 
                   data_generator: Callable[[], tf.Tensor],
                   steps_per_epoch: int,
                   latent_dim: int,
                   epoch_num: int) -> Dict[str, float]:
        """
        Train for one complete epoch.
        
        Args:
            data_generator: Function that returns batches of real data
            steps_per_epoch: Number of training steps per epoch
            latent_dim: Dimension of latent space
            epoch_num: Current epoch number
            
        Returns:
            Dictionary of epoch metrics
        """
        epoch_metrics = {
            'g_loss': [], 'd_loss': [], 'w_distance': [],
            'gradient_penalty': [], 'g_gradients': [], 'd_gradients': [],
            'entropy_bonus': [], 'physics_penalty': []
        }
        
        for step in range(steps_per_epoch):
            # Get real data batch
            real_batch = data_generator()
            batch_size = tf.shape(real_batch)[0]
            
            # Train discriminator (n_critic times)
            for _ in range(self.n_critic):
                z_batch = tf.random.normal([batch_size, latent_dim])
                d_metrics = self.train_discriminator_step(real_batch, z_batch)
                
                # Record discriminator metrics
                epoch_metrics['d_loss'].append(float(d_metrics['total_loss']))
                epoch_metrics['d_gradients'].append(d_metrics['gradient_count'])
                epoch_metrics['w_distance'].append(float(d_metrics['w_distance']))
                epoch_metrics['gradient_penalty'].append(float(d_metrics['gradient_penalty']))
            
            # Train generator (once)
            z_batch = tf.random.normal([batch_size, latent_dim])
            g_metrics = self.train_generator_step(z_batch)
            
            # Record generator metrics
            epoch_metrics['g_loss'].append(float(g_metrics['total_loss']))
            epoch_metrics['g_gradients'].append(g_metrics['gradient_count'])
            epoch_metrics['entropy_bonus'].append(float(g_metrics['entropy_bonus']))
            epoch_metrics['physics_penalty'].append(float(g_metrics['physics_penalty']))
            
            # Log progress periodically
            if step % (steps_per_epoch // 4) == 0 and self.verbose:
                logger.info(f"Epoch {epoch_num}, Step {step}/{steps_per_epoch}: "
                           f"G_loss={g_metrics['total_loss']:.4f}, "
                           f"D_loss={d_metrics['total_loss']:.4f}, "
                           f"G_grads={g_metrics['gradient_count']}, "
                           f"D_grads={d_metrics['gradient_count']}")
            
            self.step_counter += 1
        
        # Compute epoch averages
        epoch_summary = {
            'g_loss': np.mean(epoch_metrics['g_loss']),
            'd_loss': np.mean(epoch_metrics['d_loss']),
            'w_distance': np.mean(epoch_metrics['w_distance']),
            'gradient_penalty': np.mean(epoch_metrics['gradient_penalty']),
            'g_gradients': np.mean(epoch_metrics['g_gradients']),
            'd_gradients': np.mean(epoch_metrics['d_gradients']),
            'entropy_bonus': np.mean(epoch_metrics['entropy_bonus']),
            'physics_penalty': np.mean(epoch_metrics['physics_penalty'])
        }
        
        return epoch_summary
    
    def train(self,
             data_generator: Callable[[], tf.Tensor],
             epochs: int,
             steps_per_epoch: int,
             latent_dim: int,
             validation_data: Optional[tf.Tensor] = None,
             save_interval: int = 10,
             plot_interval: int = 5) -> None:
        """
        Complete training procedure.
        
        Args:
            data_generator: Function that returns batches of real data
            epochs: Number of training epochs
            steps_per_epoch: Number of training steps per epoch
            latent_dim: Dimension of latent space
            validation_data: Optional validation data for evaluation
            save_interval: Save model every N epochs
            plot_interval: Generate plots every N epochs
        """
        if self.verbose:
            logger.info(f"Starting Pure SF QGAN training: {epochs} epochs, {steps_per_epoch} steps/epoch")
            logger.info(f"Generator parameters: {len(self.generator.trainable_variables)}")
            logger.info(f"Discriminator parameters: {len(self.discriminator.trainable_variables)}")
        
        # Verify initial gradient flow
        self._verify_initial_gradient_flow(data_generator, latent_dim)
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Train for one epoch
            epoch_metrics = self.train_epoch(
                data_generator, steps_per_epoch, latent_dim, epoch + 1
            )
            
            # Record metrics
            for key, value in epoch_metrics.items():
                if key in self.metrics_history:
                    self.metrics_history[key].append(value)
            
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch summary
            if self.verbose:
                logger.info(f"Epoch {epoch + 1}/{epochs} completed in {epoch_time:.2f}s:")
                logger.info(f"  G_loss: {epoch_metrics['g_loss']:.4f}")
                logger.info(f"  D_loss: {epoch_metrics['d_loss']:.4f}")
                logger.info(f"  W_distance: {epoch_metrics['w_distance']:.4f}")
                logger.info(f"  G_gradients: {epoch_metrics['g_gradients']:.1f}/{len(self.generator.trainable_variables)}")
                logger.info(f"  D_gradients: {epoch_metrics['d_gradients']:.1f}/{len(self.discriminator.trainable_variables)}")
            
            # Generate samples and plots periodically
            if (epoch + 1) % plot_interval == 0:
                self._generate_evaluation_plots(latent_dim, epoch + 1, validation_data)
            
            # Save model periodically
            if (epoch + 1) % save_interval == 0:
                self._save_model(epoch + 1)
            
            self.epoch_counter += 1
        
        # Final evaluation
        if self.verbose:
            logger.info("Training completed!")
            self._generate_final_evaluation(latent_dim, validation_data)
            self._print_gradient_manager_summary()
    
    def _verify_initial_gradient_flow(self, data_generator: Callable[[], tf.Tensor], latent_dim: int):
        """Verify gradient flow before training starts."""
        if self.verbose:
            logger.info("Verifying initial gradient flow...")
        
        real_batch = data_generator()
        batch_size = tf.shape(real_batch)[0]
        z_batch = tf.random.normal([batch_size, latent_dim])
        
        # Test generator gradients
        with tf.GradientTape() as tape:
            fake_batch = self.generator.generate(z_batch)
            test_loss = tf.reduce_mean(fake_batch)
        
        g_gradients = tape.gradient(test_loss, self.generator.trainable_variables)
        g_valid = sum(1 for g in g_gradients if g is not None)
        
        # Test discriminator gradients
        with tf.GradientTape() as tape:
            real_output = self.discriminator.discriminate(real_batch)
            test_loss = tf.reduce_mean(real_output)
        
        d_gradients = tape.gradient(test_loss, self.discriminator.trainable_variables)
        d_valid = sum(1 for g in d_gradients if g is not None)
        
        if self.verbose:
            logger.info(f"Initial gradient flow:")
            logger.info(f"  Generator: {g_valid}/{len(self.generator.trainable_variables)} parameters")
            logger.info(f"  Discriminator: {d_valid}/{len(self.discriminator.trainable_variables)} parameters")
        
        if g_valid == 0 or d_valid == 0:
            logger.error("CRITICAL: No gradient flow detected! Check model implementation.")
            raise RuntimeError("Gradient flow verification failed")
    
    def _generate_evaluation_plots(self, latent_dim: int, epoch: int, validation_data: Optional[tf.Tensor] = None):
        """Generate evaluation plots during training."""
        # Generate samples
        z_test = tf.random.normal([100, latent_dim])
        generated_samples = self.generator.generate(z_test)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Training curves
        axes[0, 0].plot(self.metrics_history['g_loss'], label='Generator')
        axes[0, 0].plot(self.metrics_history['d_loss'], label='Discriminator')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Gradient flow
        axes[0, 1].plot(self.metrics_history['g_gradients'], label='Generator')
        axes[0, 1].plot(self.metrics_history['d_gradients'], label='Discriminator')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Gradient Count')
        axes[0, 1].set_title('Gradient Flow')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Sample distribution
        if validation_data is not None:
            axes[1, 0].scatter(validation_data[:200, 0], validation_data[:200, 1], 
                              alpha=0.5, label='Real', color='blue', s=20)
        axes[1, 0].scatter(generated_samples[:100, 0], generated_samples[:100, 1], 
                          alpha=0.5, label='Generated', color='red', s=20)
        axes[1, 0].set_xlabel('Feature 1')
        axes[1, 0].set_ylabel('Feature 2')
        axes[1, 0].set_title('Sample Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Wasserstein distance
        axes[1, 1].plot(self.metrics_history['w_distance'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Wasserstein Distance')
        axes[1, 1].set_title('Wasserstein Distance')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'pure_sf_qgan_training_epoch_{epoch}_{timestamp}.png'
        plt.savefig(filename, dpi=150)
        plt.close()
        
        if self.verbose:
            logger.info(f"Evaluation plots saved to {filename}")
    
    def _generate_final_evaluation(self, latent_dim: int, validation_data: Optional[tf.Tensor] = None):
        """Generate final evaluation after training."""
        if self.verbose:
            logger.info("Generating final evaluation...")
        
        # Generate large sample set
        z_test = tf.random.normal([1000, latent_dim])
        generated_samples = self.generator.generate(z_test)
        
        # Sample statistics
        gen_mean = tf.reduce_mean(generated_samples, axis=0)
        gen_std = tf.math.reduce_std(generated_samples, axis=0)
        
        if self.verbose:
            logger.info(f"Generated sample statistics:")
            logger.info(f"  Mean: {gen_mean.numpy()}")
            logger.info(f"  Std: {gen_std.numpy()}")
        
        # Save final samples
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        np.save(f'final_pure_sf_generated_samples_{timestamp}.npy', generated_samples.numpy())
    
    def _save_model(self, epoch: int):
        """Save model checkpoints."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = f'checkpoints/pure_sf_qgan_epoch_{epoch}_{timestamp}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model weights (only trainable variables for Pure SF)
        g_weights = {f'var_{i}': var.numpy().tolist() for i, var in enumerate(self.generator.trainable_variables)}
        d_weights = {f'var_{i}': var.numpy().tolist() for i, var in enumerate(self.discriminator.trainable_variables)}
        
        with open(f'{checkpoint_dir}/generator_weights.json', 'w') as f:
            json.dump(g_weights, f)
        
        with open(f'{checkpoint_dir}/discriminator_weights.json', 'w') as f:
            json.dump(d_weights, f)
        
        # Save training history
        with open(f'{checkpoint_dir}/training_history.json', 'w') as f:
            # Convert numpy types to native Python types
            history_copy = {}
            for key, values in self.metrics_history.items():
                history_copy[key] = [float(v) for v in values]
            json.dump(history_copy, f)
        
        if self.verbose:
            logger.info(f"Model saved to {checkpoint_dir}")
    
    def _print_gradient_manager_summary(self):
        """Print final gradient manager statistics."""
        if self.verbose:
            logger.info("Gradient Manager Summary:")
            
            g_summary = self.g_gradient_manager.get_summary()
            d_summary = self.d_gradient_manager.get_summary()
            
            logger.info(f"Generator:")
            logger.info(f"  Total steps: {g_summary['total_steps']}")
            logger.info(f"  NaN detections: {g_summary['nan_detections']}")
            logger.info(f"  Gradient clips: {g_summary['gradient_clips']}")
            
            logger.info(f"Discriminator:")
            logger.info(f"  Total steps: {d_summary['total_steps']}")
            logger.info(f"  NaN detections: {d_summary['nan_detections']}")
            logger.info(f"  Gradient clips: {d_summary['gradient_clips']}")
