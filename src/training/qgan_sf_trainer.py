"""
Quantum GAN SF Trainer - Complete Training Framework
====================================================

This is a clean, from-scratch implementation following the exact patterns from
Strawberry Fields tutorials for training CV Quantum GANs.

Training strategy: Wasserstein GAN with gradient penalty (WGAN-GP)
- More stable than vanilla GAN
- No need for sigmoid/BCE loss
- Better gradient behavior

Author: Fresh implementation for thesis comparison (CV vs DV vs Classical)
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, List, Dict, Any, Optional, Callable
import matplotlib.pyplot as plt


# =============================================================================
# QUANTUM GAN TRAINER - Complete Training Framework
# =============================================================================

class QGANSFTrainer:
    """
    Complete CV Quantum GAN implementation.
    
    Training strategy: Wasserstein GAN with gradient penalty (WGAN-GP)
    - More stable than vanilla GAN
    - No need for sigmoid/BCE loss
    - Better gradient behavior
    """
    
    def __init__(self,
                 generator,
                 discriminator,
                 latent_dim: int = 4,
                 learning_rate: float = 0.001,
                 lambda_gp: float = 10.0):
        """
        Initialize Quantum GAN trainer.
        
        Args:
            generator: quantum generator instance
            discriminator: quantum discriminator instance
            latent_dim: dimension of latent noise
            learning_rate: learning rate for optimizers
            lambda_gp: gradient penalty weight
        """
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp
        
        # Optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'w_distance': [],
            'g_gradient_flow': [],
            'd_gradient_flow': []
        }
        
        print(f"\nQGANSFTrainer initialized:")
        print(f"  Latent dim: {latent_dim}")
        print(f"  Generator: {generator.n_modes} modes, {generator.n_layers} layers")
        print(f"  Discriminator: {discriminator.n_modes} modes, {discriminator.n_layers} layers")
        print(f"  Gradient penalty weight: {lambda_gp}")
    
    def compute_gradient_penalty(self, real_samples: tf.Tensor, fake_samples: tf.Tensor) -> tf.Tensor:
        """
        Compute gradient penalty for WGAN-GP.
        """
        batch_size = tf.shape(real_samples)[0]
        epsilon = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        
        interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            d_interpolated = self.discriminator.discriminate(interpolated)
        
        gradients = tape.gradient(d_interpolated, interpolated)
        
        if gradients is not None:
            gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-8)
            gradient_penalty = self.lambda_gp * tf.reduce_mean(tf.square(gradient_norm - 1.0))
        else:
            gradient_penalty = tf.constant(0.0)
        
        return gradient_penalty
    
    @tf.function
    def train_discriminator_step(self, real_samples: tf.Tensor) -> Tuple[tf.Tensor, int, int]:
        """
        Single discriminator training step.
        
        Returns:
            (loss, valid_gradients_count, total_gradients_count)
        """
        batch_size = tf.shape(real_samples)[0]
        z = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as tape:
            # Generate fake samples
            fake_samples = self.generator.generate(z)
            
            # Discriminator outputs
            d_real = self.discriminator.discriminate(real_samples)
            d_fake = self.discriminator.discriminate(fake_samples)
            
            # Wasserstein loss
            d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
            
            # Gradient penalty
            gp = self.compute_gradient_penalty(real_samples, fake_samples)
            total_loss = d_loss + gp
        
        # Compute gradients
        gradients = tape.gradient(total_loss, self.discriminator.trainable_variables)
        
        # Count valid gradients
        valid_count = sum(1 for g in gradients if g is not None)
        total_count = len(gradients)
        
        # Apply gradients (filter None gradients)
        valid_grads_vars = [(g, v) for g, v in zip(gradients, self.discriminator.trainable_variables) if g is not None]
        if valid_grads_vars:
            self.d_optimizer.apply_gradients(valid_grads_vars)
        
        return d_loss, valid_count, total_count
    
    @tf.function
    def train_generator_step(self, batch_size: int) -> Tuple[tf.Tensor, int, int]:
        """
        Single generator training step.
        
        Returns:
            (loss, valid_gradients_count, total_gradients_count)
        """
        z = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as tape:
            # Generate fake samples
            fake_samples = self.generator.generate(z)
            
            # Discriminator output for fake samples
            d_fake = self.discriminator.discriminate(fake_samples)
            
            # Generator wants to maximize D(G(z)), so minimize -D(G(z))
            g_loss = -tf.reduce_mean(d_fake)
        
        # Compute gradients
        gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        
        # Count valid gradients
        valid_count = sum(1 for g in gradients if g is not None)
        total_count = len(gradients)
        
        # Apply gradients
        valid_grads_vars = [(g, v) for g, v in zip(gradients, self.generator.trainable_variables) if g is not None]
        if valid_grads_vars:
            self.g_optimizer.apply_gradients(valid_grads_vars)
        
        return g_loss, valid_count, total_count
    
    def train(self, 
              data_generator: Callable[[int], tf.Tensor],
              epochs: int = 50,
              n_critic: int = 5,
              batch_size: int = 8,
              verbose: bool = True,
              plot_interval: int = 10,
              save_interval: int = 20):
        """
        Train the Quantum GAN.
        
        Args:
            data_generator: callable that returns [batch_size, data_dim] samples
            epochs: number of training epochs
            n_critic: discriminator updates per generator update
            batch_size: training batch size
            verbose: print progress
            plot_interval: plot progress every N epochs
            save_interval: save checkpoint every N epochs
        """
        print("\n" + "="*60)
        print("STARTING QUANTUM GAN TRAINING")
        print("="*60)
        
        for epoch in range(epochs):
            epoch_g_loss = []
            epoch_d_loss = []
            epoch_g_flow = []
            epoch_d_flow = []
            
            # Train discriminator n_critic times
            for _ in range(n_critic):
                real_samples = data_generator(batch_size)
                d_loss, d_valid, d_total = self.train_discriminator_step(real_samples)
                epoch_d_loss.append(float(d_loss))
                epoch_d_flow.append(d_valid / d_total if d_total > 0 else 0)
            
            # Train generator once
            g_loss, g_valid, g_total = self.train_generator_step(batch_size)
            epoch_g_loss.append(float(g_loss))
            epoch_g_flow.append(g_valid / g_total if g_total > 0 else 0)
            
            # Record history
            avg_g_loss = np.mean(epoch_g_loss)
            avg_d_loss = np.mean(epoch_d_loss)
            avg_g_flow = np.mean(epoch_g_flow)
            avg_d_flow = np.mean(epoch_d_flow)
            
            self.history['g_loss'].append(avg_g_loss)
            self.history['d_loss'].append(avg_d_loss)
            self.history['w_distance'].append(-avg_d_loss)  # Approx Wasserstein distance
            self.history['g_gradient_flow'].append(avg_g_flow)
            self.history['d_gradient_flow'].append(avg_d_flow)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}")
                print(f"  G_gradient_flow: {avg_g_flow:.1%}, D_gradient_flow: {avg_d_flow:.1%}")
                
                # Sample some generated data for inspection
                z_sample = tf.random.normal([4, self.latent_dim])
                samples = self.generator.generate(z_sample)
                print(f"  Sample outputs: {samples.numpy()[:2]}")
            
            # Plot progress
            if plot_interval and (epoch + 1) % plot_interval == 0:
                self.plot_training_progress()
            
            # Save checkpoint
            if save_interval and (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        return self.history
    
    def plot_training_progress(self):
        """Plot training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Losses
        ax = axes[0, 0]
        ax.plot(self.history['g_loss'], label='Generator Loss')
        ax.plot(self.history['d_loss'], label='Discriminator Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('GAN Losses')
        ax.legend()
        ax.grid(True)
        
        # Wasserstein distance estimate
        ax = axes[0, 1]
        ax.plot(self.history['w_distance'], label='W-distance estimate')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('W-distance')
        ax.set_title('Wasserstein Distance')
        ax.legend()
        ax.grid(True)
        
        # Gradient flow
        ax = axes[1, 0]
        ax.plot(self.history['g_gradient_flow'], label='Generator')
        ax.plot(self.history['d_gradient_flow'], label='Discriminator')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Flow %')
        ax.set_title('Gradient Flow During Training')
        ax.legend()
        ax.grid(True)
        ax.set_ylim([0, 1.1])
        
        # Sample generation comparison
        ax = axes[1, 1]
        ax.text(0.5, 0.5, 'See separate\ngeneration plot', 
                ha='center', va='center', fontsize=14)
        ax.set_title('Generated vs Real Data')
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint_path = f"checkpoint_epoch_{epoch}"
        print(f"Saving checkpoint to {checkpoint_path}")
        # Note: In a real implementation, you'd save model weights and optimizer states
        # This is a placeholder for the checkpoint saving logic
    
    def generate_samples(self, n_samples: int = 200) -> np.ndarray:
        """Generate samples from the trained generator."""
        z = tf.random.normal([n_samples, self.latent_dim])
        samples = self.generator.generate(z)
        return samples.numpy()
    
    def evaluate_diversity(self, n_samples: int = 1000) -> float:
        """Evaluate sample diversity by computing variance."""
        samples = self.generate_samples(n_samples)
        sample_variance = tf.math.reduce_variance(samples, axis=0)
        diversity = tf.reduce_mean(sample_variance)
        return float(diversity.numpy())
