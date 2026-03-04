"""
Quantum GAN Trainer
===================

Training loop for CV Quantum GAN with:
- Separate generator and discriminator updates
- Monitoring and visualization callbacks
- Wasserstein loss with gradient penalty (WGAN-GP)
- Early stopping support
- Checkpoint saving

Usage:
    from training.trainer import QGANTrainer
    
    trainer = QGANTrainer(generator, discriminator, config)
    history = trainer.train(real_data_generator, epochs=100)
"""

import numpy as np
import tensorflow as tf
from typing import Callable, Optional, Dict, Any, Tuple
import os
from dataclasses import dataclass

# Import monitoring (will be in same package)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.monitoring import TrainingMonitor, QuantumStateMonitor
from utils.visualization import (
    plot_distribution_comparison,
    plot_wigner_3d,
    plot_wigner_2d
)


@dataclass
class TrainerConfig:
    """Configuration for QGAN training."""
    # Target distribution
    target_mean: float = 0.0
    target_std: float = 1.0
    
    # Training
    batch_size: int = 32
    g_lr: float = 0.001
    d_lr: float = 0.001
    n_critic: int = 1  # D updates per G update
    
    # WGAN-GP
    use_wgan_gp: bool = True
    gp_weight: float = 10.0
    
    # Monitoring
    log_dir: Optional[str] = None
    print_every: int = 10
    plot_every: int = 50
    save_every: int = 100
    
    # Early stopping
    patience: int = 100
    min_epochs: int = 50
    
    # Quantum state monitoring
    monitor_quantum_state: bool = True
    wigner_epochs: list = None  # Epochs to capture Wigner function
    
    def __post_init__(self):
        if self.wigner_epochs is None:
            self.wigner_epochs = [0, 10, 25, 50, 100, 200, 500, 1000]


class QGANTrainer:
    """
    Trainer for CV Quantum GAN.
    
    Handles:
    - Training loop with WGAN-GP loss
    - Generator and discriminator optimization
    - Monitoring and visualization
    - Checkpointing
    
    Args:
        generator: Quantum generator model
        discriminator: Discriminator model (quantum or classical)
        config: TrainerConfig object
    """
    
    def __init__(
        self,
        generator,
        discriminator,
        config: TrainerConfig = None
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.config = config or TrainerConfig()
        
        # Optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.g_lr,
            beta_1=0.0,  # WGAN recommendation
            beta_2=0.9
        )
        self.d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.d_lr,
            beta_1=0.0,
            beta_2=0.9
        )
        
        # Monitoring
        self.monitor = TrainingMonitor(
            target_mean=self.config.target_mean,
            target_std=self.config.target_std,
            log_dir=self.config.log_dir,
            verbose=True
        )
        
        # Quantum state monitoring (optional)
        self.state_monitor = None
        if self.config.monitor_quantum_state:
            # Fixed latent for consistent visualization
            fixed_z = tf.random.normal([1, generator.latent_dim], seed=42)
            self.state_monitor = QuantumStateMonitor(
                generator=generator,
                fixed_latent=fixed_z,
                log_dir=self.config.log_dir,
                capture_epochs=self.config.wigner_epochs
            )
        
        # Create log directory
        if self.config.log_dir and not os.path.exists(self.config.log_dir):
            os.makedirs(self.config.log_dir)
    
    def gradient_penalty(
        self,
        real_samples: tf.Tensor,
        fake_samples: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute gradient penalty for WGAN-GP.
        
        Interpolates between real and fake samples and penalizes
        gradients that deviate from unit norm.
        """
        batch_size = tf.shape(real_samples)[0]
        
        # Random interpolation weights
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        
        # Interpolated samples
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            d_interpolated = self.discriminator.discriminate(interpolated)
        
        # Compute gradients
        gradients = tape.gradient(d_interpolated, interpolated)
        
        # Compute gradient norm
        gradients_norm = tf.sqrt(
            tf.reduce_sum(tf.square(gradients), axis=1) + 1e-8
        )
        
        # Gradient penalty
        gp = tf.reduce_mean(tf.square(gradients_norm - 1.0))
        
        return gp
    
    def train_discriminator_step(
        self,
        real_samples: tf.Tensor,
        z: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        One discriminator training step.
        
        Returns:
            (d_loss, d_gradients)
        """
        with tf.GradientTape() as tape:
            # Generate fake samples
            fake_samples = self.generator.generate(z)
            
            # Discriminator scores
            real_scores = self.discriminator.discriminate(real_samples)
            fake_scores = self.discriminator.discriminate(fake_samples)
            
            # WGAN loss: maximize E[D(real)] - E[D(fake)]
            # Equivalent to minimize E[D(fake)] - E[D(real)]
            d_loss = tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)
            
            # Gradient penalty
            if self.config.use_wgan_gp:
                gp = self.gradient_penalty(real_samples, fake_samples)
                d_loss = d_loss + self.config.gp_weight * gp
        
        # Compute gradients
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        
        # Apply gradients
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables)
        )
        
        return d_loss, d_grads
    
    def train_generator_step(
        self,
        z: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        One generator training step.
        
        Returns:
            (g_loss, g_gradients, fake_samples)
        """
        with tf.GradientTape() as tape:
            # Generate fake samples
            fake_samples = self.generator.generate(z)
            
            # Discriminator scores
            fake_scores = self.discriminator.discriminate(fake_samples)
            
            # Generator wants to maximize D(fake) = minimize -D(fake)
            g_loss = -tf.reduce_mean(fake_scores)
        
        # Compute gradients
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        
        # Apply gradients
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_variables)
        )
        
        return g_loss, g_grads, fake_samples
    
    def train(
        self,
        data_generator: Callable[[int], np.ndarray],
        epochs: int = 100,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the QGAN.
        
        Args:
            data_generator: Function that returns batch_size real samples
            epochs: Number of training epochs
            verbose: Print progress
            
        Returns:
            Training history dictionary
        """
        print("=" * 60)
        print("Starting QGAN Training")
        print("=" * 60)
        print(f"Generator: {self.generator.get_config()}")
        print(f"Target: N({self.config.target_mean}, {self.config.target_std}²)")
        print(f"Epochs: {epochs}, Batch size: {self.config.batch_size}")
        print("=" * 60)
        
        # Capture initial quantum state
        if self.state_monitor:
            self.state_monitor.capture_state(0)
        
        for epoch in range(epochs):
            # =====================================================
            # DISCRIMINATOR UPDATE(S)
            # =====================================================
            for _ in range(self.config.n_critic):
                # Get real samples
                real_batch = data_generator(self.config.batch_size)
                real_batch = tf.cast(real_batch, tf.float32)
                
                # Generate latent vectors
                z = tf.random.normal([self.config.batch_size, self.generator.latent_dim])
                
                # Train discriminator
                d_loss, d_grads = self.train_discriminator_step(real_batch, z)
            
            # =====================================================
            # GENERATOR UPDATE
            # =====================================================
            z = tf.random.normal([self.config.batch_size, self.generator.latent_dim])
            g_loss, g_grads, fake_samples = self.train_generator_step(z)
            
            # =====================================================
            # MONITORING
            # =====================================================
            # Update metrics
            self.monitor.update(
                g_loss=g_loss,
                d_loss=d_loss,
                g_grads=g_grads,
                d_grads=d_grads,
                gen_samples=fake_samples.numpy(),
                real_samples=real_batch.numpy()
            )
            
            # Print status
            if verbose and (epoch + 1) % self.config.print_every == 0:
                self.monitor.print_status(epoch + 1, epochs)
            
            # Plot distributions
            if (epoch + 1) % self.config.plot_every == 0:
                self._plot_progress(epoch + 1, fake_samples.numpy(), real_batch.numpy())
            
            # Capture quantum state
            if self.state_monitor:
                self.state_monitor.capture_state(epoch + 1)
            
            # Save checkpoint
            if self.config.log_dir and (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(epoch + 1)
            
            # Early stopping
            if epoch >= self.config.min_epochs and self.monitor.should_stop_early(self.config.patience):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        # =====================================================
        # FINAL SUMMARY
        # =====================================================
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        
        summary = self.monitor.get_summary()
        print(f"Final Generator Loss: {summary['final_g_loss']:.4f}")
        print(f"Final Discriminator Loss: {summary['final_d_loss']:.4f}")
        print(f"Final Generated Mean: {summary['final_mean']:.4f} (target: {self.config.target_mean})")
        print(f"Final Generated Std: {summary['final_std']:.4f} (target: {self.config.target_std})")
        print(f"Final Wasserstein Distance: {summary['final_wasserstein']:.4f}")
        print(f"Best Wasserstein: {summary['best_wasserstein']:.4f} at epoch {summary['best_epoch']}")
        print(f"Converged: {summary['converged']}")
        
        # Save final metrics
        if self.config.log_dir:
            self.monitor.save_metrics()
        
        # Plot final dashboard
        self.monitor.plot_dashboard(epochs)
        
        # Plot quantum state evolution
        if self.state_monitor:
            self.state_monitor.plot_evolution()
        
        return summary
    
    def _plot_progress(self, epoch: int, gen_samples: np.ndarray, real_samples: np.ndarray):
        """Plot training progress."""
        self.monitor.plot_distributions(gen_samples, real_samples, epoch)
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        if not self.config.log_dir:
            return
        
        # Save weights
        gen_path = os.path.join(self.config.log_dir, f'generator_epoch_{epoch}.npy')
        np.save(gen_path, {
            'weights': self.generator.weights.numpy(),
            'decoder': self.generator.decoder.numpy()
        })
        
        # Save metrics
        self.monitor.save_metrics()
        
        print(f"Checkpoint saved at epoch {epoch}")


# =============================================================================
# DATA GENERATORS
# =============================================================================

def gaussian_data_generator(mean: float = 0.0, std: float = 1.0) -> Callable[[int], np.ndarray]:
    """
    Create a function that generates Gaussian samples.
    
    Args:
        mean: Distribution mean
        std: Distribution standard deviation
        
    Returns:
        Function that takes batch_size and returns samples
    """
    def generator(batch_size: int) -> np.ndarray:
        samples = np.random.normal(mean, std, size=(batch_size, 1))
        return samples.astype(np.float32)
    
    return generator


def bimodal_data_generator(
    mean1: float = -2.0,
    mean2: float = 2.0,
    std: float = 0.5
) -> Callable[[int], np.ndarray]:
    """
    Create a function that generates bimodal Gaussian samples.
    
    Args:
        mean1: First mode mean
        mean2: Second mode mean
        std: Standard deviation for both modes
        
    Returns:
        Function that takes batch_size and returns samples
    """
    def generator(batch_size: int) -> np.ndarray:
        # Randomly choose mode
        mode = np.random.randint(0, 2, size=batch_size)
        samples = np.where(
            mode == 0,
            np.random.normal(mean1, std, size=batch_size),
            np.random.normal(mean2, std, size=batch_size)
        )
        return samples.reshape(-1, 1).astype(np.float32)
    
    return generator


# =============================================================================
# QUICK START FUNCTION
# =============================================================================

def train_simple_qgan(
    n_modes: int = 1,
    n_layers: int = 1,
    target_mean: float = 2.0,
    target_std: float = 0.5,
    epochs: int = 100,
    log_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick start function to train a simple QGAN.
    
    Uses:
    - Quantum generator with displacement encoding
    - Classical discriminator (MLP)
    - WGAN-GP loss
    - Gaussian target distribution
    
    Args:
        n_modes: Number of quantum modes
        n_layers: Number of QNN layers
        target_mean: Target Gaussian mean
        target_std: Target Gaussian std
        epochs: Training epochs
        log_dir: Optional log directory
        
    Returns:
        Training summary
    """
    from models.generators.quantum_sf_generator import QuantumSFGenerator
    from models.discriminators.classical_discriminator import ClassicalDiscriminator
    
    # Create models
    generator = QuantumSFGenerator(
        n_modes=n_modes,
        n_layers=n_layers,
        cutoff_dim=6,
        output_dim=1,
        encoding_type='displacement_full'
    )
    
    discriminator = ClassicalDiscriminator(
        input_dim=1,
        hidden_dims=[32, 32],
        output_dim=1
    )
    
    # Config
    config = TrainerConfig(
        target_mean=target_mean,
        target_std=target_std,
        batch_size=32,
        g_lr=0.001,
        d_lr=0.001,
        log_dir=log_dir,
        print_every=10,
        plot_every=50
    )
    
    # Create trainer
    trainer = QGANTrainer(generator, discriminator, config)
    
    # Data generator
    data_gen = gaussian_data_generator(mean=target_mean, std=target_std)
    
    # Train
    return trainer.train(data_gen, epochs=epochs)


if __name__ == "__main__":
    # Test the trainer components
    print("Testing QGANTrainer components...")
    
    # Test data generators
    print("\n--- Testing Data Generators ---")
    gauss_gen = gaussian_data_generator(mean=2.0, std=0.5)
    samples = gauss_gen(100)
    print(f"Gaussian samples: mean={np.mean(samples):.3f}, std={np.std(samples):.3f}")
    
    bimodal_gen = bimodal_data_generator()
    samples = bimodal_gen(100)
    print(f"Bimodal samples: mean={np.mean(samples):.3f}, std={np.std(samples):.3f}")
    
    print("\n--- Trainer components ready ---")
    print("To run full training, use: train_simple_qgan()")
