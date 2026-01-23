"""
Distribution-Based QGAN Trainer
================================

Trains a quantum generator to produce homodyne probability distributions
that match a target distribution, using adversarial training.

Key difference from scalar QGAN:
- Generator outputs P_gen(x) [batch_size, num_bins]
- Discriminator inputs P(x) distributions, not scalars
- Real data is also a distribution P_real(x)
"""

import os
import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

# Import our modules
from models.generators.quantum_distribution_generator import (
    QuantumDistributionGenerator, 
    gaussian_distribution
)
from models.discriminators.distribution_discriminator import DistributionDiscriminator


@dataclass
class DistributionTrainerConfig:
    """Configuration for distribution-based QGAN training."""
    # Target distribution
    target_mean: float = 2.0
    target_std: float = 0.5
    
    # Training
    batch_size: int = 32
    g_lr: float = 0.001
    d_lr: float = 0.001
    n_critic: int = 5  # D updates per G update
    
    # WGAN-GP
    use_wgan_gp: bool = True
    gp_weight: float = 10.0
    
    # Output
    num_bins: int = 100
    x_min: float = -5.0
    x_max: float = 5.0


class DistributionQGANTrainer:
    """
    Trainer for distribution-based Quantum GAN.
    
    The generator outputs P_gen(x), which is compared to P_real(x) by
    the discriminator. Both are distribution vectors of shape [num_bins].
    """
    
    def __init__(
        self,
        generator: QuantumDistributionGenerator,
        discriminator: DistributionDiscriminator,
        config: DistributionTrainerConfig
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        
        # Optimizers
        self.g_optimizer = tf.optimizers.Adam(learning_rate=config.g_lr, beta_1=0.5)
        self.d_optimizer = tf.optimizers.Adam(learning_rate=config.d_lr, beta_1=0.5)
        
        # x-grid (same as generator)
        self.xvec = generator.xvec
        
        # Target distribution (fixed)
        self.P_real = gaussian_distribution(self.xvec, config.target_mean, config.target_std)
        
        # Metrics storage
        self.history = {
            'g_loss': [], 'd_loss': [], 
            'wasserstein': [], 
            'gen_mean': [], 'gen_std': []
        }
    
    def _gradient_penalty(self, P_real: tf.Tensor, P_fake: tf.Tensor) -> tf.Tensor:
        """
        Compute gradient penalty for WGAN-GP.
        
        Interpolates between real and fake distributions.
        """
        batch_size = tf.shape(P_fake)[0]
        
        # Random interpolation coefficient
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        
        # Interpolated distributions
        P_real_batch = tf.tile(tf.expand_dims(self.P_real, 0), [batch_size, 1])
        interpolated = alpha * P_real_batch + (1 - alpha) * P_fake
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            scores = self.discriminator.discriminate(interpolated)
        
        gradients = tape.gradient(scores, interpolated)
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-10)
        penalty = tf.reduce_mean(tf.square(grad_norm - 1.0))
        
        return penalty
    
    def train_discriminator_step(self, z: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        One discriminator training step.
        
        Returns: (d_loss, d_gradients)
        """
        batch_size = tf.shape(z)[0]
        
        with tf.GradientTape() as tape:
            # Generate fake distributions
            P_fake = self.generator.generate(z)  # [batch_size, num_bins]
            
            # Real distribution (same for all samples)
            P_real_batch = tf.tile(tf.expand_dims(self.P_real, 0), [batch_size, 1])
            
            # Discriminator scores
            real_scores = self.discriminator.discriminate(P_real_batch)
            fake_scores = self.discriminator.discriminate(P_fake)
            
            # WGAN loss: D wants to maximize D(real) - D(fake)
            # Minimize: D(fake) - D(real)
            d_loss = tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)
            
            # Gradient penalty
            if self.config.use_wgan_gp:
                gp = self._gradient_penalty(P_real_batch, P_fake)
                d_loss = d_loss + self.config.gp_weight * gp
        
        # Compute and apply gradients
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables)
        )
        
        return d_loss, d_grads
    
    def train_generator_step(self, z: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        One generator training step.
        
        Returns: (g_loss, g_gradients, P_fake)
        """
        with tf.GradientTape() as tape:
            # Generate fake distributions
            P_fake = self.generator.generate(z)
            
            # Discriminator scores fake
            fake_scores = self.discriminator.discriminate(P_fake)
            
            # Generator wants to maximize D(fake) = minimize -D(fake)
            g_loss = -tf.reduce_mean(fake_scores)
        
        # Compute and apply gradients
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_variables)
        )
        
        return g_loss, g_grads, P_fake
    
    def compute_distribution_stats(self, P_x: tf.Tensor) -> Tuple[float, float]:
        """Compute mean and std of a distribution."""
        dx = self.xvec[1] - self.xvec[0]
        
        # Mean: E[x] = Σ x * P(x) * dx
        mean = tf.reduce_sum(P_x * self.xvec) * dx
        
        # Variance: E[x²] - E[x]²
        mean_sq = tf.reduce_sum(P_x * self.xvec ** 2) * dx
        var = mean_sq - mean ** 2
        std = tf.sqrt(tf.maximum(var, 1e-10))
        
        return float(mean.numpy()), float(std.numpy())
    
    def compute_wasserstein(self, P_gen: tf.Tensor) -> float:
        """Compute Wasserstein-1 distance between generated and real distributions."""
        dx = self.xvec[1] - self.xvec[0]
        
        # CDFs
        cdf_gen = tf.cumsum(P_gen) * dx
        cdf_real = tf.cumsum(self.P_real) * dx
        
        # W₁ = ∫|CDF_gen - CDF_real|dx
        w1 = tf.reduce_sum(tf.abs(cdf_gen - cdf_real)) * dx
        
        return float(w1.numpy())
    
    def train(
        self,
        epochs: int,
        log_interval: int = 10,
        checkpoint_interval: int = 100,
        log_dir: Optional[str] = None
    ) -> Dict:
        """
        Main training loop.
        
        Args:
            epochs: Number of training epochs
            log_interval: How often to print progress
            checkpoint_interval: How often to save checkpoints
            log_dir: Directory for saving logs/checkpoints
        
        Returns:
            Training summary dictionary
        """
        print("=" * 60)
        print("Distribution-Based QGAN Training")
        print("=" * 60)
        print(f"Generator: {self.generator.get_config()}")
        print(f"Target: N({self.config.target_mean}, {self.config.target_std}²)")
        print(f"Epochs: {epochs}, Batch size: {self.config.batch_size}")
        print("=" * 60)
        
        best_wasserstein = float('inf')
        best_epoch = 0
        
        for epoch in range(1, epochs + 1):
            # Sample latent vectors
            z = tf.random.normal([self.config.batch_size, self.generator.latent_dim])
            
            # Train discriminator (n_critic times)
            d_loss = None
            for _ in range(self.config.n_critic):
                z_d = tf.random.normal([self.config.batch_size, self.generator.latent_dim])
                d_loss, _ = self.train_discriminator_step(z_d)
            
            # Train generator (once)
            g_loss, _, P_fake = self.train_generator_step(z)
            
            # Compute metrics on first sample of batch
            P_gen_sample = P_fake[0]
            gen_mean, gen_std = self.compute_distribution_stats(P_gen_sample)
            wasserstein = self.compute_wasserstein(P_gen_sample)
            
            # Store history
            self.history['g_loss'].append(float(g_loss.numpy()))
            self.history['d_loss'].append(float(d_loss.numpy()))
            self.history['wasserstein'].append(wasserstein)
            self.history['gen_mean'].append(gen_mean)
            self.history['gen_std'].append(gen_std)
            
            # Track best
            if wasserstein < best_wasserstein:
                best_wasserstein = wasserstein
                best_epoch = epoch
            
            # Log progress
            if epoch % log_interval == 0:
                print(f"Epoch {epoch}/{epochs} | "
                      f"G_loss: {g_loss:.4f} | D_loss: {d_loss:.4f} | "
                      f"Mean: {gen_mean:.3f} (target: {self.config.target_mean}) | "
                      f"Std: {gen_std:.3f} (target: {self.config.target_std}) | "
                      f"W₁: {wasserstein:.4f}")
            
            # Checkpoint
            if log_dir and epoch % checkpoint_interval == 0:
                self._save_checkpoint(epoch, log_dir)
        
        # Summary
        summary = {
            'final_g_loss': self.history['g_loss'][-1],
            'final_d_loss': self.history['d_loss'][-1],
            'final_mean': self.history['gen_mean'][-1],
            'final_std': self.history['gen_std'][-1],
            'final_wasserstein': self.history['wasserstein'][-1],
            'best_wasserstein': best_wasserstein,
            'best_epoch': best_epoch
        }
        
        print("=" * 60)
        print("Training Complete!")
        print(f"Best Wasserstein: {best_wasserstein:.4f} at epoch {best_epoch}")
        print(f"Final Mean: {summary['final_mean']:.3f} (target: {self.config.target_mean})")
        print(f"Final Std: {summary['final_std']:.3f} (target: {self.config.target_std})")
        print("=" * 60)
        
        return summary
    
    def _save_checkpoint(self, epoch: int, log_dir: str):
        """Save training checkpoint."""
        os.makedirs(log_dir, exist_ok=True)
        
        # Save weights
        np.save(os.path.join(log_dir, f'generator_epoch_{epoch}.npy'), 
                self.generator.weights.numpy())
        
        # Save history
        np.save(os.path.join(log_dir, 'history.npy'), self.history)
        
        print(f"Checkpoint saved at epoch {epoch}")
    
    def plot_comparison(self, save_path: Optional[str] = None):
        """Plot generated vs target distribution."""
        import matplotlib.pyplot as plt
        
        # Generate a distribution
        z = tf.random.normal([1, self.generator.latent_dim])
        P_gen = self.generator.generate(z)[0]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Distribution comparison
        axes[0].plot(self.xvec.numpy(), self.P_real.numpy(), 'b-', label='Target', linewidth=2)
        axes[0].plot(self.xvec.numpy(), P_gen.numpy(), 'r--', label='Generated', linewidth=2)
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('P(x)')
        axes[0].set_title('Distribution Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Training history
        axes[1].plot(self.history['wasserstein'], 'g-', linewidth=1)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Wasserstein Distance')
        axes[1].set_title('Training Progress')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            # Create directory if needed
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Distribution-Based QGAN Training')
    
    # Model
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--cutoff-dim', type=int, default=10)
    parser.add_argument('--num-bins', type=int, default=100)
    
    # Target
    parser.add_argument('--target-mean', type=float, default=2.0)
    parser.add_argument('--target-std', type=float, default=0.5)
    
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--g-lr', type=float, default=0.005)
    parser.add_argument('--d-lr', type=float, default=0.001)
    parser.add_argument('--n-critic', type=int, default=5)
    
    # Output
    parser.add_argument('--log-dir', type=str, default='./logs/dist_qgan')
    
    args = parser.parse_args()
    
    # Create generator
    generator = QuantumDistributionGenerator(
        n_modes=1,
        n_layers=args.n_layers,
        cutoff_dim=args.cutoff_dim,
        num_bins=args.num_bins,
        x_min=-5.0,
        x_max=5.0
    )
    
    # Create discriminator
    discriminator = DistributionDiscriminator(
        input_dim=args.num_bins,
        hidden_dims=[64, 32]
    )
    
    # Config
    config = DistributionTrainerConfig(
        target_mean=args.target_mean,
        target_std=args.target_std,
        batch_size=args.batch_size,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        n_critic=args.n_critic,
        num_bins=args.num_bins
    )
    
    # Create trainer
    trainer = DistributionQGANTrainer(generator, discriminator, config)
    
    # Train
    summary = trainer.train(
        epochs=args.epochs,
        log_interval=10,
        checkpoint_interval=100,
        log_dir=args.log_dir
    )
    
    # Plot results
    trainer.plot_comparison(save_path=os.path.join(args.log_dir, 'final_comparison.png'))
    
    return summary


if __name__ == "__main__":
    main()
