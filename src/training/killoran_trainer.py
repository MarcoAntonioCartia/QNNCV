"""
Killoran CV-QNN Trainer for Bimodal Distributions
===================================================

Tests whether the Kerr gate enables learning non-Gaussian distributions
like bimodal Gaussian mixtures.

Key hypothesis: Without Kerr → only Gaussian outputs (single peak)
                With Kerr → can learn non-Gaussian (potentially bimodal)
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import argparse

# Import our modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generators.killoran_cvqnn import (
    KilloranCVQNN,
    DistributionDiscriminator,
    gaussian_distribution,
    bimodal_gaussian
)


@dataclass
class KilloranTrainerConfig:
    """Configuration for Killoran CV-QNN training."""
    # Target distribution
    target_type: str = 'bimodal'  # 'gaussian' or 'bimodal'
    
    # For Gaussian target
    target_mean: float = 0.0
    target_std: float = 1.0
    
    # For bimodal target
    mean1: float = -1.5
    std1: float = 0.5
    mean2: float = 1.5
    std2: float = 0.5
    weight1: float = 0.5
    
    # Training
    batch_size: int = 16
    g_lr: float = 0.005
    d_lr: float = 0.001
    n_critic: int = 5
    
    # WGAN-GP
    gp_weight: float = 10.0
    
    # Architecture
    n_layers: int = 4
    cutoff_dim: int = 15
    use_kerr: bool = True
    num_bins: int = 100


class KilloranQGANTrainer:
    """Trainer for Killoran CV-QNN architecture."""
    
    def __init__(
        self,
        generator: KilloranCVQNN,
        discriminator: DistributionDiscriminator,
        config: KilloranTrainerConfig
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        
        # Optimizers
        self.g_optimizer = tf.optimizers.Adam(learning_rate=config.g_lr, beta_1=0.5)
        self.d_optimizer = tf.optimizers.Adam(learning_rate=config.d_lr, beta_1=0.5)
        
        self.xvec = generator.xvec
        
        # Create target distribution
        if config.target_type == 'bimodal':
            self.P_real = bimodal_gaussian(
                self.xvec, 
                config.mean1, config.std1, config.weight1,
                config.mean2, config.std2
            )
            self.target_desc = f"Bimodal: {config.weight1:.1f}*N({config.mean1},{config.std1}²) + {1-config.weight1:.1f}*N({config.mean2},{config.std2}²)"
        else:
            self.P_real = gaussian_distribution(self.xvec, config.target_mean, config.target_std)
            self.target_desc = f"Gaussian: N({config.target_mean}, {config.target_std}²)"
        
        # History
        self.history = {
            'g_loss': [], 'd_loss': [], 
            'wasserstein': [], 'kl_div': [],
            'peak_count': []
        }
    
    def _gradient_penalty(self, P_real: tf.Tensor, P_fake: tf.Tensor) -> tf.Tensor:
        """Gradient penalty for WGAN-GP."""
        batch_size = tf.shape(P_fake)[0]
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        P_real_batch = tf.tile(tf.expand_dims(self.P_real, 0), [batch_size, 1])
        interpolated = alpha * P_real_batch + (1 - alpha) * P_fake
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            scores = self.discriminator.discriminate(interpolated)
        
        gradients = tape.gradient(scores, interpolated)
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-10)
        return tf.reduce_mean(tf.square(grad_norm - 1.0))
    
    def train_discriminator_step(self, z: tf.Tensor) -> tf.Tensor:
        """One discriminator training step."""
        batch_size = tf.shape(z)[0]
        
        with tf.GradientTape() as tape:
            P_fake = self.generator.generate(z)
            P_real_batch = tf.tile(tf.expand_dims(self.P_real, 0), [batch_size, 1])
            
            real_scores = self.discriminator.discriminate(P_real_batch)
            fake_scores = self.discriminator.discriminate(P_fake)
            
            d_loss = tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)
            gp = self._gradient_penalty(P_real_batch, P_fake)
            d_loss = d_loss + self.config.gp_weight * gp
        
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        
        return d_loss
    
    def train_generator_step(self, z: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """One generator training step."""
        with tf.GradientTape() as tape:
            P_fake = self.generator.generate(z)
            fake_scores = self.discriminator.discriminate(P_fake)
            g_loss = -tf.reduce_mean(fake_scores)
        
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        
        return g_loss, P_fake
    
    def compute_wasserstein(self, P_gen: tf.Tensor) -> float:
        """Compute W₁ distance."""
        dx = self.xvec[1] - self.xvec[0]
        cdf_gen = tf.cumsum(P_gen) * dx
        cdf_real = tf.cumsum(self.P_real) * dx
        w1 = tf.reduce_sum(tf.abs(cdf_gen - cdf_real)) * dx
        return float(w1.numpy())
    
    def count_peaks(self, P_x: tf.Tensor, threshold: float = 0.1) -> int:
        """
        Count number of peaks in distribution.
        A peak is a local maximum above threshold * max_value.
        """
        P_np = P_x.numpy()
        max_val = np.max(P_np)
        min_height = threshold * max_val
        
        # Find local maxima
        peaks = []
        for i in range(1, len(P_np) - 1):
            if P_np[i] > P_np[i-1] and P_np[i] > P_np[i+1] and P_np[i] > min_height:
                peaks.append(i)
        
        return len(peaks)
    
    def train(
        self,
        epochs: int,
        log_interval: int = 10,
        checkpoint_interval: int = 50,
        log_dir: Optional[str] = None
    ) -> Dict:
        """Main training loop."""
        
        print("=" * 70)
        print("Killoran CV-QNN Training")
        print("=" * 70)
        print(f"Architecture: {'WITH Kerr gate' if self.config.use_kerr else 'WITHOUT Kerr gate'}")
        print(f"Target: {self.target_desc}")
        print(f"Layers: {self.config.n_layers}, Cutoff: {self.config.cutoff_dim}")
        print(f"Generator params: {self.generator.num_params}")
        print(f"Discriminator params: {self.discriminator.num_params}")
        print("=" * 70)
        
        best_wasserstein = float('inf')
        best_epoch = 0
        
        for epoch in range(1, epochs + 1):
            # Train discriminator
            for _ in range(self.config.n_critic):
                z_d = tf.random.normal([self.config.batch_size, self.generator.latent_dim])
                d_loss = self.train_discriminator_step(z_d)
            
            # Train generator
            z = tf.random.normal([self.config.batch_size, self.generator.latent_dim])
            g_loss, P_fake = self.train_generator_step(z)
            
            # Metrics
            P_gen_sample = P_fake[0]
            wasserstein = self.compute_wasserstein(P_gen_sample)
            peak_count = self.count_peaks(P_gen_sample)
            
            # Store history
            self.history['g_loss'].append(float(g_loss.numpy()))
            self.history['d_loss'].append(float(d_loss.numpy()))
            self.history['wasserstein'].append(wasserstein)
            self.history['peak_count'].append(peak_count)
            
            # Track best
            if wasserstein < best_wasserstein:
                best_wasserstein = wasserstein
                best_epoch = epoch
            
            # Log
            if epoch % log_interval == 0:
                print(f"Epoch {epoch:4d}/{epochs} | "
                      f"G: {g_loss:7.4f} | D: {d_loss:7.4f} | "
                      f"W₁: {wasserstein:.4f} | Peaks: {peak_count}")
            
            # Checkpoint
            if log_dir and epoch % checkpoint_interval == 0:
                self._save_checkpoint(epoch, log_dir)
        
        # Final summary
        summary = {
            'final_wasserstein': self.history['wasserstein'][-1],
            'best_wasserstein': best_wasserstein,
            'best_epoch': best_epoch,
            'final_peak_count': self.history['peak_count'][-1],
            'use_kerr': self.config.use_kerr
        }
        
        print("=" * 70)
        print("Training Complete!")
        print(f"Best W₁: {best_wasserstein:.4f} at epoch {best_epoch}")
        print(f"Final peak count: {summary['final_peak_count']} (target: 2 for bimodal)")
        print("=" * 70)
        
        return summary
    
    def _save_checkpoint(self, epoch: int, log_dir: str):
        """Save checkpoint."""
        os.makedirs(log_dir, exist_ok=True)
        np.save(os.path.join(log_dir, f'weights_epoch_{epoch}.npy'), 
                self.generator.weights.numpy())
        np.save(os.path.join(log_dir, 'history.npy'), self.history)
    
    def plot_comparison(self, save_path: Optional[str] = None, show: bool = True):
        """Plot generated vs target distribution."""
        # Generate distribution
        z = tf.random.normal([1, self.generator.latent_dim])
        P_gen = self.generator.generate(z)[0]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Distribution comparison
        axes[0].plot(self.xvec.numpy(), self.P_real.numpy(), 'b-', 
                     label='Target', linewidth=2)
        axes[0].plot(self.xvec.numpy(), P_gen.numpy(), 'r--', 
                     label='Generated', linewidth=2)
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('P(x)')
        kerr_str = "WITH Kerr" if self.config.use_kerr else "WITHOUT Kerr"
        axes[0].set_title(f'Distribution Comparison ({kerr_str})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Wasserstein history
        axes[1].plot(self.history['wasserstein'], 'g-', linewidth=1)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Wasserstein Distance')
        axes[1].set_title('Training Progress')
        axes[1].grid(True, alpha=0.3)
        
        # Peak count history
        axes[2].plot(self.history['peak_count'], 'purple', linewidth=1)
        axes[2].axhline(y=2, color='r', linestyle='--', label='Target (bimodal)')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Number of Peaks')
        axes[2].set_title('Peak Count Evolution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Killoran CV-QNN Training')
    
    # Architecture
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--cutoff-dim', type=int, default=15)
    parser.add_argument('--use-kerr', action='store_true', default=True)
    parser.add_argument('--no-kerr', dest='use_kerr', action='store_false')
    
    # Target
    parser.add_argument('--target-type', type=str, default='bimodal', 
                        choices=['gaussian', 'bimodal'])
    parser.add_argument('--mean1', type=float, default=-1.5)
    parser.add_argument('--std1', type=float, default=0.5)
    parser.add_argument('--mean2', type=float, default=1.5)
    parser.add_argument('--std2', type=float, default=0.5)
    parser.add_argument('--weight1', type=float, default=0.5)
    
    # Training
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--g-lr', type=float, default=0.005)
    parser.add_argument('--d-lr', type=float, default=0.001)
    parser.add_argument('--n-critic', type=int, default=5)
    
    # Output
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default='./logs')
    
    args = parser.parse_args()
    
    # Build experiment name
    if args.exp_name:
        exp_name = args.exp_name
    else:
        kerr_str = "kerr" if args.use_kerr else "nokerr"
        if args.target_type == 'bimodal':
            exp_name = f"killoran_{kerr_str}_bimodal_{args.mean1}_{args.mean2}"
        else:
            exp_name = f"killoran_{kerr_str}_gaussian"
    
    log_dir = os.path.join(args.log_dir, exp_name)
    
    # Create config
    config = KilloranTrainerConfig(
        target_type=args.target_type,
        mean1=args.mean1,
        std1=args.std1,
        mean2=args.mean2,
        std2=args.std2,
        weight1=args.weight1,
        n_layers=args.n_layers,
        cutoff_dim=args.cutoff_dim,
        use_kerr=args.use_kerr,
        batch_size=args.batch_size,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        n_critic=args.n_critic
    )
    
    # Create generator
    print("\nInitializing Killoran CV-QNN Generator...")
    generator = KilloranCVQNN(
        n_layers=config.n_layers,
        cutoff_dim=config.cutoff_dim,
        use_kerr=config.use_kerr
    )
    print(f"  Config: {generator.get_config()}")
    
    # Create discriminator
    print("\nInitializing Discriminator...")
    discriminator = DistributionDiscriminator(
        input_dim=100,
        hidden_dims=[128, 64, 32]
    )
    print(f"  Params: {discriminator.num_params}")
    
    # Create trainer
    trainer = KilloranQGANTrainer(generator, discriminator, config)
    
    # Train
    summary = trainer.train(
        epochs=args.epochs,
        log_interval=10,
        checkpoint_interval=50,
        log_dir=log_dir
    )
    
    # Plot
    trainer.plot_comparison(
        save_path=os.path.join(log_dir, 'final_comparison.png'),
        show=False
    )
    
    return summary


if __name__ == "__main__":
    main()
