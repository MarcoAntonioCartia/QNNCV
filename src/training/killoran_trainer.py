"""
Killoran CV-QNN Trainer for Multi-Modal Distributions
======================================================

Tests the expressivity of a single qumode CV-QNN by learning N-modal
Gaussian mixture distributions.

Key Features:
- N-modal target distributions (1, 2, 3, 4, 5+ peaks)
- Comprehensive monitoring (losses, gradients, weights, quantum state)
- Wigner function visualization
- Smooth metrics with moving average
- Weight evolution tracking

Why is the GAN learning curve noisy?
------------------------------------
1. Adversarial dynamics: G and D are competing, causing oscillations
2. Stochastic batches: Different random latents each epoch
3. Non-convex landscape: Quantum circuits have complex parameter spaces
4. Mode switching: G may oscillate between different solutions

We use moving averages and best-checkpoint tracking to handle this.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass, field
import argparse
import json

# Import our modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generators.killoran_cvqnn import (
    KilloranCVQNN,
    DistributionDiscriminator,
    gaussian_distribution,
    bimodal_gaussian,
    n_modal_gaussian,
    get_target_description
)


@dataclass
class KilloranTrainerConfig:
    """Configuration for Killoran CV-QNN training."""
    # Target distribution - N-modal
    n_peaks: int = 2  # Number of peaks (modes) in target
    x_min: float = -2.0  # Left boundary for peaks
    x_max: float = 2.0   # Right boundary for peaks
    peak_std: float = 0.3  # Std of each peak
    
    # Legacy bimodal params (for backward compatibility)
    target_type: str = 'nmodal'  # 'gaussian', 'bimodal', or 'nmodal'
    mean1: float = -1.5
    std1: float = 0.5
    mean2: float = 1.5
    std2: float = 0.5
    weight1: float = 0.5
    target_mean: float = 0.0
    target_std: float = 1.0
    
    # Training
    batch_size: int = 16
    g_lr: float = 0.005
    d_lr: float = 0.001
    n_critic: int = 5
    
    # WGAN-GP
    gp_weight: float = 10.0
    
    # Architecture
    n_layers: int = 6
    cutoff_dim: int = 6
    use_kerr: bool = True
    num_bins: int = 100
    
    # Monitoring
    save_wigner: bool = True
    wigner_epochs: List[int] = field(default_factory=lambda: [1, 10, 50, 100, 200, 300])
    smooth_window: int = 10  # Moving average window


class WeightTracker:
    """Tracks weight evolution during training."""
    
    def __init__(self, generator: KilloranCVQNN):
        self.generator = generator
        self.history = {
            'epochs': [],
            'weight_stats': [],  # Mean, std, norm per epoch
            'weights_by_layer': [],  # Full weight snapshots
            'kerr_params': [],  # Just Kerr parameters over time
            'squeeze_params': [],  # Squeeze magnitudes
            'displacement_params': []  # Displacement magnitudes
        }
    
    def record(self, epoch: int):
        """Record current weight state."""
        self.history['epochs'].append(epoch)
        self.history['weight_stats'].append(self.generator.get_weight_statistics())
        
        # Get organized weights
        weights = self.generator.get_weights_by_layer()
        
        # Extract specific gate types
        kerr_vals = [v for k, v in weights.items() if '_K' in k]
        squeeze_r = [v for k, v in weights.items() if '_S_r' in k]
        disp_r = [v for k, v in weights.items() if '_D_r' in k]
        
        self.history['kerr_params'].append(kerr_vals)
        self.history['squeeze_params'].append(squeeze_r)
        self.history['displacement_params'].append(disp_r)
    
    def plot_weight_evolution(self, save_path: Optional[str] = None):
        """Plot how weights evolve during training."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = self.history['epochs']
        
        # 1. Overall weight statistics
        ax = axes[0, 0]
        stats = self.history['weight_stats']
        means = [s['mean'] for s in stats]
        stds = [s['std'] for s in stats]
        norms = [s['norm'] for s in stats]
        ax.plot(epochs, means, label='Mean', color='blue')
        ax.fill_between(epochs, 
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.3, color='blue')
        ax.plot(epochs, norms, label='L2 Norm', color='red', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_title('Weight Statistics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Kerr parameters (if available)
        ax = axes[0, 1]
        if self.history['kerr_params'] and len(self.history['kerr_params'][0]) > 0:
            kerr_array = np.array(self.history['kerr_params'])
            for i in range(kerr_array.shape[1]):
                ax.plot(epochs, kerr_array[:, i], label=f'Layer {i}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Kerr κ')
            ax.set_title('Kerr Gate Parameters')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Kerr gates disabled', ha='center', va='center')
            ax.set_title('Kerr Gate Parameters')
        
        # 3. Squeeze parameters
        ax = axes[1, 0]
        if self.history['squeeze_params']:
            squeeze_array = np.array(self.history['squeeze_params'])
            for i in range(squeeze_array.shape[1]):
                ax.plot(epochs, squeeze_array[:, i], label=f'Layer {i}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Squeeze r')
            ax.set_title('Squeeze Gate Magnitudes')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Displacement parameters
        ax = axes[1, 1]
        if self.history['displacement_params']:
            disp_array = np.array(self.history['displacement_params'])
            for i in range(disp_array.shape[1]):
                ax.plot(epochs, disp_array[:, i], label=f'Layer {i}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Displacement r')
            ax.set_title('Displacement Gate Magnitudes')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Weight evolution plot saved to {save_path}")
        
        plt.close()
        return fig


class KilloranQGANTrainer:
    """Trainer for Killoran CV-QNN architecture with comprehensive monitoring."""
    
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
        
        # Create target distribution based on type
        if config.target_type == 'nmodal':
            self.P_real, self.peak_positions = n_modal_gaussian(
                self.xvec,
                n_peaks=config.n_peaks,
                x_min=config.x_min,
                x_max=config.x_max,
                std=config.peak_std
            )
            self.target_desc = get_target_description(
                config.n_peaks, config.x_min, config.x_max, config.peak_std
            )
        elif config.target_type == 'bimodal':
            self.P_real = bimodal_gaussian(
                self.xvec, 
                config.mean1, config.std1, config.weight1,
                config.mean2, config.std2
            )
            self.peak_positions = [config.mean1, config.mean2]
            self.target_desc = f"Bimodal: {config.weight1:.1f}*N({config.mean1},{config.std1}²) + {1-config.weight1:.1f}*N({config.mean2},{config.std2}²)"
        else:
            self.P_real = gaussian_distribution(self.xvec, config.target_mean, config.target_std)
            self.peak_positions = [config.target_mean]
            self.target_desc = f"Gaussian: N({config.target_mean}, {config.target_std}²)"
        
        # History with extended metrics
        self.history = {
            'g_loss': [], 'd_loss': [], 
            'wasserstein': [], 'wasserstein_smooth': [],
            'kl_div': [],
            'peak_count': [],
            'g_grad_norm': [], 'd_grad_norm': [],
            'epochs': []
        }
        
        # Weight tracker
        self.weight_tracker = WeightTracker(generator)
        
        # Best model tracking
        self.best_weights = None
        self.best_wasserstein = float('inf')
        self.best_epoch = 0
        
        # Wigner snapshots
        self.wigner_states = {}
        self.fixed_latent = tf.random.normal([1, generator.latent_dim])
    
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
    
    def train_discriminator_step(self, z: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """One discriminator training step. Returns loss and gradient norm."""
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
        
        # Compute gradient norm
        grad_norm = tf.sqrt(sum(tf.reduce_sum(tf.square(g)) for g in d_grads if g is not None))
        
        return d_loss, grad_norm
    
    def train_generator_step(self, z: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """One generator training step. Returns loss, generated distribution, gradient norm."""
        with tf.GradientTape() as tape:
            P_fake = self.generator.generate(z)
            fake_scores = self.discriminator.discriminate(P_fake)
            g_loss = -tf.reduce_mean(fake_scores)
        
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        
        # Compute gradient norm
        grad_norm = tf.sqrt(sum(tf.reduce_sum(tf.square(g)) for g in g_grads if g is not None))
        
        return g_loss, P_fake, grad_norm
    
    def compute_wasserstein(self, P_gen: tf.Tensor) -> float:
        """Compute W₁ distance."""
        dx = self.xvec[1] - self.xvec[0]
        cdf_gen = tf.cumsum(P_gen) * dx
        cdf_real = tf.cumsum(self.P_real) * dx
        w1 = tf.reduce_sum(tf.abs(cdf_gen - cdf_real)) * dx
        return float(w1.numpy())
    
    def count_peaks(self, P_x: tf.Tensor, threshold: float = 0.1) -> int:
        """Count number of peaks in distribution."""
        P_np = P_x.numpy()
        max_val = np.max(P_np)
        min_height = threshold * max_val
        
        peaks = []
        for i in range(1, len(P_np) - 1):
            if P_np[i] > P_np[i-1] and P_np[i] > P_np[i+1] and P_np[i] > min_height:
                peaks.append(i)
        
        return len(peaks)
    
    def moving_average(self, values: List[float], window: int) -> List[float]:
        """Compute moving average for smoothing."""
        if len(values) < window:
            return values
        
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(values[start:i+1]))
        return smoothed
    
    def capture_wigner(self, epoch: int, log_dir: str):
        """Capture Wigner function for visualization."""
        if epoch not in self.config.wigner_epochs:
            return
        
        try:
            state = self.generator.get_quantum_state(self.fixed_latent[0])
            self.wigner_states[epoch] = state
            
            # Plot and save
            xvec = np.linspace(-4, 4, 80)
            pvec = np.linspace(-4, 4, 80)
            W = state.wigner(0, xvec, pvec)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            contour = ax.contourf(xvec, pvec, W, levels=50, cmap='RdBu')
            ax.set_xlabel('x (position)')
            ax.set_ylabel('p (momentum)')
            ax.set_title(f'Wigner Function - Epoch {epoch}')
            ax.set_aspect('equal')
            fig.colorbar(contour, ax=ax, label='W(x,p)')
            
            save_path = os.path.join(log_dir, f'wigner_epoch_{epoch}.png')
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  [Wigner saved: epoch {epoch}]")
            
        except Exception as e:
            print(f"  [Warning: Could not capture Wigner: {e}]")
    
    def train(
        self,
        epochs: int,
        log_interval: int = 10,
        checkpoint_interval: int = 50,
        log_dir: Optional[str] = None
    ) -> Dict:
        """Main training loop with comprehensive monitoring."""
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        print("=" * 70)
        print("Killoran CV-QNN Training")
        print("=" * 70)
        print(f"Architecture: {'WITH Kerr gate' if self.config.use_kerr else 'WITHOUT Kerr gate'}")
        print(f"Target: {self.target_desc}")
        print(f"Layers: {self.config.n_layers}, Cutoff: {self.config.cutoff_dim}")
        print(f"Generator params: {self.generator.num_params}")
        print(f"Discriminator params: {self.discriminator.num_params}")
        print(f"Target peaks: {self.config.n_peaks}")
        print("=" * 70)
        
        for epoch in range(1, epochs + 1):
            # Train discriminator
            d_grad_norm_sum = 0.0
            for _ in range(self.config.n_critic):
                z_d = tf.random.normal([self.config.batch_size, self.generator.latent_dim])
                d_loss, d_grad_norm = self.train_discriminator_step(z_d)
                d_grad_norm_sum += float(d_grad_norm.numpy())
            d_grad_norm_avg = d_grad_norm_sum / self.config.n_critic
            
            # Train generator
            z = tf.random.normal([self.config.batch_size, self.generator.latent_dim])
            g_loss, P_fake, g_grad_norm = self.train_generator_step(z)
            
            # Metrics
            P_gen_sample = P_fake[0]
            wasserstein = self.compute_wasserstein(P_gen_sample)
            peak_count = self.count_peaks(P_gen_sample)
            
            # Store history
            self.history['epochs'].append(epoch)
            self.history['g_loss'].append(float(g_loss.numpy()))
            self.history['d_loss'].append(float(d_loss.numpy()))
            self.history['wasserstein'].append(wasserstein)
            self.history['peak_count'].append(peak_count)
            self.history['g_grad_norm'].append(float(g_grad_norm.numpy()))
            self.history['d_grad_norm'].append(d_grad_norm_avg)
            
            # Smoothed Wasserstein
            self.history['wasserstein_smooth'] = self.moving_average(
                self.history['wasserstein'], 
                self.config.smooth_window
            )
            
            # Track weights
            if epoch % 10 == 0:
                self.weight_tracker.record(epoch)
            
            # Track best
            if wasserstein < self.best_wasserstein:
                self.best_wasserstein = wasserstein
                self.best_epoch = epoch
                self.best_weights = self.generator.weights.numpy().copy()
            
            # Capture Wigner function
            if log_dir and self.config.save_wigner:
                self.capture_wigner(epoch, log_dir)
            
            # Log
            if epoch % log_interval == 0:
                w_smooth = self.history['wasserstein_smooth'][-1]
                print(f"Epoch {epoch:4d}/{epochs} | "
                      f"G: {g_loss:7.4f} | D: {d_loss:7.4f} | "
                      f"W₁: {wasserstein:.4f} (smooth: {w_smooth:.4f}) | "
                      f"Peaks: {peak_count}/{self.config.n_peaks}")
            
            # Checkpoint
            if log_dir and epoch % checkpoint_interval == 0:
                self._save_checkpoint(epoch, log_dir)
        
        # Final summary
        summary = {
            'final_wasserstein': self.history['wasserstein'][-1],
            'best_wasserstein': self.best_wasserstein,
            'best_epoch': self.best_epoch,
            'final_peak_count': self.history['peak_count'][-1],
            'target_peaks': self.config.n_peaks,
            'use_kerr': self.config.use_kerr,
            'n_layers': self.config.n_layers,
            'cutoff_dim': self.config.cutoff_dim
        }
        
        print("=" * 70)
        print("Training Complete!")
        print(f"Best W₁: {self.best_wasserstein:.4f} at epoch {self.best_epoch}")
        print(f"Final peak count: {summary['final_peak_count']} (target: {self.config.n_peaks})")
        print("=" * 70)
        
        return summary
    
    def _save_checkpoint(self, epoch: int, log_dir: str):
        """Save checkpoint with metrics."""
        os.makedirs(log_dir, exist_ok=True)
        
        # Save weights
        np.save(os.path.join(log_dir, f'weights_epoch_{epoch}.npy'), 
                self.generator.weights.numpy())
        
        # Save best weights
        if self.best_weights is not None:
            np.save(os.path.join(log_dir, 'best_weights.npy'), self.best_weights)
        
        # Save history
        with open(os.path.join(log_dir, 'history.json'), 'w') as f:
            # Convert to serializable
            history_save = {k: [float(x) for x in v] if isinstance(v, list) else v 
                          for k, v in self.history.items()}
            json.dump(history_save, f, indent=2)
    
    def plot_comparison(self, save_path: Optional[str] = None, show: bool = True, use_best: bool = True):
        """Plot generated vs target distribution."""
        # Use best weights if available
        if use_best and self.best_weights is not None:
            original_weights = self.generator.weights.numpy().copy()
            self.generator.weights.assign(self.best_weights)
        
        # Generate distribution
        z = tf.random.normal([1, self.generator.latent_dim])
        P_gen = self.generator.generate(z)[0]
        
        # Restore weights
        if use_best and self.best_weights is not None:
            self.generator.weights.assign(original_weights)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Distribution comparison
        axes[0].plot(self.xvec.numpy(), self.P_real.numpy(), 'b-', 
                     label='Target', linewidth=2)
        axes[0].plot(self.xvec.numpy(), P_gen.numpy(), 'r--', 
                     label='Generated', linewidth=2)
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('P(x)')
        kerr_str = "WITH Kerr" if self.config.use_kerr else "WITHOUT Kerr"
        title = f'{self.config.n_peaks}-Modal Distribution ({kerr_str})'
        axes[0].set_title(title)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Mark target peak positions
        for pos in self.peak_positions:
            axes[0].axvline(x=pos, color='gray', linestyle=':', alpha=0.5)
        
        # Wasserstein history (smoothed)
        axes[1].plot(self.history['wasserstein'], 'g-', alpha=0.3, linewidth=1, label='Raw')
        axes[1].plot(self.history['wasserstein_smooth'], 'g-', linewidth=2, label='Smoothed')
        axes[1].axhline(y=self.best_wasserstein, color='r', linestyle='--', 
                       label=f'Best: {self.best_wasserstein:.4f}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Wasserstein Distance')
        axes[1].set_title('Training Progress')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Peak count history
        axes[2].plot(self.history['peak_count'], 'purple', linewidth=1)
        axes[2].axhline(y=self.config.n_peaks, color='r', linestyle='--', 
                       label=f'Target ({self.config.n_peaks} peaks)')
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
    
    def plot_full_dashboard(self, log_dir: str):
        """Generate comprehensive training dashboard."""
        os.makedirs(log_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        epochs = self.history['epochs']
        
        # 1. Losses
        ax = axes[0, 0]
        ax.plot(epochs, self.history['g_loss'], label='Generator', color='blue')
        ax.plot(epochs, self.history['d_loss'], label='Discriminator', color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Gradient norms
        ax = axes[0, 1]
        ax.semilogy(epochs, self.history['g_grad_norm'], label='Generator', color='blue')
        ax.semilogy(epochs, self.history['d_grad_norm'], label='Discriminator', color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm (log)')
        ax.set_title('Gradient Norms')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Wasserstein distance
        ax = axes[0, 2]
        ax.plot(epochs, self.history['wasserstein'], alpha=0.3, color='green', label='Raw')
        ax.plot(epochs, self.history['wasserstein_smooth'], color='green', linewidth=2, label='Smooth')
        ax.axhline(y=self.best_wasserstein, color='r', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('W₁ Distance')
        ax.set_title(f'Wasserstein (Best: {self.best_wasserstein:.4f} @ {self.best_epoch})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Distribution comparison (using best)
        ax = axes[1, 0]
        if self.best_weights is not None:
            original = self.generator.weights.numpy().copy()
            self.generator.weights.assign(self.best_weights)
        z = tf.random.normal([1, self.generator.latent_dim])
        P_gen = self.generator.generate(z)[0]
        if self.best_weights is not None:
            self.generator.weights.assign(original)
        
        ax.plot(self.xvec.numpy(), self.P_real.numpy(), 'b-', label='Target', linewidth=2)
        ax.plot(self.xvec.numpy(), P_gen.numpy(), 'r--', label='Generated (Best)', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('P(x)')
        ax.set_title(f'Distribution Comparison ({self.config.n_peaks}-modal)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Peak count
        ax = axes[1, 1]
        ax.plot(epochs, self.history['peak_count'], 'purple', linewidth=1)
        ax.axhline(y=self.config.n_peaks, color='r', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Peaks')
        ax.set_title(f'Peak Count (Target: {self.config.n_peaks})')
        ax.grid(True, alpha=0.3)
        
        # 6. Summary
        ax = axes[1, 2]
        ax.axis('off')
        summary_text = f"""
        Training Summary
        ================
        
        Architecture:
          Layers: {self.config.n_layers}
          Cutoff: {self.config.cutoff_dim}
          Kerr: {self.config.use_kerr}
          Params: {self.generator.num_params}
        
        Target: {self.config.n_peaks}-modal
        
        Results:
          Best W₁: {self.best_wasserstein:.4f}
          Best Epoch: {self.best_epoch}
          Final Peaks: {self.history['peak_count'][-1]}
        """
        ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Killoran CV-QNN Training Dashboard', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(log_dir, 'training_dashboard.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Dashboard saved to {save_path}")
        
        # Also save weight evolution
        self.weight_tracker.plot_weight_evolution(
            os.path.join(log_dir, 'weight_evolution.png')
        )


def main():
    parser = argparse.ArgumentParser(description='Killoran CV-QNN Training')
    
    # Architecture
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--cutoff-dim', type=int, default=8)
    parser.add_argument('--use-kerr', action='store_true', default=True)
    parser.add_argument('--no-kerr', dest='use_kerr', action='store_false')
    
    # Target - N-modal
    parser.add_argument('--n-peaks', type=int, default=2, 
                        help='Number of peaks in target distribution')
    parser.add_argument('--x-min', type=float, default=-2.0,
                        help='Left boundary for peak placement')
    parser.add_argument('--x-max', type=float, default=2.0,
                        help='Right boundary for peak placement')
    parser.add_argument('--peak-std', type=float, default=0.3,
                        help='Standard deviation of each peak')
    
    # Legacy target options
    parser.add_argument('--target-type', type=str, default='nmodal', 
                        choices=['gaussian', 'bimodal', 'nmodal'])
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
    
    # Monitoring
    parser.add_argument('--save-wigner', action='store_true', default=True)
    parser.add_argument('--no-wigner', dest='save_wigner', action='store_false')
    
    # Output
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default='./logs')
    
    args = parser.parse_args()
    
    # Build experiment name
    if args.exp_name:
        exp_name = args.exp_name
    else:
        kerr_str = "kerr" if args.use_kerr else "nokerr"
        if args.target_type == 'nmodal':
            exp_name = f"killoran_{kerr_str}_{args.n_peaks}modal_L{args.n_layers}_C{args.cutoff_dim}"
        elif args.target_type == 'bimodal':
            exp_name = f"killoran_{kerr_str}_bimodal_{args.mean1}_{args.mean2}"
        else:
            exp_name = f"killoran_{kerr_str}_gaussian"
    
    log_dir = os.path.join(args.log_dir, exp_name)
    
    # Create config
    config = KilloranTrainerConfig(
        # N-modal
        n_peaks=args.n_peaks,
        x_min=args.x_min,
        x_max=args.x_max,
        peak_std=args.peak_std,
        target_type=args.target_type,
        # Legacy
        mean1=args.mean1,
        std1=args.std1,
        mean2=args.mean2,
        std2=args.std2,
        weight1=args.weight1,
        # Architecture
        n_layers=args.n_layers,
        cutoff_dim=args.cutoff_dim,
        use_kerr=args.use_kerr,
        batch_size=args.batch_size,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        n_critic=args.n_critic,
        # Monitoring
        save_wigner=args.save_wigner
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
    
    # Generate all plots
    trainer.plot_comparison(
        save_path=os.path.join(log_dir, 'final_comparison.png'),
        show=False,
        use_best=True
    )
    
    trainer.plot_full_dashboard(log_dir)
    
    return summary


if __name__ == "__main__":
    main()
