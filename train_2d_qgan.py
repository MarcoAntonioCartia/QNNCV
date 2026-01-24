#!/usr/bin/env python
"""
Train 2D CV-QNN GAN
===================

Complete training script for 2-mode Killoran CV-QNN on 2D distributions.

Usage:
    python train_2d_qgan.py
    python train_2d_qgan.py --target correlated --epochs 200
    python train_2d_qgan.py --target ring --n-layers 8

Targets available:
    - independent: P(x,y) = P(x)*P(y) - should work without entanglement
    - correlated: Bivariate Gaussian with rho=0.7 - REQUIRES entanglement
    - ring: Ring distribution - tests non-Gaussian (needs Kerr)
    - four: Four Gaussians at corners - multi-modal 2D
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Compatibility patches
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

# Import the multi-mode generator
# Try multiple import paths
try:
    from killoran_cvqnn_multimode import (
        KilloranCVQNNMultiMode,
        gaussian_2d_independent,
        gaussian_2d_correlated,
        ring_2d,
        four_gaussians_2d
    )
except ImportError:
    try:
        from src.models.generators.killoran_cvqnn_multimode import (
            KilloranCVQNNMultiMode,
            gaussian_2d_independent,
            gaussian_2d_correlated,
            ring_2d,
            four_gaussians_2d
        )
    except ImportError:
        print("ERROR: Could not import killoran_cvqnn_multimode.py")
        print("Make sure the file is in the same directory as this script")
        print("or in src/models/generators/")
        sys.exit(1)


# =============================================================================
# 2D Discriminator
# =============================================================================

class Discriminator2D(tf.keras.Model):
    """
    Discriminator for 2D probability distributions.
    
    Takes a flattened probability grid as input.
    Uses WGAN formulation (no sigmoid, outputs critic score).
    
    Balanced size - strong enough to guide, not so strong it dominates.
    """
    
    def __init__(self, grid_size=50, hidden_dims=[128, 64]):
        super().__init__()
        
        self.flatten = tf.keras.layers.Flatten()
        
        layers = []
        for dim in hidden_dims:
            layers.append(tf.keras.layers.Dense(dim))
            layers.append(tf.keras.layers.LeakyReLU(0.2))
        
        layers.append(tf.keras.layers.Dense(1))  # No activation for WGAN
        
        self.net = tf.keras.Sequential(layers)
    
    def call(self, x, training=False):
        x = self.flatten(x)
        return self.net(x, training=training)


# =============================================================================
# Target Distributions
# =============================================================================

def get_target_distribution(name, X, Y):
    """Get target distribution by name."""
    if name == 'independent':
        return gaussian_2d_independent(X, Y, mu=(0, 0), sigma=0.6)
    elif name == 'correlated':
        return gaussian_2d_correlated(X, Y, mu=(0, 0), cov=[[1, 0.7], [0.7, 1]])
    elif name == 'ring':
        return ring_2d(X, Y, radius=1.2, width=0.25)
    elif name == 'four':
        return four_gaussians_2d(X, Y, spread=1.2, sigma=0.25)
    else:
        raise ValueError(f"Unknown target: {name}")


# =============================================================================
# Training Loop
# =============================================================================

def train_2d_qgan(
    target_name='independent',
    n_layers=4,
    cutoff_dim=6,
    use_kerr=True,
    epochs=100,
    g_lr=0.005,
    d_lr=0.0005,  # Balanced: not too fast, not too slow
    n_critic=3,   # Train G 3 times per D update
    grid_size=40,
    x_range=3.0,
    log_dir=None,
    plot_every=20
):
    """
    Train 2D CV-QNN GAN.
    
    Args:
        target_name: 'independent', 'correlated', 'ring', 'four'
        n_layers: Number of CV-QNN layers
        cutoff_dim: Fock space cutoff
        use_kerr: Include Kerr gates
        epochs: Training epochs
        g_lr, d_lr: Learning rates
        n_critic: Update discriminator every n_critic generator steps
        grid_size: Resolution of probability grid
        x_range: Grid spans [-x_range, x_range]
        log_dir: Where to save outputs
        plot_every: Visualization frequency
    """
    
    # Setup
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"./logs/2d_qgan_{target_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 60)
    print("2D CV-QNN GAN Training")
    print("=" * 60)
    print(f"Target: {target_name}")
    print(f"Layers: {n_layers}, Cutoff: {cutoff_dim}, Kerr: {use_kerr}")
    print(f"Grid: {grid_size}x{grid_size}, Range: [-{x_range}, {x_range}]")
    print(f"D update every {n_critic} G steps, D_lr: {d_lr}, G_lr: {g_lr}")
    print(f"Output: {log_dir}")
    print("=" * 60)
    
    # Create grids
    xvec = np.linspace(-x_range, x_range, grid_size)
    yvec = np.linspace(-x_range, x_range, grid_size)
    X, Y = np.meshgrid(xvec, yvec)
    
    # Target distribution
    target = get_target_distribution(target_name, X, Y)
    target_tf = tf.constant(target, dtype=tf.float32)
    
    print(f"\nTarget distribution: shape={target.shape}, sum={target.sum():.4f}")
    
    # Models
    print("\nInitializing generator...")
    generator = KilloranCVQNNMultiMode(
        n_modes=2,
        n_layers=n_layers,
        cutoff_dim=cutoff_dim,
        use_kerr=use_kerr
    )
    
    print("\nInitializing discriminator...")
    discriminator = Discriminator2D(grid_size=grid_size)
    
    # Optimizers
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=g_lr)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=d_lr)
    
    # History
    history = {
        'g_loss': [], 'd_loss': [], 'wasserstein': [],
        'g_grad_norm': [], 'd_grad_norm': []
    }
    
    best_wasserstein = float('inf')
    best_weights = None
    
    # Plot initial state
    print("\nGenerating initial distribution...")
    try:
        init_prob = generator.generate_distribution_2d(xvec, yvec).numpy()
        plot_comparison(target, init_prob, X, Y, 
                       f"{log_dir}/epoch_000_comparison.png", 
                       title="Epoch 0 (Initial)")
        print("OK: Initial plot saved")
    except Exception as e:
        print(f"FAIL: Initial generation failed: {e}")
        return None
    
    # Training loop
    print("\n" + "-" * 60)
    print("Training...")
    print("-" * 60)
    
    for epoch in range(1, epochs + 1):
        # === Discriminator Step (only every n_critic epochs) ===
        if epoch % n_critic == 0:
            with tf.GradientTape() as tape:
                # Generated distribution
                gen_prob = generator.generate_distribution_2d(xvec, yvec)
                
                # Discriminator scores
                real_score = discriminator(tf.expand_dims(target_tf, 0), training=True)
                fake_score = discriminator(tf.expand_dims(gen_prob, 0), training=True)
                
                # WGAN loss: maximize E[D(real)] - E[D(fake)]
                # So minimize -E[D(real)] + E[D(fake)]
                d_loss = -tf.reduce_mean(real_score) + tf.reduce_mean(fake_score)
            
            d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
            
            # Clip discriminator weights (WGAN weight clipping) - less aggressive
            for var in discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -0.1, 0.1))
            
            d_loss_val = float(d_loss)
        else:
            d_loss_val = history['d_loss'][-1] if history['d_loss'] else 0.0
        
        # === Generator Step ===
        with tf.GradientTape() as tape:
            gen_prob = generator.generate_distribution_2d(xvec, yvec)
            fake_score = discriminator(tf.expand_dims(gen_prob, 0), training=False)
            
            # Generator wants to maximize D(fake), so minimize -D(fake)
            g_loss = -tf.reduce_mean(fake_score)
        
        g_grads = tape.gradient(g_loss, generator.trainable_variables)
        
        if g_grads[0] is not None:
            g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
            g_grad_norm = tf.norm(g_grads[0]).numpy()
        else:
            g_grad_norm = 0.0
            print(f"  Warning: Generator gradients are None at epoch {epoch}")
        
        # === Metrics ===
        gen_np = gen_prob.numpy()
        
        # Check for degenerate distribution
        if not np.isfinite(gen_np).all() or gen_np.sum() <= 0:
            print(f"  Warning: Degenerate distribution at epoch {epoch}, skipping metrics")
            history['g_loss'].append(float('nan'))
            history['d_loss'].append(float('nan'))
            history['wasserstein'].append(float('inf'))
            history['g_grad_norm'].append(0.0)
            history['d_grad_norm'].append(0.0)
            continue
        
        wasserstein = compute_wasserstein_2d(target, gen_np)
        
        # Track history
        history['g_loss'].append(float(g_loss))
        history['d_loss'].append(d_loss_val)
        history['wasserstein'].append(wasserstein)
        history['g_grad_norm'].append(g_grad_norm)
        history['d_grad_norm'].append(0.0)  # Simplified tracking
        
        # Best checkpoint
        if wasserstein < best_wasserstein:
            best_wasserstein = wasserstein
            best_weights = generator.weights.numpy().copy()
        
        # === Logging ===
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | G_loss: {g_loss:.4f} | D_loss: {d_loss_val:.4f} | "
                  f"W1: {wasserstein:.4f} | Best: {best_wasserstein:.4f}")
        
        # === Visualization ===
        if epoch % plot_every == 0:
            plot_comparison(target, gen_np, X, Y,
                          f"{log_dir}/epoch_{epoch:03d}_comparison.png",
                          title=f"Epoch {epoch}")
    
    # Final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Wasserstein: {best_wasserstein:.4f}")
    
    # Restore best weights
    if best_weights is not None:
        generator.weights.assign(best_weights)
    
    # Final plots
    final_prob = generator.generate_distribution_2d(xvec, yvec).numpy()
    
    plot_comparison(target, final_prob, X, Y,
                   f"{log_dir}/final_comparison.png",
                   title=f"Final (Best W1={best_wasserstein:.4f})")
    
    plot_training_history(history, f"{log_dir}/training_history.png")
    
    # Save history
    np.savez(f"{log_dir}/history.npz", **history)
    
    print(f"\nResults saved to: {log_dir}")
    
    return generator, history


# =============================================================================
# Metrics
# =============================================================================

def compute_wasserstein_2d(p, q):
    """
    Approximate 2D Wasserstein distance.
    
    Uses sum of 1D Wasserstein along each axis (upper bound).
    Handles degenerate distributions gracefully.
    """
    from scipy.stats import wasserstein_distance
    
    # Convert to numpy if needed
    p = np.asarray(p)
    q = np.asarray(q)
    
    # Check for NaN/Inf
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        return float('inf')
    
    # Normalize (with safety)
    p_sum = p.sum()
    q_sum = q.sum()
    
    if p_sum <= 0 or q_sum <= 0:
        return float('inf')
    
    p = p / p_sum
    q = q / q_sum
    
    # Marginals
    p_x = p.sum(axis=0)
    p_y = p.sum(axis=1)
    q_x = q.sum(axis=0)
    q_y = q.sum(axis=1)
    
    # Check marginals are valid
    if p_x.sum() <= 0 or q_x.sum() <= 0 or p_y.sum() <= 0 or q_y.sum() <= 0:
        return float('inf')
    
    # Ensure positive (clip tiny negatives from numerical error)
    p_x = np.clip(p_x, 1e-10, None)
    p_y = np.clip(p_y, 1e-10, None)
    q_x = np.clip(q_x, 1e-10, None)
    q_y = np.clip(q_y, 1e-10, None)
    
    # Re-normalize after clipping
    p_x = p_x / p_x.sum()
    p_y = p_y / p_y.sum()
    q_x = q_x / q_x.sum()
    q_y = q_y / q_y.sum()
    
    try:
        # 1D Wasserstein on marginals
        w_x = wasserstein_distance(range(len(p_x)), range(len(q_x)), p_x, q_x)
        w_y = wasserstein_distance(range(len(p_y)), range(len(q_y)), p_y, q_y)
        return (w_x + w_y) / 2
    except Exception:
        return float('inf')


# =============================================================================
# Visualization
# =============================================================================

def plot_comparison(target, generated, X, Y, save_path, title=""):
    """Plot target vs generated 2D distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Target
    im0 = axes[0].contourf(X, Y, target, levels=50, cmap='viridis')
    axes[0].set_title('Target Distribution')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0])
    
    # Generated
    im1 = axes[1].contourf(X, Y, generated, levels=50, cmap='viridis')
    axes[1].set_title('Generated Distribution')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1])
    
    # Difference
    diff = np.abs(target - generated)
    im2 = axes[2].contourf(X, Y, diff, levels=50, cmap='Reds')
    axes[2].set_title(f'|Difference| (sum={diff.sum():.4f})')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[2])
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_history(history, save_path):
    """Plot training metrics over time."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history['g_loss']) + 1)
    
    # Losses
    axes[0, 0].plot(epochs, history['g_loss'], label='G Loss', alpha=0.7)
    axes[0, 0].plot(epochs, history['d_loss'], label='D Loss', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('GAN Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Wasserstein
    axes[0, 1].plot(epochs, history['wasserstein'], 'g-', alpha=0.7)
    axes[0, 1].axhline(y=min(history['wasserstein']), color='r', linestyle='--', 
                       label=f'Best: {min(history["wasserstein"]):.4f}')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Wasserstein Distance')
    axes[0, 1].set_title('Distribution Distance')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gradient norms
    axes[1, 0].plot(epochs, history['g_grad_norm'], label='Generator', alpha=0.7)
    axes[1, 0].plot(epochs, history['d_grad_norm'], label='Discriminator', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].set_title('Gradient Norms')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Wasserstein smoothed
    window = min(10, len(history['wasserstein']) // 5)
    if window > 1:
        smoothed = np.convolve(history['wasserstein'], 
                               np.ones(window)/window, mode='valid')
        axes[1, 1].plot(range(window, len(history['wasserstein']) + 1), 
                        smoothed, 'b-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Wasserstein (Smoothed)')
    axes[1, 1].set_title('Convergence Trend')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train 2D CV-QNN GAN')
    parser.add_argument('--target', type=str, default='independent',
                       choices=['independent', 'correlated', 'ring', 'four'],
                       help='Target distribution')
    parser.add_argument('--n-layers', type=int, default=4,
                       help='Number of CV-QNN layers')
    parser.add_argument('--cutoff-dim', type=int, default=6,
                       help='Fock space cutoff')
    parser.add_argument('--no-kerr', action='store_true',
                       help='Disable Kerr gates')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--g-lr', type=float, default=0.005,
                       help='Generator learning rate')
    parser.add_argument('--d-lr', type=float, default=0.0005,
                       help='Discriminator learning rate (default: 0.0005)')
    parser.add_argument('--n-critic', type=int, default=3,
                       help='Update D every n_critic G steps (default: 3)')
    parser.add_argument('--grid-size', type=int, default=40,
                       help='Grid resolution')
    parser.add_argument('--plot-every', type=int, default=20,
                       help='Plot frequency')
    
    args = parser.parse_args()
    
    train_2d_qgan(
        target_name=args.target,
        n_layers=args.n_layers,
        cutoff_dim=args.cutoff_dim,
        use_kerr=not args.no_kerr,
        epochs=args.epochs,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        n_critic=args.n_critic,
        grid_size=args.grid_size,
        plot_every=args.plot_every
    )


if __name__ == "__main__":
    main()
