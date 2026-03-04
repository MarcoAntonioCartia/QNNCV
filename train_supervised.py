#!/usr/bin/env python
"""
SUPERVISED CV-QNN Training (No GAN)
===================================

This script trains the quantum generator DIRECTLY by minimizing 
the distance to the target distribution, without any discriminator.

Purpose: Verify that the quantum circuit CAN learn before debugging GAN dynamics.

If this works: The quantum circuit has sufficient expressivity
If this fails: The problem is in the quantum architecture itself

Losses available:
- MSE: Mean squared error between distributions
- KL: KL divergence (requires careful handling of zeros)
- Wasserstein: Earth mover's distance approximation
- Combined: Weighted combination
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
        sys.exit(1)


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
    elif name == 'simple_gaussian':
        # Ultra-simple: centered Gaussian
        return gaussian_2d_independent(X, Y, mu=(0, 0), sigma=0.5)
    elif name == 'shifted_gaussian':
        # Shifted Gaussian - tests displacement learning
        return gaussian_2d_independent(X, Y, mu=(1.0, 0.5), sigma=0.5)
    else:
        raise ValueError(f"Unknown target: {name}")


# =============================================================================
# Loss Functions
# =============================================================================

def mse_loss(target, generated):
    """Mean squared error between distributions."""
    return tf.reduce_mean(tf.square(target - generated))


def kl_divergence(target, generated, epsilon=1e-10):
    """KL divergence: KL(target || generated)."""
    # Add small epsilon to avoid log(0)
    target_safe = tf.clip_by_value(target, epsilon, 1.0)
    generated_safe = tf.clip_by_value(generated, epsilon, 1.0)
    
    return tf.reduce_sum(target_safe * tf.math.log(target_safe / generated_safe))


def js_divergence(target, generated, epsilon=1e-10):
    """Jensen-Shannon divergence (symmetric, bounded)."""
    m = 0.5 * (target + generated)
    return 0.5 * (kl_divergence(target, m, epsilon) + kl_divergence(generated, m, epsilon))


def wasserstein_approx_loss(target, generated):
    """
    Approximate Wasserstein loss using marginal comparisons.
    Differentiable approximation for training.
    """
    # Marginals
    target_x = tf.reduce_sum(target, axis=0)
    target_y = tf.reduce_sum(target, axis=1)
    gen_x = tf.reduce_sum(generated, axis=0)
    gen_y = tf.reduce_sum(generated, axis=1)
    
    # CDF comparison (approximation to Wasserstein)
    target_cdf_x = tf.cumsum(target_x)
    target_cdf_y = tf.cumsum(target_y)
    gen_cdf_x = tf.cumsum(gen_x)
    gen_cdf_y = tf.cumsum(gen_y)
    
    # L1 distance between CDFs ≈ Wasserstein for 1D
    w_x = tf.reduce_mean(tf.abs(target_cdf_x - gen_cdf_x))
    w_y = tf.reduce_mean(tf.abs(target_cdf_y - gen_cdf_y))
    
    return w_x + w_y


def total_variation_loss(target, generated):
    """Total variation distance."""
    return 0.5 * tf.reduce_sum(tf.abs(target - generated))


def combined_loss(target, generated, mse_weight=1.0, w_weight=1.0, tv_weight=0.1):
    """Combined loss for robust training."""
    mse = mse_loss(target, generated)
    w = wasserstein_approx_loss(target, generated)
    tv = total_variation_loss(target, generated)
    
    return mse_weight * mse + w_weight * w + tv_weight * tv


# =============================================================================
# Metrics
# =============================================================================

def compute_wasserstein_2d(p, q):
    """Exact Wasserstein using scipy."""
    from scipy.stats import wasserstein_distance
    
    p = np.asarray(p)
    q = np.asarray(q)
    
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        return float('inf')
    
    p_sum = p.sum()
    q_sum = q.sum()
    
    if p_sum <= 0 or q_sum <= 0:
        return float('inf')
    
    p = p / p_sum
    q = q / q_sum
    
    p_x = np.clip(p.sum(axis=0), 1e-10, None)
    p_y = np.clip(p.sum(axis=1), 1e-10, None)
    q_x = np.clip(q.sum(axis=0), 1e-10, None)
    q_y = np.clip(q.sum(axis=1), 1e-10, None)
    
    p_x = p_x / p_x.sum()
    p_y = p_y / p_y.sum()
    q_x = q_x / q_x.sum()
    q_y = q_y / q_y.sum()
    
    try:
        w_x = wasserstein_distance(range(len(p_x)), range(len(q_x)), p_x, q_x)
        w_y = wasserstein_distance(range(len(p_y)), range(len(q_y)), p_y, q_y)
        return (w_x + w_y) / 2
    except Exception:
        return float('inf')


# =============================================================================
# Training Loop - SUPERVISED
# =============================================================================

def train_supervised(
    target_name='simple_gaussian',
    n_layers=4,
    cutoff_dim=6,
    use_kerr=True,
    epochs=500,
    lr=0.01,
    loss_type='combined',  # 'mse', 'wasserstein', 'combined'
    grid_size=40,
    x_range=3.0,
    log_dir=None,
    plot_every=50,
    use_adam=True,
    lr_decay=0.99,  # Learning rate decay per epoch
):
    """
    Train the quantum generator with DIRECT supervision.
    
    No discriminator - just minimize distance to target.
    """
    
    # Setup
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"./logs/supervised_{target_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 60)
    print("SUPERVISED CV-QNN Training (No GAN)")
    print("=" * 60)
    print(f"Target: {target_name}")
    print(f"Layers: {n_layers}, Cutoff: {cutoff_dim}, Kerr: {use_kerr}")
    print(f"Grid: {grid_size}x{grid_size}, Range: [-{x_range}, {x_range}]")
    print(f"Learning rate: {lr}, Loss: {loss_type}")
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
    
    # Generator
    print("\nInitializing generator...")
    generator = KilloranCVQNNMultiMode(
        n_modes=2,
        n_layers=n_layers,
        cutoff_dim=cutoff_dim,
        use_kerr=use_kerr
    )
    
    # Optimizer with learning rate schedule
    if use_adam:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=100,
            decay_rate=lr_decay
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    
    # History
    history = {
        'loss': [], 'wasserstein': [], 'mse': [],
        'grad_norm': [], 'lr': []
    }
    
    best_wasserstein = float('inf')
    best_loss = float('inf')
    best_weights = None
    
    # Select loss function
    if loss_type == 'mse':
        loss_fn = mse_loss
    elif loss_type == 'wasserstein':
        loss_fn = wasserstein_approx_loss
    elif loss_type == 'kl':
        loss_fn = kl_divergence
    elif loss_type == 'js':
        loss_fn = js_divergence
    elif loss_type == 'tv':
        loss_fn = total_variation_loss
    else:  # combined
        loss_fn = lambda t, g: combined_loss(t, g, mse_weight=1.0, w_weight=2.0, tv_weight=0.1)
    
    # Initial state
    print("\nGenerating initial distribution...")
    try:
        init_prob = generator.generate_distribution_2d(xvec, yvec).numpy()
        init_w = compute_wasserstein_2d(target, init_prob)
        print(f"Initial Wasserstein: {init_w:.4f}")
        plot_comparison(target, init_prob, X, Y, 
                       f"{log_dir}/epoch_000_comparison.png", 
                       title=f"Epoch 0 (Initial, W1={init_w:.4f})")
    except Exception as e:
        print(f"FAIL: Initial generation failed: {e}")
        return None
    
    # Training loop
    print("\n" + "-" * 60)
    print("Training (SUPERVISED - direct loss minimization)...")
    print("-" * 60)
    
    for epoch in range(1, epochs + 1):
        
        with tf.GradientTape() as tape:
            # Generate distribution
            gen_prob = generator.generate_distribution_2d(xvec, yvec)
            
            # Compute loss
            loss = loss_fn(target_tf, gen_prob)
        
        # Compute gradients
        grads = tape.gradient(loss, generator.trainable_variables)
        
        if grads[0] is None:
            print(f"  Warning: Gradients are None at epoch {epoch}")
            continue
        
        # Apply gradients
        optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        
        grad_norm = float(tf.norm(grads[0]))
        
        # Metrics
        gen_np = gen_prob.numpy()
        
        # Check for degenerate
        if not np.isfinite(gen_np).all() or gen_np.sum() <= 1e-10:
            print(f"  Warning: Degenerate distribution at epoch {epoch}")
            continue
        
        wasserstein = compute_wasserstein_2d(target, gen_np)
        mse = float(tf.reduce_mean(tf.square(target_tf - gen_prob)))
        current_lr = float(optimizer.learning_rate(epoch) if hasattr(optimizer.learning_rate, '__call__') else lr)
        
        # Track history
        history['loss'].append(float(loss))
        history['wasserstein'].append(wasserstein)
        history['mse'].append(mse)
        history['grad_norm'].append(grad_norm)
        history['lr'].append(current_lr)
        
        # Best checkpoint
        if wasserstein < best_wasserstein:
            best_wasserstein = wasserstein
            best_weights = generator.weights.numpy().copy()
        
        if float(loss) < best_loss:
            best_loss = float(loss)
        
        # Logging
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | Loss: {float(loss):.6f} | W1: {wasserstein:.4f} | "
                  f"MSE: {mse:.6f} | Grad: {grad_norm:.4f} | Best W1: {best_wasserstein:.4f}")
        
        # Visualization
        if epoch % plot_every == 0:
            plot_comparison(target, gen_np, X, Y,
                          f"{log_dir}/epoch_{epoch:03d}_comparison.png",
                          title=f"Epoch {epoch} (W1={wasserstein:.4f})")
    
    # Final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Wasserstein: {best_wasserstein:.4f}")
    print(f"Best Loss: {best_loss:.6f}")
    
    # Restore best weights
    if best_weights is not None:
        generator.weights.assign(best_weights)
    
    # Final plots
    final_prob = generator.generate_distribution_2d(xvec, yvec).numpy()
    final_w = compute_wasserstein_2d(target, final_prob)
    
    plot_comparison(target, final_prob, X, Y,
                   f"{log_dir}/final_comparison.png",
                   title=f"Final (Best W1={best_wasserstein:.4f})")
    
    plot_training_history_supervised(history, f"{log_dir}/training_history.png")
    
    # Save history
    np.savez(f"{log_dir}/history.npz", **history)
    
    print(f"\nResults saved to: {log_dir}")
    
    # Summary
    print("\n" + "=" * 60)
    print("LEARNING VERIFICATION:")
    print("=" * 60)
    if best_wasserstein < init_w * 0.5:
        print("✓ SIGNIFICANT LEARNING: W1 reduced by >50%")
        print(f"  Initial W1: {init_w:.4f} → Best W1: {best_wasserstein:.4f}")
    elif best_wasserstein < init_w * 0.8:
        print("~ SOME LEARNING: W1 reduced by 20-50%")
        print(f"  Initial W1: {init_w:.4f} → Best W1: {best_wasserstein:.4f}")
    else:
        print("✗ NO SIGNIFICANT LEARNING: W1 did not improve much")
        print(f"  Initial W1: {init_w:.4f} → Best W1: {best_wasserstein:.4f}")
        print("\n  Possible issues:")
        print("  - Quantum circuit lacks expressivity (try more layers)")
        print("  - Gradient flow problem (check grad_norm)")
        print("  - Learning rate too high/low")
        print("  - Cutoff dimension too small for target")
    
    return generator, history


# =============================================================================
# Visualization
# =============================================================================

def plot_comparison(target, generated, X, Y, save_path, title=""):
    """Plot target vs generated 2D distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    im0 = axes[0].contourf(X, Y, target, levels=50, cmap='viridis')
    axes[0].set_title('Target Distribution')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].contourf(X, Y, generated, levels=50, cmap='viridis')
    axes[1].set_title('Generated Distribution')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1])
    
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


def plot_training_history_supervised(history, save_path):
    """Plot training metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['loss'], 'b-', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Wasserstein
    axes[0, 1].plot(epochs, history['wasserstein'], 'g-', alpha=0.7)
    axes[0, 1].axhline(y=min(history['wasserstein']), color='r', linestyle='--',
                       label=f'Best: {min(history["wasserstein"]):.4f}')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Wasserstein Distance')
    axes[0, 1].set_title('Distribution Quality')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gradient norm
    axes[1, 0].plot(epochs, history['grad_norm'], 'purple', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].set_title('Gradient Magnitude')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # MSE
    axes[1, 1].plot(epochs, history['mse'], 'orange', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MSE')
    axes[1, 1].set_title('Mean Squared Error')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SUPERVISED CV-QNN Training')
    parser.add_argument('--target', type=str, default='simple_gaussian',
                       choices=['independent', 'correlated', 'ring', 'four',
                               'simple_gaussian', 'shifted_gaussian'],
                       help='Target distribution')
    parser.add_argument('--n-layers', type=int, default=4,
                       help='Number of CV-QNN layers')
    parser.add_argument('--cutoff-dim', type=int, default=6,
                       help='Fock space cutoff')
    parser.add_argument('--no-kerr', action='store_true',
                       help='Disable Kerr gates')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--loss', type=str, default='combined',
                       choices=['mse', 'wasserstein', 'kl', 'js', 'tv', 'combined'],
                       help='Loss function')
    parser.add_argument('--grid-size', type=int, default=40,
                       help='Grid resolution')
    parser.add_argument('--plot-every', type=int, default=50,
                       help='Plot frequency')
    
    args = parser.parse_args()
    
    train_supervised(
        target_name=args.target,
        n_layers=args.n_layers,
        cutoff_dim=args.cutoff_dim,
        use_kerr=not args.no_kerr,
        epochs=args.epochs,
        lr=args.lr,
        loss_type=args.loss,
        grid_size=args.grid_size,
        plot_every=args.plot_every
    )


if __name__ == "__main__":
    main()
