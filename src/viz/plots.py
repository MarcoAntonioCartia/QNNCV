"""
Visualization
=============

Extracted 1:1 from train_2d_qgan.py (behavior-preserving refactor).
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_comparison(target, generated, X, Y, save_path, title=""):
    """Plot target vs generated."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].contourf(X, Y, target, levels=30, cmap='viridis')
    axes[0].set_title('Target')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(X, Y, generated, levels=30, cmap='viridis')
    axes[1].set_title('Generated')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1])

    diff = np.abs(target - generated)
    im2 = axes[2].contourf(X, Y, diff, levels=30, cmap='Reds')
    axes[2].set_title(f'|Difference|')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_history(history, save_path):
    """Plot training history."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    epochs = range(1, len(history['g_loss']) + 1)

    # Losses
    axes[0, 0].plot(epochs, history['g_loss'], label='G Total', alpha=0.7)
    axes[0, 0].plot(epochs, history['d_loss'], label='D Loss', alpha=0.7)
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('GAN Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Wasserstein
    w = history['wasserstein']
    valid_w = [x for x in w if x < float('inf')]
    if valid_w:
        axes[0, 1].plot(epochs[:len(valid_w)], valid_w, 'g-', alpha=0.5, label='Train')

    # Validation Wasserstein
    if history['val_wasserstein']:
        val_w = history['val_wasserstein']
        valid_val_w = [x for x in val_w if x < float('inf')]
        if valid_val_w:
            axes[0, 1].plot(range(1, len(valid_val_w) + 1), valid_val_w, 'b-',
                           linewidth=2, marker='o', label='Val (canonical)')
            axes[0, 1].axhline(y=min(valid_val_w), color='r', linestyle='--',
                              label=f'Best Val: {min(valid_val_w):.4f}')

    val_near = [x for x in history.get('val_nearest_w1', []) if x < float('inf')]
    if val_near:
        axes[0, 1].plot(range(1, len(val_near) + 1), val_near, 'm-',
                       linewidth=2, marker='s', label='Val (nearest member)')

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Wasserstein Distance')
    axes[0, 1].set_title('Distribution Quality')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Gradient norms
    axes[0, 2].plot(epochs, history['g_grad_norm'], label='Generator', alpha=0.7)
    axes[0, 2].plot(epochs, history['d_grad_norm'], label='Discriminator', alpha=0.7)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Gradient Norm')
    axes[0, 2].set_title('Gradient Norms')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # GP Weight Warmup (or Loss Components if no GP)
    if 'gp_weight_current' in history and history['gp_weight_current']:
        axes[1, 0].plot(epochs, history['gp_weight_current'], 'purple', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('GP Weight')
        axes[1, 0].set_title('Gradient Penalty Weight (warmup)')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].plot(epochs, history['adversarial_loss'], label='Adversarial', alpha=0.7)
        axes[1, 0].plot(epochs, history['supervised_loss'], label='Supervised', alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Loss Components')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Smoothed Wasserstein
    if valid_w:
        window = min(20, len(valid_w) // 5)
        if window > 1:
            smoothed = np.convolve(valid_w, np.ones(window)/window, mode='valid')
            axes[1, 1].plot(range(window, len(valid_w) + 1), smoothed, 'b-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Wasserstein (Smoothed)')
    axes[1, 1].set_title('Convergence Trend')
    axes[1, 1].grid(True, alpha=0.3)

    # GP Value (or G/D balance if no GP)
    if 'gp_value' in history and history['gp_value']:
        axes[1, 2].plot(epochs, history['gp_value'], 'orange', alpha=0.7)
        axes[1, 2].axhline(y=1, color='gray', linestyle='--', label='Target (1.0)')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Gradient Penalty')
        axes[1, 2].set_title('GP Value (should stabilize ~1)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        g_power = np.abs(history['g_grad_norm'])
        d_power = np.abs(history['d_grad_norm'])
        ratio = np.array(g_power) / (np.array(d_power) + 1e-8)
        axes[1, 2].plot(epochs, ratio, 'orange', alpha=0.7)
        axes[1, 2].axhline(y=1, color='gray', linestyle='--')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('G/D Gradient Ratio')
        axes[1, 2].set_title('Training Balance')
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
