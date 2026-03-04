"""
Visualization Utilities for CV Quantum GAN
==========================================

Contains all plotting functions for:
- Wigner function visualization (quantum state)
- Distribution comparisons (generated vs real)
- Training progress plots
- Latent space analysis

IMPORTANT: All functions here should be called OUTSIDE the gradient tape.
These are for monitoring only, not part of the computational graph.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, List
import tensorflow as tf


# =============================================================================
# QUANTUM STATE VISUALIZATION
# =============================================================================

def plot_wigner_3d(
    state,
    mode: int = 0,
    x_range: Tuple[float, float] = (-5, 5),
    p_range: Tuple[float, float] = (-5, 5),
    resolution: int = 100,
    title: str = "Wigner Function",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'RdBu',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot 3D Wigner function for a single mode.
    
    Args:
        state: Strawberry Fields quantum state object
        mode: Which mode to visualize (default 0)
        x_range: Range for x-quadrature
        p_range: Range for p-quadrature  
        resolution: Grid resolution
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        save_path: If provided, save figure to this path
        
    Returns:
        matplotlib Figure object
    """
    xvec = np.linspace(x_range[0], x_range[1], resolution)
    pvec = np.linspace(p_range[0], p_range[1], resolution)
    
    # Get Wigner function from SF state
    W = state.wigner(mode, xvec, pvec)
    
    # Create meshgrid
    X, P = np.meshgrid(xvec, pvec)
    
    # 3D surface plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, P, W, cmap=cmap, alpha=0.8, 
                           linewidth=0, antialiased=True)
    
    ax.set_xlabel('x-quadrature', fontsize=12)
    ax.set_ylabel('p-quadrature', fontsize=12)
    ax.set_zlabel('W(x,p)', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='W(x,p)')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_wigner_2d(
    state,
    mode: int = 0,
    x_range: Tuple[float, float] = (-5, 5),
    p_range: Tuple[float, float] = (-5, 5),
    resolution: int = 100,
    title: str = "Wigner Function",
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'RdBu',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot 2D contour Wigner function (top-down view).
    
    Useful for quick visualization and comparing multiple states.
    """
    xvec = np.linspace(x_range[0], x_range[1], resolution)
    pvec = np.linspace(p_range[0], p_range[1], resolution)
    
    W = state.wigner(mode, xvec, pvec)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Contour plot
    levels = np.linspace(W.min(), W.max(), 50)
    contour = ax.contourf(xvec, pvec, W, levels=levels, cmap=cmap)
    
    ax.set_xlabel('x (position)', fontsize=12)
    ax.set_ylabel('p (momentum)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    
    fig.colorbar(contour, ax=ax, label='W(x,p)')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_wigner_comparison(
    states: List,
    titles: List[str],
    mode: int = 0,
    x_range: Tuple[float, float] = (-5, 5),
    resolution: int = 80,
    figsize: Tuple[int, int] = (15, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare Wigner functions of multiple quantum states side by side.
    
    Useful for showing evolution during training.
    """
    n_states = len(states)
    fig, axes = plt.subplots(1, n_states, figsize=figsize)
    
    if n_states == 1:
        axes = [axes]
    
    xvec = np.linspace(x_range[0], x_range[1], resolution)
    pvec = np.linspace(x_range[0], x_range[1], resolution)
    
    # Find global min/max for consistent colorbar
    all_W = [state.wigner(mode, xvec, pvec) for state in states]
    vmin = min(W.min() for W in all_W)
    vmax = max(W.max() for W in all_W)
    
    for ax, W, title in zip(axes, all_W, titles):
        contour = ax.contourf(xvec, pvec, W, levels=50, cmap='RdBu',
                              vmin=vmin, vmax=vmax)
        ax.set_xlabel('x')
        ax.set_ylabel('p')
        ax.set_title(title)
        ax.set_aspect('equal')
    
    fig.colorbar(contour, ax=axes, label='W(x,p)', shrink=0.8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# DISTRIBUTION VISUALIZATION
# =============================================================================

def plot_distribution_comparison(
    generated_samples: np.ndarray,
    real_samples: np.ndarray,
    epoch: Optional[int] = None,
    target_mean: Optional[float] = None,
    target_std: Optional[float] = None,
    figsize: Tuple[int, int] = (15, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare generated vs real distributions with multiple views.
    
    Args:
        generated_samples: Samples from generator (N,) or (N, 1)
        real_samples: Samples from target distribution
        epoch: Current training epoch (for title)
        target_mean: True mean of target distribution
        target_std: True std of target distribution
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        matplotlib Figure
    """
    # Flatten arrays
    gen = np.array(generated_samples).flatten()
    real = np.array(real_samples).flatten()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Histogram comparison
    ax = axes[0]
    bins = np.linspace(min(gen.min(), real.min()) - 0.5,
                       max(gen.max(), real.max()) + 0.5, 40)
    ax.hist(real, bins=bins, alpha=0.5, label='Real', density=True, color='blue')
    ax.hist(gen, bins=bins, alpha=0.5, label='Generated', density=True, color='orange')
    
    if target_mean is not None and target_std is not None:
        # Overlay target Gaussian
        x = np.linspace(bins[0], bins[-1], 200)
        from scipy.stats import norm
        ax.plot(x, norm.pdf(x, target_mean, target_std), 'r--', 
                label=f'Target N({target_mean},{target_std}²)', linewidth=2)
    
    ax.legend()
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    title = 'Distribution Comparison'
    if epoch is not None:
        title += f' (Epoch {epoch})'
    ax.set_title(title)
    
    # 2. Q-Q Plot
    ax = axes[1]
    from scipy import stats
    stats.probplot(gen, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Generated vs Normal)')
    ax.get_lines()[0].set_markerfacecolor('orange')
    ax.get_lines()[0].set_markeredgecolor('orange')
    
    # 3. KDE comparison
    ax = axes[2]
    from scipy.stats import gaussian_kde
    
    x_range = np.linspace(min(gen.min(), real.min()) - 1,
                          max(gen.max(), real.max()) + 1, 200)
    
    if len(np.unique(real)) > 1:  # KDE needs variance
        real_kde = gaussian_kde(real)
        ax.plot(x_range, real_kde(x_range), label='Real KDE', 
                color='blue', linewidth=2)
    
    if len(np.unique(gen)) > 1:  # KDE needs variance
        gen_kde = gaussian_kde(gen)
        ax.plot(x_range, gen_kde(x_range), label='Generated KDE',
                color='orange', linewidth=2)
    
    ax.legend()
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('KDE Comparison')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_latent_output_mapping(
    latent_samples: np.ndarray,
    outputs: np.ndarray,
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize how latent inputs map to generator outputs.
    
    Useful for understanding what the generator learned.
    """
    latent = np.array(latent_samples)
    out = np.array(outputs).flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # If latent is 2D (displacement encoding), plot first two dims
    if latent.ndim == 2 and latent.shape[1] >= 2:
        # Scatter: z[0] vs output
        ax = axes[0]
        scatter = ax.scatter(latent[:, 0], out, c=latent[:, 1], 
                            cmap='viridis', alpha=0.6)
        ax.set_xlabel('Latent z[0] (displacement r)')
        ax.set_ylabel('Generator Output')
        ax.set_title('z[0] → Output')
        plt.colorbar(scatter, ax=ax, label='z[1] (displacement φ)')
        
        # Scatter: z[1] vs output
        ax = axes[1]
        scatter = ax.scatter(latent[:, 1], out, c=latent[:, 0],
                            cmap='viridis', alpha=0.6)
        ax.set_xlabel('Latent z[1] (displacement φ)')
        ax.set_ylabel('Generator Output')
        ax.set_title('z[1] → Output')
        plt.colorbar(scatter, ax=ax, label='z[0] (displacement r)')
    else:
        # 1D latent
        ax = axes[0]
        ax.scatter(latent.flatten(), out, alpha=0.6)
        ax.set_xlabel('Latent z')
        ax.set_ylabel('Generator Output')
        ax.set_title('Latent → Output Mapping')
        
        # Histogram of outputs
        ax = axes[1]
        ax.hist(out, bins=30, density=True, alpha=0.7)
        ax.set_xlabel('Output Value')
        ax.set_ylabel('Density')
        ax.set_title('Output Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# TRAINING PROGRESS VISUALIZATION
# =============================================================================

def plot_training_curves(
    g_losses: List[float],
    d_losses: List[float],
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot generator and discriminator loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(g_losses) + 1)
    
    # Combined plot
    ax = axes[0]
    ax.plot(epochs, g_losses, label='Generator', color='blue')
    ax.plot(epochs, d_losses, label='Discriminator', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Separate with log scale
    ax = axes[1]
    ax.semilogy(epochs, np.abs(g_losses), label='|G Loss|', color='blue')
    ax.semilogy(epochs, np.abs(d_losses), label='|D Loss|', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('|Loss| (log)')
    ax.set_title('Training Losses (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_gradient_norms(
    g_grad_norms: List[float],
    d_grad_norms: List[float],
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot gradient norm evolution to detect vanishing/exploding gradients."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(g_grad_norms) + 1)
    
    # Linear scale
    ax = axes[0]
    ax.plot(epochs, g_grad_norms, label='Generator', color='blue')
    ax.plot(epochs, d_grad_norms, label='Discriminator', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log scale
    ax = axes[1]
    ax.semilogy(epochs, g_grad_norms, label='Generator', color='blue')
    ax.semilogy(epochs, d_grad_norms, label='Discriminator', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm (log)')
    ax.set_title('Gradient Norms (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_generation_statistics(
    gen_means: List[float],
    gen_stds: List[float],
    target_mean: float,
    target_std: float,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot how generated statistics evolve toward target."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(gen_means) + 1)
    
    # Mean
    ax = axes[0]
    ax.plot(epochs, gen_means, label='Generated Mean', color='blue', linewidth=2)
    ax.axhline(y=target_mean, color='red', linestyle='--', 
               label=f'Target Mean ({target_mean})', linewidth=2)
    ax.fill_between(epochs, target_mean - 0.3, target_mean + 0.3, 
                    alpha=0.2, color='red', label='±0.3 tolerance')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean')
    ax.set_title('Generated Mean Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Std
    ax = axes[1]
    ax.plot(epochs, gen_stds, label='Generated Std', color='blue', linewidth=2)
    ax.axhline(y=target_std, color='red', linestyle='--',
               label=f'Target Std ({target_std})', linewidth=2)
    ax.fill_between(epochs, target_std - 0.2, target_std + 0.2,
                    alpha=0.2, color='red', label='±0.2 tolerance')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Generated Std Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_wasserstein_distance(
    wasserstein_distances: List[float],
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot Wasserstein distance evolution (should decrease during training)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(wasserstein_distances) + 1)
    
    # Linear
    ax = axes[0]
    ax.plot(epochs, wasserstein_distances, color='purple', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Wasserstein Distance')
    ax.set_title('W₁ Distance (Linear)')
    ax.grid(True, alpha=0.3)
    
    # Log
    ax = axes[1]
    ax.semilogy(epochs, wasserstein_distances, color='purple', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Wasserstein Distance (log)')
    ax.set_title('W₁ Distance (Log Scale)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# COMPREHENSIVE DASHBOARD
# =============================================================================

def plot_training_dashboard(
    g_losses: List[float],
    d_losses: List[float],
    g_grad_norms: List[float],
    d_grad_norms: List[float],
    gen_means: List[float],
    gen_stds: List[float],
    wasserstein_distances: List[float],
    target_mean: float,
    target_std: float,
    epoch: int,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Comprehensive training dashboard showing all key metrics.
    
    Call this every N epochs for full overview.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    epochs = range(1, len(g_losses) + 1)
    
    # 1. Losses
    ax = axes[0, 0]
    ax.plot(epochs, g_losses, label='Generator', color='blue')
    ax.plot(epochs, d_losses, label='Discriminator', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Gradient norms
    ax = axes[0, 1]
    ax.semilogy(epochs, g_grad_norms, label='Generator', color='blue')
    ax.semilogy(epochs, d_grad_norms, label='Discriminator', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm (log)')
    ax.set_title('Gradient Norms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Generated mean
    ax = axes[0, 2]
    ax.plot(epochs, gen_means, label='Generated', color='blue', linewidth=2)
    ax.axhline(y=target_mean, color='red', linestyle='--', label='Target')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean')
    ax.set_title(f'Mean (Target: {target_mean})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Generated std
    ax = axes[1, 0]
    ax.plot(epochs, gen_stds, label='Generated', color='blue', linewidth=2)
    ax.axhline(y=target_std, color='red', linestyle='--', label='Target')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Std')
    ax.set_title(f'Std (Target: {target_std})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Wasserstein distance
    ax = axes[1, 1]
    ax.semilogy(epochs, wasserstein_distances, color='purple', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('W₁ (log)')
    ax.set_title('Wasserstein Distance')
    ax.grid(True, alpha=0.3)
    
    # 6. Summary stats
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = f"""
    Training Summary (Epoch {epoch})
    ================================
    
    Generator Loss:     {g_losses[-1]:.4f}
    Discriminator Loss: {d_losses[-1]:.4f}
    
    Generated Mean:     {gen_means[-1]:.4f} (target: {target_mean})
    Generated Std:      {gen_stds[-1]:.4f} (target: {target_std})
    
    Wasserstein Dist:   {wasserstein_distances[-1]:.4f}
    
    G Gradient Norm:    {g_grad_norms[-1]:.6f}
    D Gradient Norm:    {d_grad_norms[-1]:.6f}
    """
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'QGAN Training Dashboard - Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
