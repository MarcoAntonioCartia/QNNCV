"""
Training Monitor for CV Quantum GAN
====================================

Tracks and stores training metrics, provides callbacks for visualization.

Usage:
    monitor = TrainingMonitor(target_mean=2.0, target_std=0.5)
    
    for epoch in range(epochs):
        # ... training step ...
        monitor.update(g_loss, d_loss, g_grad, d_grad, gen_samples, real_samples)
        
        if epoch % 10 == 0:
            monitor.plot_dashboard(epoch)
"""

import numpy as np
import tensorflow as tf
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from scipy.stats import wasserstein_distance
import json
import os


@dataclass
class TrainingMetrics:
    """Container for all training metrics."""
    g_losses: List[float] = field(default_factory=list)
    d_losses: List[float] = field(default_factory=list)
    g_grad_norms: List[float] = field(default_factory=list)
    d_grad_norms: List[float] = field(default_factory=list)
    gen_means: List[float] = field(default_factory=list)
    gen_stds: List[float] = field(default_factory=list)
    wasserstein_distances: List[float] = field(default_factory=list)
    
    # Optional: per-epoch sample statistics
    gen_samples_history: List[np.ndarray] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, List[float]]:
        """Convert to dictionary (for JSON serialization)."""
        return {
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'g_grad_norms': self.g_grad_norms,
            'd_grad_norms': self.d_grad_norms,
            'gen_means': self.gen_means,
            'gen_stds': self.gen_stds,
            'wasserstein_distances': self.wasserstein_distances
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, List[float]]) -> 'TrainingMetrics':
        """Create from dictionary."""
        metrics = cls()
        metrics.g_losses = data.get('g_losses', [])
        metrics.d_losses = data.get('d_losses', [])
        metrics.g_grad_norms = data.get('g_grad_norms', [])
        metrics.d_grad_norms = data.get('d_grad_norms', [])
        metrics.gen_means = data.get('gen_means', [])
        metrics.gen_stds = data.get('gen_stds', [])
        metrics.wasserstein_distances = data.get('wasserstein_distances', [])
        return metrics


class TrainingMonitor:
    """
    Monitors and tracks CV Quantum GAN training.
    
    Features:
    - Tracks losses, gradients, and distribution statistics
    - Computes Wasserstein distance
    - Provides visualization callbacks
    - Saves/loads training history
    - Early stopping support
    
    Args:
        target_mean: Target distribution mean
        target_std: Target distribution standard deviation
        log_dir: Directory for saving logs and checkpoints
        save_samples: Whether to store generated samples history
        verbose: Print updates during training
    """
    
    def __init__(
        self,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        log_dir: Optional[str] = None,
        save_samples: bool = False,
        verbose: bool = True
    ):
        self.target_mean = target_mean
        self.target_std = target_std
        self.log_dir = log_dir
        self.save_samples = save_samples
        self.verbose = verbose
        
        self.metrics = TrainingMetrics()
        self.best_wasserstein = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        # Create log directory if specified
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def update(
        self,
        g_loss: tf.Tensor,
        d_loss: tf.Tensor,
        g_grads: List[tf.Tensor],
        d_grads: List[tf.Tensor],
        gen_samples: np.ndarray,
        real_samples: np.ndarray
    ) -> Dict[str, float]:
        """
        Update metrics after a training step.
        
        Args:
            g_loss: Generator loss (tensor)
            d_loss: Discriminator loss (tensor)
            g_grads: Generator gradients (list of tensors)
            d_grads: Discriminator gradients (list of tensors)
            gen_samples: Generated samples (numpy array)
            real_samples: Real samples (numpy array)
            
        Returns:
            Dictionary with current epoch metrics
        """
        # Convert to numpy if needed
        g_loss_val = float(g_loss.numpy() if hasattr(g_loss, 'numpy') else g_loss)
        d_loss_val = float(d_loss.numpy() if hasattr(d_loss, 'numpy') else d_loss)
        
        # Compute gradient norms
        g_grad_norm = self._compute_grad_norm(g_grads)
        d_grad_norm = self._compute_grad_norm(d_grads)
        
        # Flatten samples
        gen_flat = np.array(gen_samples).flatten()
        real_flat = np.array(real_samples).flatten()
        
        # Compute statistics
        gen_mean = float(np.mean(gen_flat))
        gen_std = float(np.std(gen_flat))
        
        # Wasserstein distance
        w_dist = float(wasserstein_distance(gen_flat, real_flat))
        
        # Store metrics
        self.metrics.g_losses.append(g_loss_val)
        self.metrics.d_losses.append(d_loss_val)
        self.metrics.g_grad_norms.append(g_grad_norm)
        self.metrics.d_grad_norms.append(d_grad_norm)
        self.metrics.gen_means.append(gen_mean)
        self.metrics.gen_stds.append(gen_std)
        self.metrics.wasserstein_distances.append(w_dist)
        
        if self.save_samples:
            self.metrics.gen_samples_history.append(gen_flat.copy())
        
        # Track best
        if w_dist < self.best_wasserstein:
            self.best_wasserstein = w_dist
            self.best_epoch = len(self.metrics.g_losses)
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        return {
            'g_loss': g_loss_val,
            'd_loss': d_loss_val,
            'g_grad_norm': g_grad_norm,
            'd_grad_norm': d_grad_norm,
            'gen_mean': gen_mean,
            'gen_std': gen_std,
            'wasserstein': w_dist
        }
    
    def _compute_grad_norm(self, grads: List[tf.Tensor]) -> float:
        """Compute total gradient norm across all variables."""
        if grads is None or all(g is None for g in grads):
            return 0.0
        
        total_norm = 0.0
        for g in grads:
            if g is not None:
                norm = tf.norm(g).numpy()
                total_norm += norm ** 2
        
        return float(np.sqrt(total_norm))
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get the most recent metrics."""
        if len(self.metrics.g_losses) == 0:
            return {}
        
        return {
            'g_loss': self.metrics.g_losses[-1],
            'd_loss': self.metrics.d_losses[-1],
            'g_grad_norm': self.metrics.g_grad_norms[-1],
            'd_grad_norm': self.metrics.d_grad_norms[-1],
            'gen_mean': self.metrics.gen_means[-1],
            'gen_std': self.metrics.gen_stds[-1],
            'wasserstein': self.metrics.wasserstein_distances[-1]
        }
    
    def print_status(self, epoch: int, total_epochs: int):
        """Print current training status."""
        if not self.verbose or len(self.metrics.g_losses) == 0:
            return
        
        m = self.get_current_metrics()
        mean_err = abs(m['gen_mean'] - self.target_mean)
        std_err = abs(m['gen_std'] - self.target_std)
        
        print(f"Epoch {epoch}/{total_epochs} | "
              f"G_loss: {m['g_loss']:.4f} | "
              f"D_loss: {m['d_loss']:.4f} | "
              f"Mean: {m['gen_mean']:.3f} (err: {mean_err:.3f}) | "
              f"Std: {m['gen_std']:.3f} (err: {std_err:.3f}) | "
              f"W₁: {m['wasserstein']:.4f}")
    
    def should_stop_early(self, patience: int = 50) -> bool:
        """Check if training should stop early."""
        return self.epochs_without_improvement >= patience
    
    def plot_dashboard(self, epoch: int, save: bool = True) -> None:
        """
        Plot comprehensive training dashboard.
        
        Uses visualization module - import here to avoid circular imports.
        """
        from utils.visualization import plot_training_dashboard
        
        save_path = None
        if save and self.log_dir:
            save_path = os.path.join(self.log_dir, f'dashboard_epoch_{epoch}.png')
        
        plot_training_dashboard(
            g_losses=self.metrics.g_losses,
            d_losses=self.metrics.d_losses,
            g_grad_norms=self.metrics.g_grad_norms,
            d_grad_norms=self.metrics.d_grad_norms,
            gen_means=self.metrics.gen_means,
            gen_stds=self.metrics.gen_stds,
            wasserstein_distances=self.metrics.wasserstein_distances,
            target_mean=self.target_mean,
            target_std=self.target_std,
            epoch=epoch,
            save_path=save_path
        )
    
    def plot_distributions(
        self, 
        gen_samples: np.ndarray, 
        real_samples: np.ndarray,
        epoch: int,
        save: bool = True
    ) -> None:
        """Plot distribution comparison."""
        from utils.visualization import plot_distribution_comparison
        
        save_path = None
        if save and self.log_dir:
            save_path = os.path.join(self.log_dir, f'distributions_epoch_{epoch}.png')
        
        plot_distribution_comparison(
            generated_samples=gen_samples,
            real_samples=real_samples,
            epoch=epoch,
            target_mean=self.target_mean,
            target_std=self.target_std,
            save_path=save_path
        )
    
    def save_metrics(self, filepath: Optional[str] = None) -> str:
        """Save training metrics to JSON file."""
        if filepath is None:
            if self.log_dir:
                filepath = os.path.join(self.log_dir, 'training_metrics.json')
            else:
                filepath = 'training_metrics.json'
        
        data = {
            'metrics': self.metrics.to_dict(),
            'target_mean': self.target_mean,
            'target_std': self.target_std,
            'best_wasserstein': self.best_wasserstein,
            'best_epoch': self.best_epoch
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        if self.verbose:
            print(f"Metrics saved to {filepath}")
        
        return filepath
    
    def load_metrics(self, filepath: str) -> None:
        """Load training metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.metrics = TrainingMetrics.from_dict(data['metrics'])
        self.target_mean = data.get('target_mean', self.target_mean)
        self.target_std = data.get('target_std', self.target_std)
        self.best_wasserstein = data.get('best_wasserstein', float('inf'))
        self.best_epoch = data.get('best_epoch', 0)
        
        if self.verbose:
            print(f"Metrics loaded from {filepath}")
            print(f"  Epochs: {len(self.metrics.g_losses)}")
            print(f"  Best W₁: {self.best_wasserstein:.4f} at epoch {self.best_epoch}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        if len(self.metrics.g_losses) == 0:
            return {}
        
        return {
            'total_epochs': len(self.metrics.g_losses),
            'final_g_loss': self.metrics.g_losses[-1],
            'final_d_loss': self.metrics.d_losses[-1],
            'final_mean': self.metrics.gen_means[-1],
            'final_std': self.metrics.gen_stds[-1],
            'final_wasserstein': self.metrics.wasserstein_distances[-1],
            'best_wasserstein': self.best_wasserstein,
            'best_epoch': self.best_epoch,
            'mean_error': abs(self.metrics.gen_means[-1] - self.target_mean),
            'std_error': abs(self.metrics.gen_stds[-1] - self.target_std),
            'converged': (
                abs(self.metrics.gen_means[-1] - self.target_mean) < 0.3 and
                abs(self.metrics.gen_stds[-1] - self.target_std) < 0.2
            )
        }


class QuantumStateMonitor:
    """
    Monitors quantum state evolution during training.
    
    Captures Wigner functions and quantum state properties at specified epochs.
    
    IMPORTANT: Call capture_state() OUTSIDE the gradient tape!
    """
    
    def __init__(
        self,
        generator,
        fixed_latent: tf.Tensor,
        log_dir: Optional[str] = None,
        capture_epochs: List[int] = None
    ):
        """
        Args:
            generator: The quantum generator model
            fixed_latent: Fixed latent vector for consistent visualization
            log_dir: Directory for saving state visualizations
            capture_epochs: Which epochs to capture (default: [0, 10, 50, 100, ...])
        """
        self.generator = generator
        self.fixed_latent = fixed_latent
        self.log_dir = log_dir
        self.capture_epochs = capture_epochs or [0, 1, 5, 10, 25, 50, 100, 200, 500]
        
        self.captured_states = {}
        
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def should_capture(self, epoch: int) -> bool:
        """Check if this epoch should be captured."""
        return epoch in self.capture_epochs
    
    def capture_state(self, epoch: int):
        """
        Capture the quantum state for visualization.
        
        CALL THIS OUTSIDE THE GRADIENT TAPE!
        """
        if not self.should_capture(epoch):
            return
        
        # Execute circuit to get state (not just measurements)
        state = self._get_quantum_state()
        
        if state is not None:
            self.captured_states[epoch] = state
            
            # Optionally plot immediately
            if self.log_dir:
                from utils.visualization import plot_wigner_2d
                save_path = os.path.join(self.log_dir, f'wigner_epoch_{epoch}.png')
                plot_wigner_2d(state, title=f'Wigner Function - Epoch {epoch}',
                              save_path=save_path)
    
    def _get_quantum_state(self):
        """
        Execute generator circuit and return quantum state.
        
        This requires accessing the SF engine and running without collapsing
        to just measurements.
        """
        try:
            # Get the single latent sample
            z_single = self.fixed_latent[0] if len(self.fixed_latent.shape) > 1 else self.fixed_latent
            
            # Build mapping for the circuit
            mapping = {}
            
            # Input encoding parameters
            n_modes = self.generator.n_modes
            latent_dim = self.generator.latent_dim
            
            # Handle displacement encoding (2 params per mode)
            if hasattr(self.generator, 'input_params'):
                if latent_dim == 2 * n_modes:  # Full displacement encoding
                    for i in range(n_modes):
                        mapping[self.generator.input_params[2*i].name] = z_single[2*i]
                        mapping[self.generator.input_params[2*i+1].name] = z_single[2*i+1]
                else:  # Simple encoding
                    for i in range(min(latent_dim, n_modes)):
                        mapping[self.generator.input_params[i].name] = z_single[i]
            
            # QNN parameters
            for p, w in zip(self.generator.sf_params.flatten(), 
                           tf.reshape(self.generator.weights, [-1])):
                mapping[p.name] = w
            
            # Reset and execute
            if self.generator.eng.run_progs:
                self.generator.eng.reset()
            
            result = self.generator.eng.run(self.generator.prog, args=mapping)
            return result.state
            
        except Exception as e:
            print(f"Warning: Could not capture quantum state: {e}")
            return None
    
    def plot_evolution(self, save_path: Optional[str] = None):
        """Plot Wigner function evolution across captured epochs."""
        if len(self.captured_states) == 0:
            print("No states captured yet.")
            return
        
        from utils.visualization import plot_wigner_comparison
        
        epochs = sorted(self.captured_states.keys())
        states = [self.captured_states[e] for e in epochs]
        titles = [f'Epoch {e}' for e in epochs]
        
        save_path = save_path or (
            os.path.join(self.log_dir, 'wigner_evolution.png') 
            if self.log_dir else None
        )
        
        plot_wigner_comparison(states, titles, save_path=save_path)
