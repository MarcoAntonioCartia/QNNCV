"""
Quantum GAN Training Monitor

Comprehensive monitoring system for quantum GAN training with:
- Real-time performance tracking
- Mode activation visualization
- Training quality metrics
- Automatic performance alerts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import time
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import psutil


class QuantumGANMonitor:
    """
    Comprehensive monitoring system for quantum GAN training.
    
    Features:
    - Real-time performance tracking
    - Mode activation visualization ("bulbs on 1D line")
    - Training quality metrics
    - Automatic alerts and recommendations
    """
    
    def __init__(self, 
                 n_modes: int = 4,
                 max_epoch_time: float = 15.0,
                 min_gradient_flow: float = 0.95,
                 save_dir: str = "results/monitored_training"):
        """
        Initialize monitoring system.
        
        Args:
            n_modes: Number of quantum modes to monitor
            max_epoch_time: Maximum acceptable epoch time (seconds)
            min_gradient_flow: Minimum gradient flow percentage
            save_dir: Directory to save monitoring results
        """
        self.n_modes = n_modes
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Performance thresholds
        self.thresholds = {
            'max_epoch_time': max_epoch_time,
            'min_gradient_flow': min_gradient_flow,
            'max_memory_growth': 0.15,  # 15% per epoch
            'min_loss_improvement': 1e-6
        }
        
        # Monitoring data
        self.metrics_history = {
            'epoch_times': [],
            'g_losses': [],
            'd_losses': [],
            'w_distances': [],
            'gradient_flows_g': [],
            'gradient_flows_d': [],
            'memory_usage': [],
            'cpu_usage': [],
            'sample_diversity': [],
            'mode_balance': []
        }
        
        # Mode-specific tracking
        self.mode_activations = {f'mode_{i}': [] for i in range(n_modes)}
        self.mode_measurements = {f'mode_{i}': [] for i in range(n_modes)}
        self.mode_specialization = {f'mode_{i}': [] for i in range(n_modes)}
        
        # System baseline
        self.baseline_memory = psutil.virtual_memory().percent
        self.training_start_time = None
        
        # Alerts and recommendations
        self.alerts = []
        self.recommendations = []
        
        # Real-time plotting setup
        self.setup_realtime_plots()
        
        print(f"🔍 QuantumGANMonitor initialized:")
        print(f"   Modes: {n_modes}")
        print(f"   Max epoch time: {max_epoch_time}s")
        print(f"   Min gradient flow: {min_gradient_flow*100}%")
        print(f"   Save directory: {save_dir}")
    
    def setup_realtime_plots(self):
        """Setup real-time plotting infrastructure."""
        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
        self.fig.suptitle('Quantum GAN Training Monitor', fontsize=16, fontweight='bold')
        
        # Plot configurations
        self.plot_configs = {
            'losses': {'ax': self.axes[0, 0], 'title': 'Training Losses'},
            'performance': {'ax': self.axes[0, 1], 'title': 'Performance Metrics'},
            'mode_activations': {'ax': self.axes[0, 2], 'title': 'Mode Activations (Quantum Bulbs)'},
            'gradient_flow': {'ax': self.axes[1, 0], 'title': 'Gradient Flow'},
            'system_resources': {'ax': self.axes[1, 1], 'title': 'System Resources'},
            'sample_quality': {'ax': self.axes[1, 2], 'title': 'Sample Quality'}
        }
        
        # Initialize empty plots
        self.plot_lines = {}
        self.mode_bulbs = []
        
        plt.ion()  # Interactive mode on
        plt.tight_layout()
    
    def start_training_monitoring(self, total_epochs: int):
        """Start monitoring a training session."""
        self.training_start_time = time.time()
        self.total_epochs = total_epochs
        
        print(f"\n🚀 Starting monitored training:")
        print(f"   Total epochs: {total_epochs}")
        print(f"   Baseline memory: {self.baseline_memory:.1f}%")
        print(f"   Monitoring thresholds:")
        for key, value in self.thresholds.items():
            print(f"     {key}: {value}")
        print()
    
    def track_epoch(self, 
                   epoch: int,
                   epoch_metrics: Dict[str, float],
                   mode_measurements: Optional[np.ndarray] = None,
                   generated_samples: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Track metrics for a single epoch.
        
        Args:
            epoch: Current epoch number
            epoch_metrics: Dictionary of training metrics
            mode_measurements: Quantum mode measurements [n_modes, n_measurements]
            generated_samples: Generated samples for quality analysis
            
        Returns:
            Dictionary with monitoring results and alerts
        """
        epoch_start = time.time()
        
        # Extract core metrics
        epoch_time = epoch_metrics.get('epoch_time', 0.0)
        g_loss = epoch_metrics.get('g_loss', 0.0)
        d_loss = epoch_metrics.get('d_loss', 0.0)
        w_distance = epoch_metrics.get('w_distance', 0.0)
        g_grad_flow = epoch_metrics.get('g_gradient_flow', 0.0)
        d_grad_flow = epoch_metrics.get('d_gradient_flow', 0.0)
        
        # System metrics
        current_memory = psutil.virtual_memory().percent
        current_cpu = psutil.cpu_percent()
        
        # Store metrics
        self.metrics_history['epoch_times'].append(epoch_time)
        self.metrics_history['g_losses'].append(g_loss)
        self.metrics_history['d_losses'].append(d_loss)
        self.metrics_history['w_distances'].append(w_distance)
        self.metrics_history['gradient_flows_g'].append(g_grad_flow)
        self.metrics_history['gradient_flows_d'].append(d_grad_flow)
        self.metrics_history['memory_usage'].append(current_memory)
        self.metrics_history['cpu_usage'].append(current_cpu)
        
        # Process mode measurements
        if mode_measurements is not None:
            self._track_mode_activations(mode_measurements)
        
        # Analyze sample quality
        if generated_samples is not None:
            diversity_score = self._compute_sample_diversity(generated_samples)
            self.metrics_history['sample_diversity'].append(diversity_score)
        
        # Performance analysis
        alerts = self._analyze_performance(epoch, epoch_time, g_grad_flow, d_grad_flow, current_memory)
        
        # Update real-time plots
        self._update_realtime_plots(epoch)
        
        # Generate status report
        status_report = self._generate_status_report(epoch, epoch_metrics, alerts)
        
        return {
            'status': 'healthy' if not alerts else 'warning',
            'alerts': alerts,
            'recommendations': self.recommendations[-3:],  # Last 3 recommendations
            'metrics_summary': status_report
        }
    
    def _track_mode_activations(self, mode_measurements: np.ndarray):
        """Track quantum mode activations and specialization."""
        n_samples, n_measurements = mode_measurements.shape
        measurements_per_mode = n_measurements // self.n_modes
        
        for mode_idx in range(self.n_modes):
            start_idx = mode_idx * measurements_per_mode
            end_idx = start_idx + measurements_per_mode
            
            mode_data = mode_measurements[:, start_idx:end_idx]
            
            # Compute activation level (magnitude of measurements)
            activation = np.mean(np.abs(mode_data))
            self.mode_activations[f'mode_{mode_idx}'].append(activation)
            
            # Store raw measurements for analysis
            self.mode_measurements[f'mode_{mode_idx}'].append(np.mean(mode_data, axis=0))
            
            # Compute specialization (how distinct this mode is)
            if len(self.mode_activations[f'mode_{mode_idx}']) > 1:
                recent_activations = self.mode_activations[f'mode_{mode_idx}'][-5:]
                specialization = np.std(recent_activations) / (np.mean(recent_activations) + 1e-8)
                self.mode_specialization[f'mode_{mode_idx}'].append(specialization)
    
    def _compute_sample_diversity(self, samples: np.ndarray) -> float:
        """Compute diversity score for generated samples."""
        if len(samples) < 2:
            return 0.0
        
        # Compute pairwise distances
        n_samples = len(samples)
        distances = []
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.linalg.norm(samples[i] - samples[j])
                distances.append(dist)
        
        # Diversity score: mean distance
        return np.mean(distances) if distances else 0.0
    
    def _analyze_performance(self, 
                           epoch: int, 
                           epoch_time: float, 
                           g_grad_flow: float, 
                           d_grad_flow: float,
                           memory_usage: float) -> List[str]:
        """Analyze performance and generate alerts."""
        alerts = []
        
        # Epoch time analysis
        if epoch_time > self.thresholds['max_epoch_time']:
            alerts.append(f"⚠️ Slow epoch: {epoch_time:.1f}s > {self.thresholds['max_epoch_time']}s")
            self.recommendations.append("Consider reducing complexity (modes/layers)")
        
        # Gradient flow analysis
        if g_grad_flow < self.thresholds['min_gradient_flow']:
            alerts.append(f"⚠️ Poor generator gradient flow: {g_grad_flow:.1%}")
            self.recommendations.append("Check generator architecture and loss function")
        
        if d_grad_flow < self.thresholds['min_gradient_flow']:
            alerts.append(f"⚠️ Poor discriminator gradient flow: {d_grad_flow:.1%}")
            self.recommendations.append("Check discriminator architecture and loss function")
        
        # Memory analysis
        memory_growth = memory_usage - self.baseline_memory
        if memory_growth > self.thresholds['max_memory_growth'] * 100 * epoch:
            alerts.append(f"⚠️ High memory growth: +{memory_growth:.1f}%")
            self.recommendations.append("Check for memory leaks in quantum circuit execution")
        
        # Training stability analysis
        if len(self.metrics_history['epoch_times']) > 3:
            recent_times = self.metrics_history['epoch_times'][-3:]
            if max(recent_times) / min(recent_times) > 3.0:
                alerts.append("⚠️ Unstable training times detected")
                self.recommendations.append("Training performance is fluctuating - check system load")
        
        return alerts
    
    def _update_realtime_plots(self, epoch: int):
        """Update real-time visualization plots."""
        if epoch % 1 == 0:  # Update every epoch
            try:
                self._plot_training_losses()
                self._plot_performance_metrics()
                self._plot_mode_activations()
                self._plot_gradient_flow()
                self._plot_system_resources()
                self._plot_sample_quality()
                
                plt.pause(0.01)  # Brief pause for plot update
            except Exception as e:
                print(f"Warning: Plot update failed: {e}")
    
    def _plot_training_losses(self):
        """Plot training losses."""
        ax = self.plot_configs['losses']['ax']
        ax.clear()
        
        epochs = range(len(self.metrics_history['g_losses']))
        
        ax.plot(epochs, self.metrics_history['g_losses'], 'b-', label='Generator', linewidth=2)
        ax.plot(epochs, self.metrics_history['d_losses'], 'r-', label='Discriminator', linewidth=2)
        ax.plot(epochs, self.metrics_history['w_distances'], 'g-', label='Wasserstein Distance', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_metrics(self):
        """Plot performance metrics."""
        ax = self.plot_configs['performance']['ax']
        ax.clear()
        
        epochs = range(len(self.metrics_history['epoch_times']))
        
        # Dual y-axis plot
        ax2 = ax.twinx()
        
        # Epoch times
        line1 = ax.plot(epochs, self.metrics_history['epoch_times'], 'purple', 
                       linewidth=2, label='Epoch Time (s)')
        ax.axhline(y=self.thresholds['max_epoch_time'], color='red', linestyle='--', alpha=0.7)
        
        # Memory usage
        line2 = ax2.plot(epochs, self.metrics_history['memory_usage'], 'orange', 
                        linewidth=2, label='Memory Usage (%)')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (seconds)', color='purple')
        ax2.set_ylabel('Memory (%)', color='orange')
        ax.set_title('Performance Metrics')
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_mode_activations(self):
        """Plot mode activations as 'bulbs on 1D line'."""
        ax = self.plot_configs['mode_activations']['ax']
        ax.clear()
        
        if not self.mode_activations['mode_0']:
            return
        
        # Get latest activations
        current_activations = []
        for mode_idx in range(self.n_modes):
            if self.mode_activations[f'mode_{mode_idx}']:
                activation = self.mode_activations[f'mode_{mode_idx}'][-1]
                current_activations.append(activation)
            else:
                current_activations.append(0.0)
        
        # Normalize activations for visualization
        max_activation = max(current_activations) if max(current_activations) > 0 else 1.0
        normalized_activations = [a / max_activation for a in current_activations]
        
        # Plot as bulbs on a line
        y_positions = [0] * self.n_modes
        x_positions = range(self.n_modes)
        
        # Create bulbs (circles) with size proportional to activation
        for i, activation in enumerate(normalized_activations):
            size = 200 + 800 * activation  # Base size + proportional size
            color_intensity = activation
            
            # Color based on activation level
            circle = Circle((i, 0), 0.3 * activation + 0.1, 
                          color=plt.cm.plasma(color_intensity), alpha=0.8)
            ax.add_patch(circle)
            
            # Add activation value text
            ax.text(i, -0.6, f'{current_activations[i]:.3f}', 
                   ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Plot the base line
        ax.plot(x_positions, y_positions, 'k-', linewidth=3, alpha=0.3)
        
        # Formatting
        ax.set_xlim(-0.5, self.n_modes - 0.5)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('Quantum Mode')
        ax.set_title('Mode Activations (Quantum Bulbs)')
        ax.set_xticks(range(self.n_modes))
        ax.set_xticklabels([f'Mode {i}' for i in range(self.n_modes)])
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_gradient_flow(self):
        """Plot gradient flow percentages."""
        ax = self.plot_configs['gradient_flow']['ax']
        ax.clear()
        
        epochs = range(len(self.metrics_history['gradient_flows_g']))
        
        ax.plot(epochs, [g*100 for g in self.metrics_history['gradient_flows_g']], 
               'b-', label='Generator', linewidth=2)
        ax.plot(epochs, [d*100 for d in self.metrics_history['gradient_flows_d']], 
               'r-', label='Discriminator', linewidth=2)
        
        ax.axhline(y=self.thresholds['min_gradient_flow']*100, 
                  color='orange', linestyle='--', alpha=0.7, label='Threshold')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Flow (%)')
        ax.set_title('Gradient Flow')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
    
    def _plot_system_resources(self):
        """Plot system resource usage."""
        ax = self.plot_configs['system_resources']['ax']
        ax.clear()
        
        epochs = range(len(self.metrics_history['memory_usage']))
        
        ax.plot(epochs, self.metrics_history['memory_usage'], 
               'orange', linewidth=2, label='Memory %')
        ax.plot(epochs, self.metrics_history['cpu_usage'], 
               'green', linewidth=2, label='CPU %')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Usage (%)')
        ax.set_title('System Resources')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    
    def _plot_sample_quality(self):
        """Plot sample quality metrics."""
        ax = self.plot_configs['sample_quality']['ax']
        ax.clear()
        
        if self.metrics_history['sample_diversity']:
            epochs = range(len(self.metrics_history['sample_diversity']))
            ax.plot(epochs, self.metrics_history['sample_diversity'], 
                   'purple', linewidth=2, label='Sample Diversity')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Diversity Score')
            ax.set_title('Sample Quality')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _generate_status_report(self, 
                              epoch: int, 
                              epoch_metrics: Dict[str, float], 
                              alerts: List[str]) -> Dict[str, Any]:
        """Generate comprehensive status report."""
        elapsed_time = time.time() - self.training_start_time if self.training_start_time else 0
        estimated_total = elapsed_time * self.total_epochs / (epoch + 1) if epoch > 0 else 0
        estimated_remaining = estimated_total - elapsed_time
        
        return {
            'epoch': epoch,
            'status': 'healthy' if not alerts else 'warning',
            'timing': {
                'epoch_time': epoch_metrics.get('epoch_time', 0.0),
                'elapsed_total': elapsed_time,
                'estimated_remaining': estimated_remaining,
                'avg_epoch_time': np.mean(self.metrics_history['epoch_times']) if self.metrics_history['epoch_times'] else 0
            },
            'performance': {
                'g_gradient_flow': epoch_metrics.get('g_gradient_flow', 0.0),
                'd_gradient_flow': epoch_metrics.get('d_gradient_flow', 0.0),
                'memory_usage': self.metrics_history['memory_usage'][-1] if self.metrics_history['memory_usage'] else 0,
                'cpu_usage': self.metrics_history['cpu_usage'][-1] if self.metrics_history['cpu_usage'] else 0
            },
            'training': {
                'g_loss': epoch_metrics.get('g_loss', 0.0),
                'd_loss': epoch_metrics.get('d_loss', 0.0),
                'w_distance': epoch_metrics.get('w_distance', 0.0)
            },
            'alerts_count': len(alerts),
            'mode_activations': {f'mode_{i}': self.mode_activations[f'mode_{i}'][-1] 
                               if self.mode_activations[f'mode_{i}'] else 0.0 
                               for i in range(self.n_modes)}
        }
    
    def save_monitoring_data(self, filename_prefix: str = "monitoring"):
        """Save complete monitoring data to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics history
        metrics_file = os.path.join(self.save_dir, f"{filename_prefix}_metrics_{timestamp}.json")
        with open(metrics_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = {}
            for key, values in self.metrics_history.items():
                serializable_metrics[key] = [float(v) for v in values]
            json.dump(serializable_metrics, f, indent=2)
        
        # Save mode activations
        mode_file = os.path.join(self.save_dir, f"{filename_prefix}_modes_{timestamp}.json")
        with open(mode_file, 'w') as f:
            serializable_modes = {}
            for key, values in self.mode_activations.items():
                serializable_modes[key] = [float(v) for v in values]
            json.dump(serializable_modes, f, indent=2)
        
        # Save final plot
        plot_file = os.path.join(self.save_dir, f"{filename_prefix}_plots_{timestamp}.png")
        self.fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        print(f"📊 Monitoring data saved:")
        print(f"   Metrics: {metrics_file}")
        print(f"   Modes: {mode_file}")
        print(f"   Plots: {plot_file}")
        
        return {
            'metrics_file': metrics_file,
            'mode_file': mode_file,
            'plot_file': plot_file
        }
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        if not self.metrics_history['epoch_times']:
            return "No training data available for summary."
        
        total_epochs = len(self.metrics_history['epoch_times'])
        avg_epoch_time = np.mean(self.metrics_history['epoch_times'])
        total_time = sum(self.metrics_history['epoch_times'])
        
        # Performance analysis
        fast_epochs = sum(1 for t in self.metrics_history['epoch_times'] 
                         if t <= self.thresholds['max_epoch_time'])
        performance_score = fast_epochs / total_epochs * 100
        
        # Gradient flow analysis
        avg_g_flow = np.mean(self.metrics_history['gradient_flows_g']) * 100
        avg_d_flow = np.mean(self.metrics_history['gradient_flows_d']) * 100
        
        # Memory stability
        memory_range = (max(self.metrics_history['memory_usage']) - 
                       min(self.metrics_history['memory_usage']))
        
        # Mode activation balance
        final_activations = []
        for i in range(self.n_modes):
            if self.mode_activations[f'mode_{i}']:
                final_activations.append(self.mode_activations[f'mode_{i}'][-1])
        
        mode_balance = (1.0 - np.std(final_activations) / np.mean(final_activations)) * 100 \
                      if final_activations and np.mean(final_activations) > 0 else 0
        
        summary = f"""
🎯 QUANTUM GAN TRAINING SUMMARY
{'='*50}

📊 PERFORMANCE METRICS:
   Total epochs: {total_epochs}
   Average epoch time: {avg_epoch_time:.2f}s
   Total training time: {total_time:.1f}s
   Performance score: {performance_score:.1f}% (epochs ≤ {self.thresholds['max_epoch_time']}s)

🔄 GRADIENT FLOW:
   Generator: {avg_g_flow:.1f}% (avg)
   Discriminator: {avg_d_flow:.1f}% (avg)
   Target: ≥ {self.thresholds['min_gradient_flow']*100:.1f}%

💾 SYSTEM RESOURCES:
   Memory usage range: {memory_range:.1f}%
   Final memory: {self.metrics_history['memory_usage'][-1]:.1f}%
   Baseline memory: {self.baseline_memory:.1f}%

🔬 QUANTUM MODES:
   Mode balance score: {mode_balance:.1f}%
   Active modes: {len([a for a in final_activations if a > 0.01])} / {self.n_modes}
"""
        
        if final_activations:
            summary += "\n   Final mode activations:\n"
            for i, activation in enumerate(final_activations):
                summary += f"     Mode {i}: {activation:.4f}\n"
        
        if self.alerts:
            summary += f"\n⚠️ ALERTS GENERATED: {len(self.alerts)}\n"
            for alert in self.alerts[-5:]:  # Last 5 alerts
                summary += f"   {alert}\n"
        
        if self.recommendations:
            summary += f"\n💡 RECOMMENDATIONS:\n"
            for rec in self.recommendations[-3:]:  # Last 3 recommendations
                summary += f"   • {rec}\n"
        
        summary += f"\n{'='*50}"
        
        return summary


def create_monitor(n_modes: int = 4, **kwargs) -> QuantumGANMonitor:
    """Factory function to create a QuantumGANMonitor."""
    return QuantumGANMonitor(n_modes=n_modes, **kwargs)
