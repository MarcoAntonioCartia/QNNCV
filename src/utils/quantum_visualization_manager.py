"""
Comprehensive Quantum Visualization Manager for Pure SF Quantum GANs

This module provides a complete visualization system specifically designed for Pure SF
quantum circuits, integrating circuit structure, quantum states, measurements, and
training dynamics visualization.

Features:
- Real-time quantum state visualization during training
- Pure SF circuit structure analysis and visualization
- Measurement outcome tracking and analysis
- Parameter evolution monitoring
- Entanglement and correlation analysis
- Training dynamics dashboards
- Interactive 3D quantum state visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.subplots as sp
from datetime import datetime
import seaborn as sns
import pandas as pd
from scipy.stats import entropy
from sklearn.decomposition import PCA
import warnings

logger = logging.getLogger(__name__)


class QuantumVisualizationManager:
    """
    Advanced visualization manager for Pure SF Quantum GANs.
    
    This class provides comprehensive visualization capabilities specifically
    designed for Pure SF circuits, including real-time state monitoring,
    circuit analysis, and training visualization.
    """
    
    def __init__(self, save_directory: str = "visualizations"):
        """
        Initialize the visualization manager.
        
        Args:
            save_directory: Directory to save visualization outputs
        """
        self.save_dir = save_directory
        self.visualization_history = []
        self.parameter_history = []
        self.state_history = []
        self.measurement_history = []
        
        # Create save directory
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Suppress warnings for cleaner output
        warnings.filterwarnings('ignore', category=UserWarning)
        
        logger.info(f"QuantumVisualizationManager initialized")
        logger.info(f"  Save directory: {save_directory}")
    
    # =====================================================================
    # PURE SF CIRCUIT VISUALIZATION
    # =====================================================================
    
    def visualize_pure_sf_circuit(self, circuit, title: str = "Pure SF Circuit", 
                                 show_parameters: bool = True, 
                                 show_values: bool = True,
                                 save: bool = True):
        """
        Visualize Pure SF circuit structure with detailed parameter analysis.
        
        Args:
            circuit: PureSFCircuit instance
            title: Plot title
            show_parameters: Show parameter names
            show_values: Show parameter values
            save: Save the visualization
        """
        print(f"{title.upper()}")
        print("=" * 60)
        
        # Circuit configuration
        print(f"Circuit Configuration:")
        print(f"  Architecture: Pure SF (Program-Engine)")
        print(f"  Modes: {circuit.n_modes}")
        print(f"  Layers: {circuit.layers}")
        print(f"  Parameters: {len(circuit.trainable_variables)}")
        print(f"  Cutoff: {circuit.cutoff_dim}")
        print(f"  Circuit type: {getattr(circuit, 'circuit_type', 'variational')}")
        
        # Parameter analysis
        if show_parameters and hasattr(circuit, 'trainable_variables'):
            self._analyze_circuit_parameters(circuit, show_values)
        
        # Circuit structure visualization
        if hasattr(circuit, 'build_program'):
            try:
                # Build symbolic program for visualization
                prog = circuit.build_program({})
                print(f"\n📋 Circuit Structure:")
                print("-" * 40)
                prog.print()
                
                # Save circuit diagram if requested
                if save:
                    self._save_circuit_diagram(prog, title)
                    
            except Exception as e:
                logger.warning(f"Could not build circuit program for visualization: {e}")
        
        # Parameter distribution analysis
        if hasattr(circuit, 'trainable_variables') and len(circuit.trainable_variables) > 0:
            self._visualize_parameter_distribution(circuit, title, save)
    
    def _analyze_circuit_parameters(self, circuit, show_values: bool = True):
        """Analyze and display circuit parameters."""
        print(f"\n Parameter Analysis ({len(circuit.trainable_variables)} parameters):")
        print("-" * 40)
        
        param_groups = {}
        for i, var in enumerate(circuit.trainable_variables):
            param_name = var.name.split(':')[0]
            param_type = param_name.split('_')[0]  # e.g., 'squeeze', 'displacement', etc.
            
            if param_type not in param_groups:
                param_groups[param_type] = []
            
            if show_values:
                value = float(var.numpy())
                param_groups[param_type].append((param_name, value))
                print(f"  {i:2d}. {param_name:<25} = {value:8.4f}")
            else:
                param_groups[param_type].append((param_name, None))
                print(f"  {i:2d}. {param_name}")
        
        # Parameter type summary
        print(f"\nParameter Type Summary:")
        for param_type, params in param_groups.items():
            print(f"  {param_type.capitalize()}: {len(params)} parameters")
    
    def _save_circuit_diagram(self, prog, title: str):
        """Save circuit diagram to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.save_dir}/circuit_{title.lower().replace(' ', '_')}_{timestamp}.txt"
            
            # Capture program print output
            import io
            import contextlib
            
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                prog.print()
            
            circuit_text = buffer.getvalue()
            
            with open(filename, 'w') as f:
                f.write(f"Circuit Diagram: {title}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                f.write(circuit_text)
            
            logger.info(f"Circuit diagram saved to {filename}")
            
        except Exception as e:
            logger.warning(f"Could not save circuit diagram: {e}")
    
    def _visualize_parameter_distribution(self, circuit, title: str, save: bool = True):
        """Visualize parameter value distributions."""
        if not hasattr(circuit, 'trainable_variables') or len(circuit.trainable_variables) == 0:
            return
        
        # Extract parameter values
        param_values = [float(var.numpy()) for var in circuit.trainable_variables]
        param_names = [var.name.split(':')[0] for var in circuit.trainable_variables]
        
        # Create parameter analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Parameter value histogram
        axes[0, 0].hist(param_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Parameter Value Distribution')
        axes[0, 0].set_xlabel('Parameter Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Parameter values by index
        axes[0, 1].plot(param_values, 'o-', alpha=0.7, markersize=4)
        axes[0, 1].set_title('Parameter Values by Index')
        axes[0, 1].set_xlabel('Parameter Index')
        axes[0, 1].set_ylabel('Parameter Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Parameter magnitude
        param_magnitudes = [abs(val) for val in param_values]
        axes[1, 0].bar(range(len(param_magnitudes)), param_magnitudes, alpha=0.7, color='coral')
        axes[1, 0].set_title('Parameter Magnitudes')
        axes[1, 0].set_xlabel('Parameter Index')
        axes[1, 0].set_ylabel('|Parameter Value|')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Parameter statistics
        stats_text = f"""Parameter Statistics:
Mean: {np.mean(param_values):.4f}
Std:  {np.std(param_values):.4f}
Min:  {np.min(param_values):.4f}
Max:  {np.max(param_values):.4f}
Range: {np.max(param_values) - np.min(param_values):.4f}

Total Parameters: {len(param_values)}
Zero Parameters: {sum(1 for v in param_values if abs(v) < 1e-6)}
Large Parameters: {sum(1 for v in param_values if abs(v) > 1.0)}"""
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 1].set_title('Parameter Statistics')
        axes[1, 1].axis('off')
        
        plt.suptitle(f'{title} - Parameter Analysis', fontsize=16)
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.save_dir}/params_{title.lower().replace(' ', '_')}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Parameter analysis saved to {filename}")
        
        plt.show()
    
    # =====================================================================
    # QUANTUM STATE VISUALIZATION
    # =====================================================================
    
    def visualize_quantum_state(self, state, title: str = "Quantum State",
                               modes: Optional[List[int]] = None,
                               x_range: Tuple[float, float] = (-4, 4),
                               p_range: Tuple[float, float] = (-4, 4),
                               resolution: int = 100,
                               save: bool = True):
        """
        Comprehensive quantum state visualization.
        
        Args:
            state: SF quantum state object
            title: Visualization title
            modes: List of modes to visualize (default: all)
            x_range: X quadrature range
            p_range: P quadrature range
            resolution: Grid resolution
            save: Save visualizations
        """
        if modes is None:
            modes = list(range(min(state.num_modes, 4)))  # Limit to 4 modes for display
        
        print(f"🎭 {title.upper()} - QUANTUM STATE ANALYSIS")
        print("=" * 60)
        print(f"State Configuration:")
        print(f"  Modes: {state.num_modes}")
        print(f"  Cutoff: {getattr(state, 'cutoff_dim', 'Unknown')}")
        print(f"  Visualizing modes: {modes}")
        
        # 1. 3D Wigner function mountains
        self._create_wigner_mountains(state, modes, title, x_range, p_range, resolution, save)
        
        # 2. Fock probability distributions
        self._visualize_fock_distributions(state, modes, title, save)
        
        # 3. Quadrature distributions
        self._visualize_quadrature_distributions(state, modes, title, save)
        
        # 4. State entropy analysis
        self._analyze_state_entropy(state, modes, title, save)
    
    def _create_wigner_mountains(self, state, modes: List[int], title: str,
                                x_range: Tuple[float, float], p_range: Tuple[float, float],
                                resolution: int, save: bool):
        """Create 3D Wigner function visualizations."""
        print(f"\n Creating Wigner function mountains...")
        
        n_modes = len(modes)
        cols = min(2, n_modes)
        rows = (n_modes + cols - 1) // cols
        
        fig = plt.figure(figsize=(8*cols, 6*rows))
        
        # Create coordinate grids
        x = np.linspace(x_range[0], x_range[1], resolution)
        p = np.linspace(p_range[0], p_range[1], resolution)
        X, P = np.meshgrid(x, p)
        
        for i, mode in enumerate(modes):
            ax = fig.add_subplot(rows, cols, i+1, projection='3d')
            
            try:
                # Calculate Wigner function
                W = state.wigner(mode, x, p)
                
                # Create 3D surface
                surface = ax.plot_surface(X, P, W, cmap='RdYlBu_r',
                                        alpha=0.9, linewidth=0.3,
                                        rstride=max(1, resolution//40),
                                        cstride=max(1, resolution//40))
                
                # Add contour lines at base
                contour_levels = np.linspace(np.min(W), np.max(W), 10)
                ax.contour(X, P, W, levels=contour_levels, colors='gray',
                          alpha=0.5, offset=np.min(W)-0.02)
                
                # Styling
                ax.set_title(f'Mode {mode} Wigner Function', fontsize=12, pad=10)
                ax.set_xlabel('Position (x)', fontsize=10)
                ax.set_ylabel('Momentum (p)', fontsize=10)
                ax.set_zlabel('W(x,p)', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Analyze state properties
                max_w = np.max(W)
                min_w = np.min(W)
                neg_volume = np.sum(W[W < 0]) * (x[1] - x[0]) * (p[1] - p[0])
                
                print(f"    Mode {mode}: W_max={max_w:.3f}, W_min={min_w:.3f}, Neg_vol={neg_volume:.3f}")
                
            except Exception as e:
                logger.warning(f"Could not create Wigner function for mode {mode}: {e}")
        
        plt.suptitle(f'{title} - Wigner Function Mountains', fontsize=16)
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.save_dir}/wigner_{title.lower().replace(' ', '_')}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Wigner functions saved to {filename}")
        
        plt.show()
    
    def _visualize_fock_distributions(self, state, modes: List[int], title: str, save: bool):
        """Visualize Fock state probability distributions."""
        print(f"\nAnalyzing Fock state distributions...")
        
        cutoff = min(8, getattr(state, 'cutoff_dim', 8))
        n_modes = len(modes)
        
        fig, axes = plt.subplots(1, n_modes, figsize=(4*n_modes, 4))
        if n_modes == 1:
            axes = [axes]
        
        for i, mode in enumerate(modes):
            try:
                # Calculate Fock probabilities
                fock_probs = []
                for n in range(cutoff):
                    fock_state = [0] * state.num_modes
                    fock_state[mode] = n
                    prob = state.fock_prob(fock_state)
                    fock_probs.append(prob)
                
                # Plot distribution
                bars = axes[i].bar(range(cutoff), fock_probs, alpha=0.7, 
                                  color=plt.cm.viridis(i/max(1, n_modes-1)))
                axes[i].set_title(f'Mode {mode} Fock Distribution')
                axes[i].set_xlabel('Fock Number |n⟩')
                axes[i].set_ylabel('Probability')
                axes[i].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for j, (bar, prob) in enumerate(zip(bars, fock_probs)):
                    if prob > 0.01:  # Only label significant probabilities
                        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                                   f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
                
                # Calculate mean photon number
                mean_n = sum(n * prob for n, prob in enumerate(fock_probs))
                axes[i].axvline(mean_n, color='red', linestyle='--', alpha=0.8, 
                               label=f'⟨n⟩={mean_n:.2f}')
                axes[i].legend()
                
                print(f"    Mode {mode}: ⟨n⟩={mean_n:.3f}, max_prob={max(fock_probs):.3f}")
                
            except Exception as e:
                logger.warning(f"Could not analyze Fock distribution for mode {mode}: {e}")
        
        plt.suptitle(f'{title} - Fock State Distributions', fontsize=14)
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.save_dir}/fock_{title.lower().replace(' ', '_')}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _visualize_quadrature_distributions(self, state, modes: List[int], title: str, save: bool):
        """Visualize quadrature (X and P) distributions."""
        print(f"\nAnalyzing quadrature distributions...")
        
        fig, axes = plt.subplots(2, len(modes), figsize=(4*len(modes), 8))
        if len(modes) == 1:
            axes = axes.reshape(-1, 1)
        
        x_vals = np.linspace(-4, 4, 200)
        
        for i, mode in enumerate(modes):
            try:
                # Calculate quadrature distributions
                x_quad = state.quad_expectation(mode, 0)  # X quadrature (phi=0)
                p_quad = state.quad_expectation(mode, np.pi/2)  # P quadrature (phi=π/2)
                
                # X quadrature distribution
                x_probs = [state.quad_prob(mode, x, 0) for x in x_vals]
                axes[0, i].plot(x_vals, x_probs, 'b-', linewidth=2, label='X quadrature')
                axes[0, i].axvline(x_quad, color='red', linestyle='--', label=f'⟨X⟩={x_quad:.3f}')
                axes[0, i].set_title(f'Mode {mode} X Quadrature')
                axes[0, i].set_xlabel('x')
                axes[0, i].set_ylabel('P(x)')
                axes[0, i].grid(True, alpha=0.3)
                axes[0, i].legend()
                
                # P quadrature distribution
                p_probs = [state.quad_prob(mode, p, np.pi/2) for p in x_vals]
                axes[1, i].plot(x_vals, p_probs, 'g-', linewidth=2, label='P quadrature')
                axes[1, i].axvline(p_quad, color='red', linestyle='--', label=f'⟨P⟩={p_quad:.3f}')
                axes[1, i].set_title(f'Mode {mode} P Quadrature')
                axes[1, i].set_xlabel('p')
                axes[1, i].set_ylabel('P(p)')
                axes[1, i].grid(True, alpha=0.3)
                axes[1, i].legend()
                
                print(f"    Mode {mode}: ⟨X⟩={x_quad:.3f}, ⟨P⟩={p_quad:.3f}")
                
            except Exception as e:
                logger.warning(f"Could not analyze quadratures for mode {mode}: {e}")
                # Fill with placeholder
                axes[0, i].text(0.5, 0.5, f'Mode {mode}\nN/A', ha='center', va='center', 
                               transform=axes[0, i].transAxes)
                axes[1, i].text(0.5, 0.5, f'Mode {mode}\nN/A', ha='center', va='center',
                               transform=axes[1, i].transAxes)
        
        plt.suptitle(f'{title} - Quadrature Distributions', fontsize=16)
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.save_dir}/quadratures_{title.lower().replace(' ', '_')}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _analyze_state_entropy(self, state, modes: List[int], title: str, save: bool):
        """Analyze quantum state entropy and entanglement properties."""
        print(f"\n🔬 Analyzing state entropy and entanglement...")
        
        try:
            # Calculate various entropy measures
            cutoff = min(6, getattr(state, 'cutoff_dim', 6))
            
            entropies = {}
            for mode in modes:
                # Calculate single-mode entropy (approximation using Fock probabilities)
                fock_probs = []
                for n in range(cutoff):
                    fock_state = [0] * state.num_modes
                    fock_state[mode] = n
                    prob = state.fock_prob(fock_state)
                    fock_probs.append(prob)
                
                # Renormalize (in case of truncation)
                total_prob = sum(fock_probs)
                if total_prob > 0:
                    fock_probs = [p/total_prob for p in fock_probs]
                    
                    # Calculate von Neumann entropy (approximation)
                    mode_entropy = entropy(fock_probs, base=2) if any(p > 0 for p in fock_probs) else 0
                    entropies[mode] = mode_entropy
                    
                    print(f"    Mode {mode}: von Neumann entropy ≈ {mode_entropy:.3f} bits")
            
            # Create entropy visualization
            if entropies:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Entropy by mode
                modes_list = list(entropies.keys())
                entropy_vals = list(entropies.values())
                bars = ax1.bar(modes_list, entropy_vals, alpha=0.7, color='purple')
                ax1.set_title('Von Neumann Entropy by Mode')
                ax1.set_xlabel('Mode')
                ax1.set_ylabel('Entropy (bits)')
                ax1.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, val in zip(bars, entropy_vals):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{val:.3f}', ha='center', va='bottom')
                
                # Entropy statistics
                if len(entropy_vals) > 1:
                    mean_entropy = np.mean(entropy_vals)
                    std_entropy = np.std(entropy_vals)
                    total_entropy = sum(entropy_vals)
                    
                    stats_text = f"""Entropy Statistics:
Mean: {mean_entropy:.3f} bits
Std:  {std_entropy:.3f} bits
Total: {total_entropy:.3f} bits
Max:  {max(entropy_vals):.3f} bits
Min:  {min(entropy_vals):.3f} bits

Modes analyzed: {len(entropies)}
Cutoff used: {cutoff}"""
                else:
                    stats_text = f"""Single Mode Analysis:
Entropy: {entropy_vals[0]:.3f} bits
Mode: {modes_list[0]}
Cutoff: {cutoff}"""
                
                ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                ax2.set_title('Entropy Analysis')
                ax2.axis('off')
                
                plt.suptitle(f'{title} - Quantum State Entropy Analysis', fontsize=14)
                plt.tight_layout()
                
                if save:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{self.save_dir}/entropy_{title.lower().replace(' ', '_')}_{timestamp}.png"
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                
                plt.show()
            
        except Exception as e:
            logger.warning(f"Could not analyze state entropy: {e}")
    
    # =====================================================================
    # TRAINING VISUALIZATION AND MONITORING
    # =====================================================================
    
    def create_training_dashboard(self, training_history: Dict[str, List[float]], 
                                 title: str = "Quantum GAN Training",
                                 save: bool = True):
        """
        Create comprehensive training dashboard.
        
        Args:
            training_history: Dictionary with training metrics
            title: Dashboard title
            save: Save the dashboard
        """
        print(f"📊 Creating {title} Dashboard...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Loss curves
        ax1 = fig.add_subplot(gs[0, 0:2])
        if 'g_loss' in training_history and 'd_loss' in training_history:
            epochs = range(len(training_history['g_loss']))
            ax1.plot(epochs, training_history['g_loss'], 'b-', linewidth=2, label='Generator', alpha=0.8)
            ax1.plot(epochs, training_history['d_loss'], 'r-', linewidth=2, label='Discriminator', alpha=0.8)
            ax1.set_title('Training Losses', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Wasserstein distance
        ax2 = fig.add_subplot(gs[0, 2])
        if 'w_distance' in training_history:
            epochs = range(len(training_history['w_distance']))
            ax2.plot(epochs, training_history['w_distance'], 'g-', linewidth=2)
            ax2.set_title('Wasserstein Distance', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Distance')
            ax2.grid(True, alpha=0.3)
        
        # 3. Gradient flow
        ax3 = fig.add_subplot(gs[0, 3])
        if 'g_gradients' in training_history:
            epochs = range(len(training_history['g_gradients']))
            ax3.plot(epochs, training_history['g_gradients'], 'b-', linewidth=2, label='Generator')
            if 'd_gradients' in training_history:
                ax3.plot(epochs, training_history['d_gradients'], 'r-', linewidth=2, label='Discriminator')
            ax3.set_title('Gradient Flow', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Active Gradients')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Gradient penalty
        ax4 = fig.add_subplot(gs[1, 0])
        if 'gradient_penalty' in training_history:
            epochs = range(len(training_history['gradient_penalty']))
            ax4.plot(epochs, training_history['gradient_penalty'], 'purple', linewidth=2)
            ax4.set_title('Gradient Penalty', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Penalty')
            ax4.grid(True, alpha=0.3)
        
        # 5. Entropy evolution
        ax5 = fig.add_subplot(gs[1, 1])
        if 'entropy_bonus' in training_history:
            epochs = range(len(training_history['entropy_bonus']))
            ax5.plot(epochs, training_history['entropy_bonus'], 'orange', linewidth=2)
            ax5.set_title('Entropy Regularization', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Entropy Bonus')
            ax5.grid(True, alpha=0.3)
        
        # 6. Physics penalty
        ax6 = fig.add_subplot(gs[1, 2])
        if 'physics_penalty' in training_history:
            epochs = range(len(training_history['physics_penalty']))
            ax6.plot(epochs, training_history['physics_penalty'], 'brown', linewidth=2)
            ax6.set_title('Physics Penalty', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('Penalty')
            ax6.grid(True, alpha=0.3)
        
        # 7. Training summary
        ax7 = fig.add_subplot(gs[1, 3])
        if training_history:
            latest_metrics = {}
            for key, values in training_history.items():
                if values:
                    latest_metrics[key] = values[-1]
            
            summary_text = "Latest Metrics:\n"
            summary_text += "-" * 15 + "\n"
            for key, value in latest_metrics.items():
                summary_text += f"{key}: {value:.4f}\n"
            
            ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            ax7.set_title('Current Status', fontsize=12, fontweight='bold')
            ax7.axis('off')
        
        # 8. Parameter evolution heatmap (if available)
        ax8 = fig.add_subplot(gs[2, :])
        if hasattr(self, 'parameter_history') and self.parameter_history:
            param_matrix = np.array(self.parameter_history).T
            im = ax8.imshow(param_matrix, cmap='RdBu_r', aspect='auto', interpolation='bilinear')
            ax8.set_title('Parameter Evolution Heatmap', fontsize=14, fontweight='bold')
            ax8.set_xlabel('Training Step')
            ax8.set_ylabel('Parameter Index')
            plt.colorbar(im, ax=ax8, label='Parameter Value')
        else:
            ax8.text(0.5, 0.5, 'Parameter Evolution\n(Enable parameter tracking)', 
                    ha='center', va='center', transform=ax8.transAxes, fontsize=14)
            ax8.set_title('Parameter Evolution', fontsize=14, fontweight='bold')
            ax8.axis('off')
        
        plt.suptitle(f'{title} Dashboard', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.save_dir}/dashboard_{title.lower().replace(' ', '_')}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Training dashboard saved to {filename}")
        
        plt.show()
    
    def track_parameters(self, generator, discriminator):
        """
        Track parameter evolution during training.
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
        """
        if hasattr(generator, 'trainable_variables') and hasattr(discriminator, 'trainable_variables'):
            g_params = [float(var.numpy()) for var in generator.trainable_variables]
            d_params = [float(var.numpy()) for var in discriminator.trainable_variables]
            all_params = g_params + d_params
            self.parameter_history.append(all_params)
    
    def create_quantum_state_evolution_gif(self, states: List[Any], 
                                          title: str = "Quantum State Evolution",
                                          mode: int = 0,
                                          save: bool = True):
        """
        Create animated GIF showing quantum state evolution.
        
        Args:
            states: List of quantum states
            title: Animation title
            mode: Mode to visualize
            save: Save the animation
        """
        print(f"🎬 Creating quantum state evolution animation...")
        
        try:
            from matplotlib.animation import FuncAnimation
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Prepare coordinate grid
            x = np.linspace(-4, 4, 80)
            p = np.linspace(-4, 4, 80)
            X, P = np.meshgrid(x, p)
            
            def animate(frame):
                ax.clear()
                state = states[frame]
                W = state.wigner(mode, x, p)
                
                surface = ax.plot_surface(X, P, W, cmap='RdYlBu_r', alpha=0.9)
                ax.set_title(f'{title} - Frame {frame+1}/{len(states)}')
                ax.set_xlabel('Position (x)')
                ax.set_ylabel('Momentum (p)')
                ax.set_zlabel('Wigner Function')
                
                return surface,
            
            anim = FuncAnimation(fig, animate, frames=len(states), interval=500, blit=False)
            
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.save_dir}/evolution_{title.lower().replace(' ', '_')}_{timestamp}.gif"
                anim.save(filename, writer='pillow', fps=2)
                logger.info(f"State evolution animation saved to {filename}")
            
            plt.show()
            
        except Exception as e:
            logger.warning(f"Could not create state evolution animation: {e}")
    
    # =====================================================================
    # INTEGRATION METHODS FOR PURE SF MODELS
    # =====================================================================
    
    def integrate_with_generator(self, generator, title: str = "Generator"):
        """
        Integrate visualization with Pure SF generator.
        
        Args:
            generator: PureSFGenerator instance
            title: Visualization title
        """
        print(f"🔗 Integrating visualization with {title}...")
        
        # Visualize generator circuit
        if hasattr(generator, 'circuit'):
            self.visualize_pure_sf_circuit(generator.circuit, f"{title} Circuit")
        
        # Generate and visualize sample states
        if hasattr(generator, 'generate'):
            try:
                # Generate samples
                z_sample = tf.random.normal([1, generator.latent_dim])
                generated_samples = generator.generate(z_sample)
                
                print(f"Generated samples shape: {generated_samples.shape}")
                print(f"Sample values: {generated_samples.numpy()}")
                
                # If generator has method to get quantum state, visualize it
                if hasattr(generator, 'get_quantum_state'):
                    state = generator.get_quantum_state(z_sample)
                    self.visualize_quantum_state(state, f"{title} Generated State")
                
            except Exception as e:
                logger.warning(f"Could not generate samples for visualization: {e}")
    
    def integrate_with_discriminator(self, discriminator, title: str = "Discriminator"):
        """
        Integrate visualization with Pure SF discriminator.
        
        Args:
            discriminator: PureSFDiscriminator instance  
            title: Visualization title
        """
        print(f"🔗 Integrating visualization with {title}...")
        
        # Visualize discriminator circuit
        if hasattr(discriminator, 'circuit'):
            self.visualize_pure_sf_circuit(discriminator.circuit, f"{title} Circuit")
        
        # Test discriminator with sample data
        if hasattr(discriminator, 'discriminate'):
            try:
                # Create test samples
                test_samples = tf.random.normal([4, discriminator.input_dim])
                outputs = discriminator.discriminate(test_samples)
                
                print(f"Discriminator test:")
                print(f"  Input shape: {test_samples.shape}")
                print(f"  Output shape: {outputs.shape}")
                print(f"  Output values: {outputs.numpy()}")
                
                # If discriminator has method to get quantum state, visualize it
                if hasattr(discriminator, 'get_quantum_state'):
                    state = discriminator.get_quantum_state(test_samples)
                    self.visualize_quantum_state(state, f"{title} Processing State")
                
            except Exception as e:
                logger.warning(f"Could not test discriminator for visualization: {e}")
    
    def create_qgan_comparison_dashboard(self, generator, discriminator, 
                                       real_data: tf.Tensor,
                                       n_samples: int = 100,
                                       title: str = "QGAN Analysis"):
        """
        Create comprehensive QGAN comparison dashboard.
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            real_data: Real training data
            n_samples: Number of samples to generate
            title: Dashboard title
        """
        print(f"🎯 Creating {title} comparison dashboard...")
        
        try:
            # Generate samples
            z_batch = tf.random.normal([n_samples, generator.latent_dim])
            generated_samples = generator.generate(z_batch)
            
            # Get discriminator scores
            real_scores = discriminator.discriminate(real_data[:n_samples])
            fake_scores = discriminator.discriminate(generated_samples)
            
            # Create comprehensive comparison
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Sample distribution comparison
            axes[0, 0].scatter(real_data[:n_samples, 0], real_data[:n_samples, 1], 
                              alpha=0.6, color='blue', s=20, label='Real Data')
            axes[0, 0].scatter(generated_samples[:, 0], generated_samples[:, 1], 
                              alpha=0.6, color='red', s=20, label='Generated')
            axes[0, 0].set_title('Data Distribution Comparison')
            axes[0, 0].set_xlabel('Feature 1')
            axes[0, 0].set_ylabel('Feature 2')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Discriminator score distributions
            axes[0, 1].hist(real_scores.numpy(), bins=20, alpha=0.7, color='blue', 
                           label=f'Real (μ={np.mean(real_scores):.3f})')
            axes[0, 1].hist(fake_scores.numpy(), bins=20, alpha=0.7, color='red',
                           label=f'Fake (μ={np.mean(fake_scores):.3f})')
            axes[0, 1].set_title('Discriminator Score Distribution')
            axes[0, 1].set_xlabel('Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Score scatter plot
            axes[0, 2].scatter(range(len(real_scores)), real_scores, alpha=0.6, 
                              color='blue', label='Real', s=15)
            axes[0, 2].scatter(range(len(fake_scores)), fake_scores, alpha=0.6, 
                              color='red', label='Fake', s=15)
            axes[0, 2].set_title('Individual Sample Scores')
            axes[0, 2].set_xlabel('Sample Index')
            axes[0, 2].set_ylabel('Discriminator Score')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Feature statistics comparison
            real_mean = np.mean(real_data[:n_samples], axis=0)
            real_std = np.std(real_data[:n_samples], axis=0)
            fake_mean = np.mean(generated_samples, axis=0)
            fake_std = np.std(generated_samples, axis=0)
            
            feature_indices = range(len(real_mean))
            width = 0.35
            axes[1, 0].bar([i - width/2 for i in feature_indices], real_mean, width, 
                          alpha=0.7, color='blue', label='Real Mean')
            axes[1, 0].bar([i + width/2 for i in feature_indices], fake_mean, width,
                          alpha=0.7, color='red', label='Fake Mean')
            axes[1, 0].set_title('Feature Means Comparison')
            axes[1, 0].set_xlabel('Feature Index')
            axes[1, 0].set_ylabel('Mean Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Standard deviation comparison
            axes[1, 1].bar([i - width/2 for i in feature_indices], real_std, width,
                          alpha=0.7, color='blue', label='Real Std')
            axes[1, 1].bar([i + width/2 for i in feature_indices], fake_std, width,
                          alpha=0.7, color='red', label='Fake Std')
            axes[1, 1].set_title('Feature Standard Deviations')
            axes[1, 1].set_xlabel('Feature Index')
            axes[1, 1].set_ylabel('Standard Deviation')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Quality metrics
            wasserstein_approx = np.mean(real_scores) - np.mean(fake_scores)
            mse_means = np.mean((real_mean - fake_mean)**2)
            mse_stds = np.mean((real_std - fake_std)**2)
            
            metrics_text = f"""Quality Metrics:
Wasserstein Distance: {wasserstein_approx:.4f}
MSE (Means): {mse_means:.4f}  
MSE (Stds): {mse_stds:.4f}

Sample Statistics:
Real samples: {len(real_data)}
Generated samples: {n_samples}
Discriminator accuracy: {np.mean((real_scores > 0).numpy()) * 50 + np.mean((fake_scores < 0).numpy()) * 50:.1f}%

Model Info:
Generator params: {len(generator.trainable_variables)}
Discriminator params: {len(discriminator.trainable_variables)}"""
            
            axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes,
                            fontsize=10, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            axes[1, 2].set_title('Quality Assessment')
            axes[1, 2].axis('off')
            
            plt.suptitle(f'{title} - Comprehensive Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.save_dir}/qgan_analysis_{title.lower().replace(' ', '_')}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"QGAN analysis saved to {filename}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Could not create QGAN comparison dashboard: {e}")
    
    # =====================================================================
    # CONVENIENCE METHODS
    # =====================================================================
    
    def quick_circuit_viz(self, circuit, title: str = "Circuit"):
        """Quick circuit visualization."""
        self.visualize_pure_sf_circuit(circuit, title, show_parameters=True, show_values=True)
    
    def quick_state_viz(self, state, title: str = "State"):
        """Quick quantum state visualization."""
        self.visualize_quantum_state(state, title, modes=[0], save=False)
    
    def get_visualization_summary(self) -> Dict[str, Any]:
        """Get summary of visualization capabilities."""
        return {
            'features': [
                'Pure SF circuit structure analysis',
                '3D Wigner function mountains',  
                'Fock probability distributions',
                'Quadrature distribution analysis',
                'Quantum state entropy analysis',
                'Training progress dashboards',
                'Parameter evolution tracking',
                'QGAN comparison analysis',
                'State evolution animations'
            ],
            'save_directory': self.save_dir,
            'tracked_parameters': len(self.parameter_history),
            'tracked_states': len(self.state_history),
            'visualization_count': len(self.visualization_history)
        }


# =====================================================================
# CONVENIENCE FUNCTIONS
# =====================================================================

def create_visualization_manager(save_dir: str = "visualizations") -> QuantumVisualizationManager:
    """Create and return a visualization manager instance."""
    return QuantumVisualizationManager(save_dir)

def quick_circuit_visualization(circuit, title: str = "Circuit"):
    """Quick circuit visualization function."""
    viz = QuantumVisualizationManager()
    viz.quick_circuit_viz(circuit, title)

def quick_state_visualization(state, title: str = "State"):
    """Quick state visualization function.""" 
    viz = QuantumVisualizationManager()
    viz.quick_state_viz(state, title)
