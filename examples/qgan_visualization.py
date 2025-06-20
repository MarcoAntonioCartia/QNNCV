"""
Enhanced Quantum GAN Visualization System
=========================================

Comprehensive visualization tools for quantum circuits and states using
Strawberry Fields native capabilities plus custom enhancements.

Features:
- Circuit diagram printing with parameter details
- 3D Wigner function "mountain maps" for all modes
- Multi-mode state visualization
- Training progress visualization
- Parameter evolution tracking
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import strawberryfields as sf
from strawberryfields import ops
import tensorflow as tf
import logging
from typing import List, Dict, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.subplots as sp
from datetime import datetime
import seaborn as sns

logger = logging.getLogger(__name__)


class QuantumGANVisualizer:
    """Comprehensive visualization system for Quantum GANs."""
    
    def __init__(self, qgan_model=None):
        """
        Initialize visualizer.
        
        Args:
            qgan_model: Your QGAN model instance (optional)
        """
        self.qgan = qgan_model
        self.visualization_history = []
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def print_circuit_diagram(self, circuit_or_program, show_parameters=True, show_values=False):
        """
        Print detailed quantum circuit diagram.
        
        Args:
            circuit_or_program: SF Program or your PureQuantumCircuit
            show_parameters: Show parameter names
            show_values: Show parameter values
        """
        print("🔬 QUANTUM CIRCUIT DIAGRAM")
        print("=" * 60)
        
        if hasattr(circuit_or_program, 'build_program'):
            # It's our PureQuantumCircuit
            prog = circuit_or_program.build_program({})
            trainable_vars = circuit_or_program.trainable_variables
            
            print(f"Circuit Configuration:")
            print(f"  Modes: {circuit_or_program.n_modes}")
            print(f"  Layers: {circuit_or_program.layers}")
            print(f"  Parameters: {len(trainable_vars)}")
            print(f"  Cutoff: {circuit_or_program.cutoff_dim}")
            
        else:
            # It's an SF Program
            prog = circuit_or_program
            trainable_vars = []
        
        print("\n📋 Circuit Operations:")
        print("-" * 40)
        
        # Use SF's built-in print method
        prog.print()
        
        if show_parameters and trainable_vars:
            print(f"\n🎛️  Trainable Parameters ({len(trainable_vars)}):")
            print("-" * 40)
            for i, var in enumerate(trainable_vars):
                param_name = var.name.split(':')[0]
                if show_values:
                    value = float(var.numpy())
                    print(f"  {i:2d}. {param_name:<20} = {value:8.4f}")
                else:
                    print(f"  {i:2d}. {param_name}")
    
    def visualize_state_3d_mountain(self, state, mode=0, x_range=(-4, 4), p_range=(-4, 4), 
                                  resolution=100, title_suffix=""):
        """
        Create 3D "mountain map" visualization of Wigner function.
        
        Args:
            state: SF quantum state object
            mode: Mode to visualize
            x_range: X quadrature range
            p_range: P quadrature range
            resolution: Grid resolution
            title_suffix: Additional title text
        """
        # Create coordinate arrays
        x = np.linspace(x_range[0], x_range[1], resolution)
        p = np.linspace(p_range[0], p_range[1], resolution)
        X, P = np.meshgrid(x, p)
        
        # Calculate Wigner function
        W = state.wigner(mode, x, p)
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create mountain surface
        surface = ax.plot_surface(X, P, W, cmap='RdYlBu_r', 
                                alpha=0.9, linewidth=0.5, 
                                rstride=2, cstride=2)
        
        # Add contour lines at the base
        ax.contour(X, P, W, levels=15, colors='gray', 
                  alpha=0.6, offset=np.min(W)-0.05)
        
        # Styling
        ax.set_xlabel('Position (x)', fontsize=12, labelpad=10)
        ax.set_ylabel('Momentum (p)', fontsize=12, labelpad=10)
        ax.set_zlabel('Wigner Function', fontsize=12, labelpad=10)
        ax.set_title(f'Quantum State Wigner Function - Mode {mode}{title_suffix}', 
                    fontsize=14, pad=20)
        
        # Add colorbar
        fig.colorbar(surface, shrink=0.6, aspect=20, pad=0.1)
        
        # Clean up axes
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig, ax
    
    def visualize_all_modes_mountain(self, state, x_range=(-4, 4), p_range=(-4, 4), 
                                   resolution=80, title_suffix=""):
        """
        Create mountain map visualization for all modes in a single figure.
        
        Args:
            state: SF quantum state object  
            x_range: X quadrature range
            p_range: P quadrature range
            resolution: Grid resolution
            title_suffix: Additional title text
        """
        n_modes = state.num_modes
        
        # Calculate grid layout
        cols = min(3, n_modes)
        rows = (n_modes + cols - 1) // cols
        
        fig = plt.figure(figsize=(5*cols, 4*rows))
        
        # Create coordinate arrays
        x = np.linspace(x_range[0], x_range[1], resolution)
        p = np.linspace(p_range[0], p_range[1], resolution)
        X, P = np.meshgrid(x, p)
        
        for mode in range(n_modes):
            ax = fig.add_subplot(rows, cols, mode + 1, projection='3d')
            
            # Calculate Wigner function for this mode
            W = state.wigner(mode, x, p)
            
            # Create surface
            surface = ax.plot_surface(X, P, W, cmap='RdYlBu_r', 
                                    alpha=0.9, linewidth=0.3,
                                    rstride=2, cstride=2)
            
            # Add contours
            ax.contour(X, P, W, levels=10, colors='gray', 
                      alpha=0.5, offset=np.min(W)-0.02)
            
            # Styling
            ax.set_title(f'Mode {mode}', fontsize=12, pad=10)
            ax.set_xlabel('x', fontsize=10)
            ax.set_ylabel('p', fontsize=10)
            ax.set_zlabel('W', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Make axes labels smaller for space
            ax.tick_params(axis='both', which='major', labelsize=8)
        
        fig.suptitle(f'Multi-Mode Quantum State Visualization{title_suffix}', 
                    fontsize=16, y=0.95)
        plt.tight_layout()
        
        return fig
    
    def visualize_state_with_plotly(self, state, mode=0, x_range=(-4, 4), 
                                  p_range=(-4, 4), resolution=100):
        """
        Interactive 3D Wigner function visualization using Plotly.
        
        Args:
            state: SF quantum state object
            mode: Mode to visualize  
            x_range: X quadrature range
            p_range: P quadrature range
            resolution: Grid resolution
        """
        # Create coordinate arrays
        x = np.linspace(x_range[0], x_range[1], resolution)
        p = np.linspace(p_range[0], p_range[1], resolution)
        
        # Calculate Wigner function
        W = state.wigner(mode, x, p)
        
        # Create interactive 3D surface
        fig = go.Figure(data=[go.Surface(
            x=x, y=p, z=W,
            colorscale='RdYlBu_r',
            opacity=0.9,
            contours={
                "z": {"show": True, "start": np.min(W), "end": np.max(W), "size": 0.02}
            }
        )])
        
        fig.update_layout(
            title=f'Interactive Wigner Function - Mode {mode}',
            scene=dict(
                xaxis_title='Position (x)',
                yaxis_title='Momentum (p)', 
                zaxis_title='Wigner Function',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800, height=600
        )
        
        return fig
    
    def visualize_fock_probabilities(self, state, modes=None, cutoff=None):
        """
        Visualize Fock state probabilities for specified modes.
        
        Args:
            state: SF quantum state object
            modes: List of modes to plot (default: all)
            cutoff: Fock space cutoff (default: auto-detect)
        """
        if modes is None:
            modes = list(range(state.num_modes))
        
        if cutoff is None:
            cutoff = state.cutoff_dim if hasattr(state, 'cutoff_dim') else 10
        
        # Use SF's built-in plot function
        try:
            sf.plot.plot_fock(state, modes, cutoff=cutoff, renderer="browser")
        except Exception as e:
            logger.warning(f"SF plot_fock failed: {e}, using custom implementation")
            self._custom_fock_plot(state, modes, cutoff)
    
    def _custom_fock_plot(self, state, modes, cutoff):
        """Custom Fock probability visualization."""
        n_modes = len(modes)
        fig, axes = plt.subplots(1, n_modes, figsize=(4*n_modes, 4))
        if n_modes == 1:
            axes = [axes]
        
        for i, mode in enumerate(modes):
            probs = []
            for n in range(cutoff):
                fock_state = [0] * state.num_modes
                fock_state[mode] = n
                prob = state.fock_prob(fock_state)
                probs.append(prob)
            
            axes[i].bar(range(cutoff), probs, alpha=0.7)
            axes[i].set_title(f'Mode {mode} Fock Probabilities')
            axes[i].set_xlabel('Fock Number')
            axes[i].set_ylabel('Probability')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def visualize_training_progress(self, training_history, save_path=None):
        """
        Visualize QGAN training progress.
        
        Args:
            training_history: Dictionary with training metrics
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        if 'g_loss' in training_history and 'd_loss' in training_history:
            axes[0, 0].plot(training_history['g_loss'], label='Generator', color='blue')
            axes[0, 0].plot(training_history['d_loss'], label='Discriminator', color='red')
            axes[0, 0].set_title('Training Losses')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Wasserstein distance
        if 'w_distance' in training_history:
            axes[0, 1].plot(training_history['w_distance'], color='green')
            axes[0, 1].set_title('Wasserstein Distance')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Distance')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Gradient penalty
        if 'gradient_penalty' in training_history:
            axes[1, 0].plot(training_history['gradient_penalty'], color='purple')
            axes[1, 0].set_title('Gradient Penalty')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Penalty')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Gradient norms
        if 'g_gradients' in training_history:
            axes[1, 1].plot(training_history['g_gradients'], label='Generator', color='blue')
            if 'd_gradients' in training_history:
                axes[1, 1].plot(training_history['d_gradients'], label='Discriminator', color='red')
            axes[1, 1].set_title('Gradient Norms')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Norm')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Quantum GAN Training Progress', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_parameter_evolution(self, parameter_history, parameter_names=None):
        """
        Visualize how quantum parameters evolve during training.
        
        Args:
            parameter_history: List of parameter snapshots during training
            parameter_names: Names of parameters (optional)
        """
        if not parameter_history:
            logger.warning("No parameter history provided")
            return None
        
        n_params = len(parameter_history[0])
        n_epochs = len(parameter_history)
        
        # Create parameter evolution matrix
        param_matrix = np.array(parameter_history).T  # Shape: (n_params, n_epochs)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(param_matrix, cmap='RdBu_r', aspect='auto', interpolation='bilinear')
        
        # Set labels
        ax.set_title('Quantum Parameter Evolution During Training', fontsize=14, pad=20)
        ax.set_xlabel('Training Epoch', fontsize=12)
        ax.set_ylabel('Parameter Index', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Parameter Value', fontsize=12)
        
        # Set ticks
        epoch_ticks = np.linspace(0, n_epochs-1, min(10, n_epochs)).astype(int)
        ax.set_xticks(epoch_ticks)
        ax.set_xticklabels(epoch_ticks)
        
        param_ticks = np.arange(0, n_params, max(1, n_params//20))
        ax.set_yticks(param_ticks)
        
        if parameter_names:
            ax.set_yticklabels([parameter_names[i] for i in param_ticks], fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def demonstrate_qgan_visualization(self, qgan_model=None):
        """
        Demonstrate all visualization capabilities with a simple example.
        
        Args:
            qgan_model: Your QGAN model (optional)
        """
        print("🎨 QUANTUM GAN VISUALIZATION DEMONSTRATION")
        print("=" * 60)
        
        if qgan_model is not None:
            print("Using provided QGAN model...")
            # Visualize actual QGAN
            
            # 1. Circuit diagrams
            print("\n1. 📋 Generator Circuit:")
            self.print_circuit_diagram(qgan_model.generator.circuit, 
                                     show_parameters=True, show_values=True)
            
            print("\n2. 📋 Discriminator Circuit:")  
            self.print_circuit_diagram(qgan_model.discriminator.circuit,
                                     show_parameters=True, show_values=True)
            
            # 3. Generate samples and visualize states
            print("\n3. 🎭 Generated Sample Visualization:")
            z = tf.random.normal([1, qgan_model.generator.latent_dim])
            
            # This would need to be adapted to your specific QGAN interface
            # samples = qgan_model.generator.generate(z)
            
        else:
            print("Creating demonstration with synthetic quantum state...")
            
            # Create a demo quantum state for visualization
            prog = sf.Program(3)
            with prog.context as q:
                ops.Sgate(0.8) | q[0]
                ops.Sgate(1.2) | q[1] 
                ops.Dgate(0.5+0.3j) | q[2]
                ops.BSgate(0.7, 0.2) | (q[0], q[1])
                ops.BSgate(0.4, -0.1) | (q[1], q[2])
            
            # Execute on Fock backend
            eng = sf.Engine("fock", backend_options={"cutoff_dim": 8})
            result = eng.run(prog)
            state = result.state
            
            print("\n1. 📋 Demo Circuit:")
            self.print_circuit_diagram(prog)
            
            print("\n2. 🏔️  Single Mode Mountain Visualization:")
            fig1 = self.visualize_state_3d_mountain(state, mode=0, title_suffix=" (Demo)")
            plt.show()
            
            print("\n3. 🗻 All Modes Mountain Visualization:")
            fig2 = self.visualize_all_modes_mountain(state, title_suffix=" (Demo)")
            plt.show()
            
            print("\n4. 📊 Fock Probability Visualization:")
            self._custom_fock_plot(state, [0, 1, 2], cutoff=6)
            plt.show()
            
            # 5. Demo training progress
            print("\n5. 📈 Training Progress Visualization (Synthetic Data):")
            demo_history = {
                'g_loss': np.random.exponential(1, 50) + np.linspace(2, 0.5, 50),
                'd_loss': np.random.exponential(1, 50) + np.linspace(1.5, 0.8, 50),
                'w_distance': np.random.normal(0, 0.1, 50) + np.exp(-np.linspace(0, 3, 50)),
                'gradient_penalty': np.random.exponential(0.1, 50),
                'g_gradients': np.random.exponential(0.5, 50),
                'd_gradients': np.random.exponential(0.3, 50)
            }
            
            fig3 = self.visualize_training_progress(demo_history)
            plt.show()
            
            print("\n🎉 Visualization demonstration complete!")
            print("Available methods:")
            print("  - print_circuit_diagram()")
            print("  - visualize_state_3d_mountain()")  
            print("  - visualize_all_modes_mountain()")
            print("  - visualize_state_with_plotly()")
            print("  - visualize_training_progress()")
            print("  - visualize_parameter_evolution()")


# Convenience functions for easy access
def visualize_circuit(circuit, show_params=True, show_values=False):
    """Quick circuit visualization."""
    viz = QuantumGANVisualizer()
    viz.print_circuit_diagram(circuit, show_params, show_values)

def create_mountain_plot(state, mode=0, title=""):
    """Quick 3D mountain plot."""
    viz = QuantumGANVisualizer()
    return viz.visualize_state_3d_mountain(state, mode, title_suffix=title)

def create_all_modes_plot(state, title=""):
    """Quick all-modes visualization."""
    viz = QuantumGANVisualizer()
    return viz.visualize_all_modes_mountain(state, title_suffix=title)


# Demo usage
if __name__ == "__main__":
    visualizer = QuantumGANVisualizer()
    visualizer.demonstrate_qgan_visualization()
