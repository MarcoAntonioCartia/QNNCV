"""
Quantum State Visualizer - Wigner Functions and State Analysis

This module provides specialized visualization for quantum states including:
- 3D Wigner function mountains
- Fock probability distributions  
- State vector analysis
- Quantum state purity and entropy visualization

Extracted from quantum_visualization_manager.py to focus on the valuable
quantum state visualization capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import warnings
import os

logger = logging.getLogger(__name__)


class QuantumStateVisualizer:
    """
    Specialized visualizer for quantum states with focus on Wigner functions.
    
    Features:
    - 3D Wigner function mountains (SF compatible)
    - Fock probability distributions
    - State vector analysis
    - Quantum state properties
    """
    
    def __init__(self, save_directory: str = "quantum_state_visualizations"):
        """Initialize the quantum state visualizer."""
        self.save_dir = save_directory
        
        # Create save directory
        os.makedirs(save_directory, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        warnings.filterwarnings('ignore', category=UserWarning)
        
        logger.info(f"QuantumStateVisualizer initialized")
        logger.info(f"  Save directory: {save_directory}")
    
    def visualize_quantum_state(self, state, title: str = "Quantum State",
                               modes: Optional[List[int]] = None,
                               save: bool = True):
        """
        Complete quantum state visualization with Wigner functions.
        
        Args:
            state: Quantum state from Strawberry Fields
            title: Title for the visualization
            modes: List of modes to visualize (default: first 2 modes)
            save: Whether to save the visualization
        """
        if modes is None:
            modes = list(range(min(state.num_modes, 2)))  # Limit for visualization
        
        print(f"🎭 {title.upper()} - QUANTUM STATE ANALYSIS")
        print("=" * 60)
        print(f"State Configuration:")
        print(f"  Modes: {state.num_modes}")
        print(f"  Visualizing modes: {modes}")
        
        try:
            # 1. 3D Wigner function visualization
            self.create_wigner_mountains(state, modes, title, save)
            
            # 2. Fock probability analysis
            self.analyze_fock_probabilities(state, modes, title, save)
            
            # 3. State vector analysis
            self.analyze_state_vector(state, title, save)
            
        except Exception as e:
            logger.error(f"State visualization failed: {e}")
    
    def create_wigner_mountains(self, state, modes: List[int], title: str, save: bool):
        """Create 3D Wigner function visualizations (SF compatible)."""
        print(f"\n🏔️  Creating 3D Wigner function mountains...")
        
        try:
            n_modes = len(modes)
            cols = min(2, n_modes)
            rows = (n_modes + cols - 1) // cols
            
            fig = plt.figure(figsize=(8*cols, 6*rows))
            
            # Create coordinate grids
            x = np.linspace(-4, 4, 60)  # Reduced resolution for performance
            p = np.linspace(-4, 4, 60)
            X, P = np.meshgrid(x, p)
            
            for i, mode in enumerate(modes):
                if mode >= state.num_modes:
                    continue
                    
                ax = fig.add_subplot(rows, cols, i+1, projection='3d')
                
                try:
                    # Calculate Wigner function - SF supports this!
                    W = state.wigner(mode, x, p)
                    
                    # Create 3D surface
                    surface = ax.plot_surface(X, P, W, cmap='RdYlBu_r',
                                            alpha=0.9, linewidth=0.3,
                                            rstride=2, cstride=2)
                    
                    # Add contour lines at base
                    contour_levels = np.linspace(np.min(W), np.max(W), 8)
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
                    # Create placeholder
                    ax.text(0.5, 0.5, 0.5, f'Mode {mode}\nWigner N/A', 
                           ha='center', va='center', transform=ax.transData)
            
            plt.suptitle(f'{title} - Wigner Function Mountains', fontsize=16)
            plt.tight_layout()
            
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.save_dir}/wigner_{title.lower().replace(' ', '_')}_{timestamp}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"Wigner functions saved to {filename}")
            
            plt.show()
            
        except Exception as e:
            logger.warning(f"Could not create Wigner visualizations: {e}")
    
    def analyze_fock_probabilities(self, state, modes: List[int], title: str, save: bool):
        """Analyze Fock probabilities using SF state methods."""
        print(f"\n📊 Analyzing Fock state probabilities...")
        
        try:
            # Get state vector (this works in SF)
            ket = state.ket()
            prob_amplitudes = tf.abs(ket) ** 2
            
            # For single mode, we can extract marginal probabilities
            if len(modes) == 1 and state.num_modes == 1:
                mode = modes[0]
                cutoff = len(prob_amplitudes)
                
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                
                # Plot Fock probabilities
                fock_probs = prob_amplitudes.numpy()
                bars = ax.bar(range(cutoff), fock_probs, alpha=0.7, color='skyblue')
                ax.set_title(f'{title} - Fock State Probabilities')
                ax.set_xlabel('Fock Number |n⟩')
                ax.set_ylabel('Probability')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for i, (bar, prob) in enumerate(zip(bars, fock_probs)):
                    if prob > 0.01:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                               f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
                
                # Calculate mean photon number
                mean_n = sum(n * prob for n, prob in enumerate(fock_probs))
                ax.axvline(mean_n, color='red', linestyle='--', alpha=0.8, 
                          label=f'⟨n⟩={mean_n:.2f}')
                ax.legend()
                
                print(f"    Mode {mode}: ⟨n⟩={mean_n:.3f}, max_prob={max(fock_probs):.3f}")
                
                if save:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{self.save_dir}/fock_{title.lower().replace(' ', '_')}_{timestamp}.png"
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                
                plt.show()
            else:
                print("    Multi-mode Fock analysis not implemented (requires state tomography)")
                
        except Exception as e:
            logger.warning(f"Could not analyze Fock probabilities: {e}")
    
    def analyze_state_vector(self, state, title: str, save: bool):
        """Analyze state vector using SF methods."""
        print(f"\n🔬 Analyzing state vector...")
        
        try:
            ket = state.ket()
            prob_amplitudes = tf.abs(ket) ** 2
            
            # State vector analysis
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Amplitude magnitudes
            amplitudes = tf.abs(ket).numpy()
            axes[0].plot(amplitudes, 'o-', alpha=0.7, markersize=4)
            axes[0].set_title('State Amplitude Magnitudes')
            axes[0].set_xlabel('Basis State Index')
            axes[0].set_ylabel('|⟨n|ψ⟩|')
            axes[0].grid(True, alpha=0.3)
            
            # Probability distribution
            probs = prob_amplitudes.numpy()
            axes[1].bar(range(len(probs)), probs, alpha=0.7, color='coral')
            axes[1].set_title('Probability Distribution')
            axes[1].set_xlabel('Basis State Index')
            axes[1].set_ylabel('|⟨n|ψ⟩|²')
            axes[1].grid(True, alpha=0.3)
            
            # State statistics
            purity = tf.reduce_sum(prob_amplitudes ** 2).numpy()
            entropy = -tf.reduce_sum(prob_amplitudes * tf.math.log(prob_amplitudes + 1e-12)).numpy()
            participation = 1.0 / tf.reduce_sum(prob_amplitudes ** 2).numpy()
            
            stats_text = f"""State Statistics:
Norm: {tf.norm(ket).numpy():.4f}
Purity: {purity:.4f}
Entropy: {entropy:.4f}
Participation Ratio: {participation:.2f}

Basis States: {len(ket)}
Non-zero Components: {np.sum(probs > 1e-6)}
Max Probability: {np.max(probs):.4f}"""
            
            axes[2].text(0.1, 0.9, stats_text, transform=axes[2].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[2].set_title('State Properties')
            axes[2].axis('off')
            
            plt.suptitle(f'{title} - State Vector Analysis', fontsize=14)
            plt.tight_layout()
            
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.save_dir}/state_vector_{title.lower().replace(' ', '_')}_{timestamp}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            logger.warning(f"Could not analyze state vector: {e}")
    
    def quick_wigner_viz(self, state, mode: int = 0, title: str = "Wigner Function"):
        """Quick Wigner function visualization for a single mode."""
        self.create_wigner_mountains(state, [mode], title, save=False)
    
    def quick_state_analysis(self, state, title: str = "Quantum State"):
        """Quick complete state analysis."""
        self.visualize_quantum_state(state, title, save=False)


# Convenience functions
def visualize_wigner_function(state, mode: int = 0, title: str = "Wigner Function", 
                             save_dir: str = "quantum_visualizations"):
    """Quick Wigner function visualization."""
    viz = QuantumStateVisualizer(save_dir)
    viz.quick_wigner_viz(state, mode, title)

def analyze_quantum_state(state, title: str = "Quantum State", 
                         save_dir: str = "quantum_visualizations"):
    """Quick quantum state analysis."""
    viz = QuantumStateVisualizer(save_dir)
    viz.quick_state_analysis(state, title)


def demo_quantum_state_visualizer():
    """Demonstrate the quantum state visualizer."""
    print("🚀 QUANTUM STATE VISUALIZER DEMO")
    print("="*50)
    
    try:
        # Create visualization manager
        viz = QuantumStateVisualizer("demo_quantum_states")
        
        print("✅ Quantum state visualizer demo setup completed!")
        print("Note: Requires actual quantum state from Strawberry Fields for full demo")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    demo_quantum_state_visualizer()
