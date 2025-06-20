"""
ENHANCED Corrected Quantum Visualization Manager for Pure SF Quantum GANs

This enhanced version combines the corrected base with proven SF-compatible features:
1. Proper SF state methods compatibility
2. Individual gate parameter analysis
3. 3D Wigner function visualization (SF actually supports this!)
4. Enhanced training dashboards
5. QGAN comparison functionality
6. Robust error handling throughout
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


class CorrectedQuantumVisualizationManager:
    """
    ENHANCED Corrected visualization manager for Pure SF Quantum GANs.
    
    Fixed to work properly with:
    - Strawberry Fields actual API
    - Pure quantum circuit architecture  
    - Individual gate parameters
    - Static transformation matrices
    + Added proven SF-compatible enhancements
    """
    
    def __init__(self, save_directory: str = "visualizations"):
        """Initialize the enhanced corrected visualization manager."""
        self.save_dir = save_directory
        self.visualization_history = []
        self.parameter_history = []
        self.state_history = []
        self.measurement_history = []
        
        # Create save directory
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')  # More compatible
        warnings.filterwarnings('ignore', category=UserWarning)
        
        logger.info(f"Enhanced Corrected QuantumVisualizationManager initialized")
        logger.info(f"  Save directory: {save_directory}")
    
    # =====================================================================
    # FIXED PURE SF CIRCUIT VISUALIZATION
    # =====================================================================
    
    def visualize_pure_sf_circuit(self, circuit, title: str = "Pure SF Circuit", 
                                 show_parameters: bool = True, 
                                 show_values: bool = True,
                                 save: bool = True):
        """
        FIXED: Visualize Pure SF circuit structure with proper parameter analysis.
        
        Compatible with PureQuantumCircuitCorrected architecture.
        """
        print(f"🔬 {title.upper()}")
        print("=" * 60)
        
        # Circuit configuration - FIXED to match your pure quantum architecture
        print(f"Circuit Configuration:")
        print(f"  Architecture: Pure SF with Individual Gate Parameters")
        print(f"  Modes: {circuit.n_modes}")
        print(f"  Layers: {circuit.n_layers}")
        print(f"  Cutoff: {circuit.cutoff_dim}")
        
        # FIXED: Check for trainable_variables (individual gate parameters)
        if hasattr(circuit, 'trainable_variables'):
            print(f"  Individual Parameters: {len(circuit.trainable_variables)}")
            
            # FIXED: Parameter analysis for individual gate parameters
            if show_parameters:
                self._analyze_individual_gate_parameters(circuit, show_values)
        
        # FIXED: Circuit structure analysis using param_names
        if hasattr(circuit, 'param_names') and hasattr(circuit, 'gate_params'):
            self._visualize_circuit_structure(circuit)
        
        # FIXED: Parameter distribution analysis
        if hasattr(circuit, 'trainable_variables') and len(circuit.trainable_variables) > 0:
            self._visualize_parameter_distribution(circuit, title, save)
    
    def _analyze_individual_gate_parameters(self, circuit, show_values: bool = True):
        """FIXED: Analyze individual gate parameters properly."""
        print(f"\n🎛️  Individual Gate Parameter Analysis:")
        print("-" * 40)
        
        # Group parameters by gate type
        param_groups = {}
        
        if hasattr(circuit, 'param_names') and hasattr(circuit, 'gate_params'):
            for i, (param_name, tf_var) in enumerate(zip(circuit.param_names, circuit.gate_params)):
                # Parse parameter name: L{layer}_{gate_type}_{param_type}_{index}
                parts = param_name.split('_')
                if len(parts) >= 3:
                    gate_type = parts[1]  # BS1, SQUEEZE, DISP, etc.
                    
                    if gate_type not in param_groups:
                        param_groups[gate_type] = []
                    
                    if show_values:
                        value = float(tf_var.numpy())
                        param_groups[gate_type].append((param_name, value))
                        print(f"  {i:2d}. {param_name:<30} = {value:8.4f}")
                    else:
                        param_groups[gate_type].append((param_name, None))
                        print(f"  {i:2d}. {param_name}")
        elif hasattr(circuit, 'trainable_variables'):
            # Fallback for standard trainable variables
            for i, var in enumerate(circuit.trainable_variables):
                param_name = var.name.split(':')[0]
                if show_values:
                    value = float(var.numpy())
                    print(f"  {i:2d}. {param_name:<30} = {value:8.4f}")
                else:
                    print(f"  {i:2d}. {param_name}")
        
        # Parameter type summary
        if param_groups:
            print(f"\n📊 Gate Type Summary:")
            for gate_type, params in param_groups.items():
                print(f"  {gate_type}: {len(params)} parameters")
    
    def _visualize_circuit_structure(self, circuit):
        """FIXED: Visualize circuit structure using actual parameter names."""
        print(f"\n📋 Circuit Structure Analysis:")
        print("-" * 40)
        
        if hasattr(circuit, 'param_names'):
            # Analyze structure by layers
            layer_structure = {}
            
            for param_name in circuit.param_names:
                parts = param_name.split('_')
                if len(parts) >= 2:
                    layer = parts[0]  # L0, L1, etc.
                    gate_type = parts[1]  # BS1, SQUEEZE, etc.
                    
                    if layer not in layer_structure:
                        layer_structure[layer] = {}
                    
                    if gate_type not in layer_structure[layer]:
                        layer_structure[layer][gate_type] = 0
                    
                    layer_structure[layer][gate_type] += 1
            
            # Print layer-by-layer structure
            for layer, gates in layer_structure.items():
                print(f"  {layer}:")
                for gate_type, count in gates.items():
                    print(f"    {gate_type}: {count} parameters")
    
    def _visualize_parameter_distribution(self, circuit, title: str, save: bool = True):
        """FIXED: Visualize parameter distributions with proper error handling."""
        if not hasattr(circuit, 'trainable_variables') or len(circuit.trainable_variables) == 0:
            print("⚠️  No trainable variables found for parameter distribution analysis")
            return
        
        try:
            # Extract parameter values safely
            param_values = []
            param_names = []
            
            for var in circuit.trainable_variables:
                try:
                    param_values.append(float(var.numpy()))
                    param_names.append(var.name.split(':')[0])
                except:
                    logger.warning(f"Could not extract value from parameter {var.name}")
            
            if not param_values:
                print("⚠️  No parameter values could be extracted")
                return
            
            # Create parameter analysis plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Parameter value histogram
            axes[0, 0].hist(param_values, bins=min(20, len(param_values)), 
                           alpha=0.7, color='skyblue', edgecolor='black')
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
            axes[1, 0].bar(range(len(param_magnitudes)), param_magnitudes, 
                          alpha=0.7, color='coral')
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
            
        except Exception as e:
            logger.error(f"Failed to visualize parameter distribution: {e}")
    
    # =====================================================================
    # ENHANCED QUANTUM STATE VISUALIZATION  
    # =====================================================================
    
    def visualize_quantum_state(self, state, title: str = "Quantum State",
                               modes: Optional[List[int]] = None,
                               save: bool = True):
        """
        ENHANCED: Quantum state visualization with 3D Wigner functions.
        """
        if modes is None:
            modes = list(range(min(state.num_modes, 2)))  # Limit for visualization
        
        print(f"🎭 {title.upper()} - QUANTUM STATE ANALYSIS")
        print("=" * 60)
        print(f"State Configuration:")
        print(f"  Modes: {state.num_modes}")
        print(f"  Visualizing modes: {modes}")
        
        try:
            # 1. NEW: 3D Wigner function visualization (SF supports this!)
            self._create_wigner_mountains(state, modes, title, save)
            
            # 2. Fock probability analysis
            self._analyze_fock_probabilities(state, modes, title, save)
            
            # 3. State vector analysis
            self._analyze_state_vector(state, title, save)
            
        except Exception as e:
            logger.error(f"State visualization failed: {e}")
    
    def _create_wigner_mountains(self, state, modes: List[int], title: str, save: bool):
        """NEW: Create 3D Wigner function visualizations (SF actually supports this!)."""
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
                    # Calculate Wigner function - SF DOES support this!
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
    
    def _analyze_fock_probabilities(self, state, modes: List[int], title: str, save: bool):
        """FIXED: Analyze Fock probabilities using actual SF state methods."""
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
    
    def _analyze_state_vector(self, state, title: str, save: bool):
        """FIXED: Analyze state vector using actual SF methods."""
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
    
    # =====================================================================
    # ENHANCED TRAINING VISUALIZATION
    # =====================================================================
    
    def create_training_dashboard(self, training_history: Dict[str, List[float]], 
                                 title: str = "Quantum GAN Training",
                                 save: bool = True):
        """ENHANCED: Training dashboard with parameter evolution."""
        print(f"📊 Creating {title} Dashboard...")
        
        try:
            # Check what metrics are available
            available_metrics = [key for key, values in training_history.items() if values]
            
            if not available_metrics:
                print("⚠️  No training metrics available for dashboard")
                return
            
            # Create enhanced dashboard with parameter evolution
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # Main metrics plots
            metric_plots = [
                ('g_loss', 'Generator Loss', 'blue'),
                ('d_loss', 'Discriminator Loss', 'red'),
                ('w_distance', 'Wasserstein Distance', 'green'),
                ('gradient_penalty', 'Gradient Penalty', 'purple'),
                ('entropy_bonus', 'Entropy Bonus', 'orange'),
                ('physics_penalty', 'Physics Penalty', 'brown')
            ]
            
            plot_idx = 0
            for metric, label, color in metric_plots:
                if metric in training_history and training_history[metric] and plot_idx < 6:
                    row = plot_idx // 3
                    col = plot_idx % 3
                    ax = fig.add_subplot(gs[row, col])
                    
                    values = training_history[metric]
                    epochs = range(len(values))
                    ax.plot(epochs, values, color=color, linewidth=2, alpha=0.8)
                    ax.set_title(label, fontsize=12, fontweight='bold')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Value')
                    ax.grid(True, alpha=0.3)
                    
                    plot_idx += 1
            
            # Training summary
            ax_summary = fig.add_subplot(gs[0, 3])
            if training_history:
                latest_metrics = {}
                for key, values in training_history.items():
                    if values:
                        latest_metrics[key] = values[-1]
                
                summary_text = "Latest Metrics:\n"
                summary_text += "-" * 15 + "\n"
                for key, value in latest_metrics.items():
                    summary_text += f"{key}: {value:.4f}\n"
                
                ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                               fontsize=10, verticalalignment='top', fontfamily='monospace',
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                ax_summary.set_title('Current Status', fontsize=12, fontweight='bold')
                ax_summary.axis('off')
            
            # Parameter evolution heatmap (if available)
            ax_params = fig.add_subplot(gs[2, :])
            if hasattr(self, 'parameter_history') and self.parameter_history:
                param_matrix = np.array(self.parameter_history).T
                im = ax_params.imshow(param_matrix, cmap='RdBu_r', aspect='auto', interpolation='bilinear')
                ax_params.set_title('Parameter Evolution Heatmap', fontsize=14, fontweight='bold')
                ax_params.set_xlabel('Training Step')
                ax_params.set_ylabel('Parameter Index')
                plt.colorbar(im, ax=ax_params, label='Parameter Value')
            else:
                ax_params.text(0.5, 0.5, 'Parameter Evolution\n(Enable parameter tracking)', 
                              ha='center', va='center', transform=ax_params.transAxes, fontsize=14)
                ax_params.set_title('Parameter Evolution', fontsize=14, fontweight='bold')
                ax_params.axis('off')
            
            plt.suptitle(f'{title} Dashboard', fontsize=18, fontweight='bold')
            plt.tight_layout()
            
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.save_dir}/dashboard_{title.lower().replace(' ', '_')}_{timestamp}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"Training dashboard saved to {filename}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Could not create training dashboard: {e}")
    
    # =====================================================================
    # NEW: QGAN COMPARISON FUNCTIONALITY
    # =====================================================================
    
    def create_qgan_comparison_dashboard(self, generator, discriminator, 
                                       real_data: tf.Tensor,
                                       n_samples: int = 100,
                                       title: str = "QGAN Analysis"):
        """NEW: Create comprehensive QGAN comparison dashboard."""
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
    # FIXED INTEGRATION METHODS
    # =====================================================================
    
    def integrate_with_pure_quantum_generator(self, generator, title: str = "Generator"):
        """FIXED: Integration with pure quantum generator."""
        print(f"🔗 Integrating visualization with {title}...")
        
        try:
            # Visualize generator quantum circuit
            if hasattr(generator, 'circuit'):
                self.visualize_pure_sf_circuit(generator.circuit, f"{title} Circuit")
            
            # Test generation
            if hasattr(generator, 'generate') and hasattr(generator, 'latent_dim'):
                z_sample = tf.random.normal([1, generator.latent_dim])
                generated_samples = generator.generate(z_sample)
                
                print(f"✅ Generation test successful:")
                print(f"   Input shape: {z_sample.shape}")
                print(f"   Output shape: {generated_samples.shape}")
                print(f"   Output values: {generated_samples.numpy()}")
            
            # Check transformation matrices (should be static)
            if hasattr(generator, 'T_matrix') and hasattr(generator, 'A_matrix'):
                print(f"📋 Transformation matrices:")
                print(f"   T matrix trainable: {generator.T_matrix.trainable}")
                print(f"   A matrix trainable: {generator.A_matrix.trainable}")
                print(f"   T matrix shape: {generator.T_matrix.shape}")
                print(f"   A matrix shape: {generator.A_matrix.shape}")
                
        except Exception as e:
            logger.error(f"Could not integrate with generator: {e}")
    
    def integrate_with_pure_quantum_discriminator(self, discriminator, title: str = "Discriminator"):
        """FIXED: Integration with pure quantum discriminator."""
        print(f"🔗 Integrating visualization with {title}...")
        
        try:
            # Visualize discriminator quantum circuit
            if hasattr(discriminator, 'circuit'):
                self.visualize_pure_sf_circuit(discriminator.circuit, f"{title} Circuit")
            
            # Test discrimination
            if hasattr(discriminator, 'discriminate') and hasattr(discriminator, 'input_dim'):
                test_samples = tf.random.normal([2, discriminator.input_dim])
                outputs = discriminator.discriminate(test_samples)
                
                print(f"✅ Discrimination test successful:")
                print(f"   Input shape: {test_samples.shape}")
                print(f"   Output shape: {outputs.shape}")
                print(f"   Output values: {outputs.numpy()}")
            
            # Check transformation matrix (should be static)
            if hasattr(discriminator, 'A_matrix'):
                print(f"📋 Transformation matrix:")
                print(f"   A matrix trainable: {discriminator.A_matrix.trainable}")
                print(f"   A matrix shape: {discriminator.A_matrix.shape}")
                
        except Exception as e:
            logger.error(f"Could not integrate with discriminator: {e}")
    
    # =====================================================================
    # UTILITY METHODS
    # =====================================================================
    
    def quick_circuit_viz(self, circuit, title: str = "Circuit"):
        """Quick circuit visualization."""
        self.visualize_pure_sf_circuit(circuit, title, show_parameters=True, show_values=True)
    
    def quick_state_viz(self, state, title: str = "State"):
        """Quick quantum state visualization."""
        self.visualize_quantum_state(state, title, save=False)
    
    def track_parameters(self, generator, discriminator):
        """Track parameter evolution during training."""
        try:
            if hasattr(generator, 'trainable_variables') and hasattr(discriminator, 'trainable_variables'):
                g_params = [float(var.numpy()) for var in generator.trainable_variables]
                d_params = [float(var.numpy()) for var in discriminator.trainable_variables]
                all_params = g_params + d_params
                self.parameter_history.append(all_params)
        except Exception as e:
            logger.warning(f"Could not track parameters: {e}")

    def get_visualization_summary(self) -> Dict[str, Any]:
        """Get summary of visualization capabilities."""
        return {
            'features': [
                'Pure SF circuit structure analysis',
                '3D Wigner function mountains',  
                'Fock probability distributions',
                'State vector analysis',
                'Training progress dashboards',
                'Parameter evolution tracking',
                'QGAN comparison analysis'
            ],
            'save_directory': self.save_dir,
            'tracked_parameters': len(self.parameter_history),
            'tracked_states': len(self.state_history),
            'visualization_count': len(self.visualization_history)
        }


# =====================================================================
# CONVENIENCE FUNCTIONS
# =====================================================================

def create_corrected_visualization_manager(save_dir: str = "visualizations") -> CorrectedQuantumVisualizationManager:
    """Create corrected visualization manager."""
    return CorrectedQuantumVisualizationManager(save_dir)

def quick_circuit_visualization(circuit, title: str = "Circuit"):
    """Quick circuit visualization."""
    viz = CorrectedQuantumVisualizationManager()
    viz.quick_circuit_viz(circuit, title)

def quick_state_visualization(state, title: str = "State"):
    """Quick state visualization."""
    viz = CorrectedQuantumVisualizationManager()
    viz.quick_state_viz(state, title)


# =====================================================================
# DEMO FUNCTION
# =====================================================================

def demo_corrected_visualization():
    """Demonstrate the corrected visualization manager."""
    print("🚀 ENHANCED CORRECTED QUANTUM VISUALIZATION DEMO")
    print("="*50)
    
    try:
        # Create visualization manager
        viz = CorrectedQuantumVisualizationManager("demo_visualizations")
        
        # Create synthetic training history for demo
        training_history = {
            'g_loss': [2.5, 2.1, 1.8, 1.5, 1.2, 1.0],
            'd_loss': [1.8, 1.5, 1.2, 1.0, 0.8, 0.7],
            'w_distance': [1.0, 0.8, 0.6, 0.4, 0.3, 0.2],
            'gradient_penalty': [0.1, 0.08, 0.06, 0.05, 0.04, 0.03]
        }
        
        # Test training dashboard
        viz.create_training_dashboard(training_history, "Demo Training")
        
        print("✅ Enhanced corrected visualization demo completed!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    demo_corrected_visualization()
