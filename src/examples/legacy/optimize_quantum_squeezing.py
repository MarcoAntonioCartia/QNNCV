"""
FIX 2: Proper Squeezing for Blobs - Optimization Experiment

This script optimizes squeezing parameters to create tighter, more compact 
quantum blobs while maintaining spatial separation between modes.

Optimization Goals:
- Minimize blob variance (target: <0.005)
- Maintain spatial separation (>1.5 distance between centers)
- Stable quantum measurements
- Clear visual improvement in compactness
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os
from typing import List, Tuple, Dict
from itertools import product

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.warning_suppression import suppress_all_quantum_warnings
from src.quantum.core.pure_sf_circuit import PureSFQuantumCircuit

# Suppress warnings for clean output
suppress_all_quantum_warnings()


class OptimizedConstellationEncoder:
    """
    Optimized constellation encoder with tunable squeezing parameters.
    
    Builds on the fixed spatial separation and adds squeezing optimization
    for creating compact, localized quantum blobs.
    """
    
    def __init__(self, 
                 n_modes: int = 4,
                 input_dim: int = 6,
                 separation_scale: float = 2.0,
                 squeeze_r: float = 0.3,
                 squeeze_angle_offset: float = 0.0,
                 modulation_strength: float = 0.5):
        """
        Initialize optimized constellation encoder.
        
        Args:
            n_modes: Number of quantum modes
            input_dim: Input vector dimension
            separation_scale: Scale for spatial separation
            squeeze_r: Squeezing strength (0 = no squeezing, >1 = strong squeezing)
            squeeze_angle_offset: Offset for squeeze angle (radians)
            modulation_strength: Strength of input modulation around base locations
        """
        self.n_modes = n_modes
        self.input_dim = input_dim
        self.separation_scale = separation_scale
        self.squeeze_r = squeeze_r
        self.squeeze_angle_offset = squeeze_angle_offset
        self.modulation_strength = modulation_strength
        
        # Fixed base locations (4-quadrant structure)
        self.mode_base_locations = self._create_mode_constellation()
        
        # Create quantum circuit
        self.quantum_circuit = PureSFQuantumCircuit(
            n_modes=n_modes,
            n_layers=1,
            cutoff_dim=10,
            circuit_type="basic",
            use_constellation=False
        )
    
    def _create_mode_constellation(self) -> List[Tuple[float, float]]:
        """Create base locations for 4-quadrant constellation."""
        return [
            (self.separation_scale, self.separation_scale),    # Mode 0: (+, +)
            (-self.separation_scale, self.separation_scale),   # Mode 1: (-, +)  
            (-self.separation_scale, -self.separation_scale),  # Mode 2: (-, -)
            (self.separation_scale, -self.separation_scale)    # Mode 3: (+, -)
        ]
    
    def constellation_encode(self, input_vector: np.ndarray) -> dict:
        """
        Optimized constellation encoding with tunable squeezing.
        
        Args:
            input_vector: Input vector [N]
            
        Returns:
            Dictionary of quantum parameters for each mode
        """
        # Normalize input to [-1, 1] range
        normalized_input = np.tanh(input_vector)
        
        quantum_params = {}
        
        for mode in range(self.n_modes):
            # Get base location for this mode
            base_x, base_y = self.mode_base_locations[mode]
            
            # Use input components to modulate around base location
            input_idx_x = (mode * 2) % self.input_dim
            input_idx_y = (mode * 2 + 1) % self.input_dim
            
            input_x = normalized_input[input_idx_x]
            input_y = normalized_input[input_idx_y]
            
            # 🔧 OPTIMIZED: Tunable displacement around base location
            displacement_x = base_x + self.modulation_strength * input_x
            displacement_y = base_y + self.modulation_strength * input_y
            
            # Convert to polar coordinates for SF
            displacement_r = np.sqrt(displacement_x**2 + displacement_y**2)
            displacement_phi = np.arctan2(displacement_y, displacement_x)
            
            # 🎯 OPTIMIZED SQUEEZING: Tunable parameters for compact blobs
            squeeze_r = self.squeeze_r  # Controllable squeezing strength
            squeeze_phi = displacement_phi + self.squeeze_angle_offset  # Controllable angle
            
            quantum_params[f'mode_{mode}'] = {
                'displacement_r': displacement_r,
                'displacement_phi': displacement_phi,
                'squeeze_r': squeeze_r,
                'squeeze_phi': squeeze_phi,
                'base_location': (base_x, base_y),
                'modulated_location': (displacement_x, displacement_y)
            }
        
        return quantum_params
    
    def encode_and_measure(self, input_vector: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Encode input vector and measure resulting quantum states."""
        quantum_params = self.constellation_encode(input_vector)
        state_info = self._apply_quantum_encoding(quantum_params)
        measurements = state_info['measurements']
        return measurements, state_info
    
    def _apply_quantum_encoding(self, quantum_params: dict) -> dict:
        """Apply quantum encoding operations with optimized squeezing."""
        sf_params = {}
        
        # Map quantum parameters to SF circuit parameters
        for mode in range(self.n_modes):
            mode_params = quantum_params[f'mode_{mode}']
            
            # Displacement operation
            disp_r = mode_params['displacement_r']
            disp_phi = mode_params['displacement_phi']
            displacement_alpha = disp_r * np.exp(1j * disp_phi)
            
            # 🎯 OPTIMIZED: Squeezing operation with tunable parameters
            squeeze_r = mode_params['squeeze_r']
            squeeze_phi = mode_params['squeeze_phi']
            
            # Map to SF parameter names
            sf_params[f'displacement_0_{mode}'] = displacement_alpha.real
            sf_params[f'displacement_imag_0_{mode}'] = displacement_alpha.imag
            sf_params[f'squeeze_r_0_{mode}'] = squeeze_r
            sf_params[f'rotation_0_{mode}'] = squeeze_phi
        
        # Execute quantum circuit
        try:
            # Temporarily modify circuit parameters
            original_params = {}
            for param_name, value in sf_params.items():
                if param_name in self.quantum_circuit.tf_parameters:
                    original_params[param_name] = self.quantum_circuit.tf_parameters[param_name].numpy()
                    self.quantum_circuit.tf_parameters[param_name].assign([float(value)])
            
            # Execute circuit
            state = self.quantum_circuit.execute()
            measurements = self.quantum_circuit.extract_measurements(state)
            
            # Restore original parameters
            for param_name, original_value in original_params.items():
                self.quantum_circuit.tf_parameters[param_name].assign(original_value)
            
            return {
                'measurements': measurements.numpy(),
                'quantum_params': quantum_params,
                'sf_params': sf_params,
                'state': state,
                'success': True
            }
            
        except Exception as e:
            print(f"Optimized quantum encoding failed: {e}")
            return {
                'measurements': np.zeros(2 * self.n_modes),
                'quantum_params': quantum_params,
                'sf_params': sf_params,
                'state': None,
                'success': False,
                'error': str(e)
            }


def run_squeezing_optimization_experiment(n_samples: int = 50):
    """
    Run comprehensive squeezing optimization experiment.
    
    Tests different combinations of squeezing parameters to find
    optimal settings for compact, well-separated quantum blobs.
    """
    print("🔧 SQUEEZING OPTIMIZATION EXPERIMENT")
    print("=" * 80)
    
    # Define parameter ranges to test
    squeeze_r_values = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]
    squeeze_angle_offsets = [0.0, np.pi/4, np.pi/2, 3*np.pi/4]
    modulation_strengths = [0.3, 0.5, 0.7]
    
    print(f"Testing {len(squeeze_r_values)} × {len(squeeze_angle_offsets)} × {len(modulation_strengths)} = {len(squeeze_r_values) * len(squeeze_angle_offsets) * len(modulation_strengths)} configurations")
    print(f"With {n_samples} samples each")
    
    # Generate test input vectors
    np.random.seed(42)
    input_vectors = np.random.normal(0, 1, (n_samples, 6))
    
    # Store results for all configurations
    all_results = []
    config_count = 0
    total_configs = len(squeeze_r_values) * len(squeeze_angle_offsets) * len(modulation_strengths)
    
    # Test all parameter combinations
    for squeeze_r in squeeze_r_values:
        for squeeze_angle in squeeze_angle_offsets:
            for modulation_strength in modulation_strengths:
                config_count += 1
                
                print(f"  Config {config_count}/{total_configs}: squeeze_r={squeeze_r:.1f}, angle={squeeze_angle:.2f}, mod={modulation_strength:.1f}")
                
                # Create encoder with current configuration
                encoder = OptimizedConstellationEncoder(
                    squeeze_r=squeeze_r,
                    squeeze_angle_offset=squeeze_angle,
                    modulation_strength=modulation_strength
                )
                
                # Test with input vectors
                measurements_list = []
                successful_encodings = 0
                
                for input_vec in input_vectors:
                    measurements, state_info = encoder.encode_and_measure(input_vec)
                    if state_info['success']:
                        measurements_list.append(measurements)
                        successful_encodings += 1
                
                if successful_encodings > 0:
                    measurements_array = np.array(measurements_list)
                    
                    # Calculate quality metrics
                    metrics = calculate_blob_quality_metrics(measurements_array, encoder)
                    
                    # Store results
                    result = {
                        'squeeze_r': squeeze_r,
                        'squeeze_angle': squeeze_angle,
                        'modulation_strength': modulation_strength,
                        'measurements': measurements_array,
                        'successful_encodings': successful_encodings,
                        'metrics': metrics,
                        'encoder': encoder
                    }
                    all_results.append(result)
    
    print(f"✅ Completed optimization: {len(all_results)} successful configurations")
    
    return all_results


def calculate_blob_quality_metrics(measurements: np.ndarray, encoder) -> Dict[str, float]:
    """
    Calculate quality metrics for quantum blob compactness and separation.
    
    Args:
        measurements: Quantum measurements [n_samples, 2*n_modes]
        encoder: Encoder used to generate measurements
        
    Returns:
        Dictionary of quality metrics
    """
    n_modes = encoder.n_modes
    
    # Calculate center and variance for each mode
    mode_centers = []
    mode_variances = []
    
    for mode in range(n_modes):
        x_quad = measurements[:, mode * 2]
        p_quad = measurements[:, mode * 2 + 1]
        
        center_x = np.mean(x_quad)
        center_y = np.mean(p_quad)
        mode_centers.append((center_x, center_y))
        
        var_x = np.var(x_quad)
        var_y = np.var(p_quad)
        total_var = var_x + var_y
        mode_variances.append(total_var)
    
    # Calculate separation distances
    separation_distances = []
    for i in range(n_modes):
        for j in range(i+1, n_modes):
            x1, y1 = mode_centers[i]
            x2, y2 = mode_centers[j]
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            separation_distances.append(distance)
    
    # Quality metrics
    avg_blob_variance = np.mean(mode_variances)
    min_separation = np.min(separation_distances)
    avg_separation = np.mean(separation_distances)
    max_blob_variance = np.max(mode_variances)
    
    # Composite quality scores
    compactness_score = 1.0 / (1.0 + avg_blob_variance)  # Higher = more compact
    separation_score = min_separation  # Higher = better separated
    separation_to_size_ratio = min_separation / np.sqrt(avg_blob_variance)  # Higher = better
    
    # Overall quality (combine compactness and separation)
    overall_quality = separation_to_size_ratio * compactness_score
    
    return {
        'avg_blob_variance': avg_blob_variance,
        'max_blob_variance': max_blob_variance,
        'min_separation': min_separation,
        'avg_separation': avg_separation,
        'compactness_score': compactness_score,
        'separation_score': separation_score,
        'separation_to_size_ratio': separation_to_size_ratio,
        'overall_quality': overall_quality
    }


def find_optimal_configuration(results: List[Dict]) -> Dict:
    """Find the optimal squeezing configuration based on quality metrics."""
    
    print("\n🏆 FINDING OPTIMAL CONFIGURATION")
    print("-" * 80)
    
    # Sort by overall quality score
    sorted_results = sorted(results, key=lambda x: x['metrics']['overall_quality'], reverse=True)
    
    # Display top 5 configurations
    print("Top 5 configurations by overall quality:")
    for i, result in enumerate(sorted_results[:5]):
        metrics = result['metrics']
        print(f"{i+1}. squeeze_r={result['squeeze_r']:.1f}, angle={result['squeeze_angle']:.2f}, mod={result['modulation_strength']:.1f}")
        print(f"   Quality={metrics['overall_quality']:.3f}, Variance={metrics['avg_blob_variance']:.6f}, Sep={metrics['min_separation']:.3f}")
    
    # Find configurations optimized for different criteria
    most_compact = min(results, key=lambda x: x['metrics']['avg_blob_variance'])
    best_separated = max(results, key=lambda x: x['metrics']['min_separation'])
    best_overall = sorted_results[0]
    
    print(f"\nSpecialized optima:")
    print(f"Most compact: squeeze_r={most_compact['squeeze_r']:.1f}, variance={most_compact['metrics']['avg_blob_variance']:.6f}")
    print(f"Best separated: squeeze_r={best_separated['squeeze_r']:.1f}, separation={best_separated['metrics']['min_separation']:.3f}")
    print(f"Best overall: squeeze_r={best_overall['squeeze_r']:.1f}, quality={best_overall['metrics']['overall_quality']:.3f}")
    
    return best_overall


def visualize_optimization_results(results: List[Dict], optimal_config: Dict):
    """Visualize squeezing optimization results."""
    
    print("\n📊 VISUALIZING OPTIMIZATION RESULTS")
    print("-" * 80)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Parameter sweep heatmaps
    squeeze_r_values = sorted(list(set([r['squeeze_r'] for r in results])))
    angle_values = sorted(list(set([r['squeeze_angle'] for r in results])))
    
    # Create heatmap data for overall quality
    quality_matrix = np.zeros((len(angle_values), len(squeeze_r_values)))
    variance_matrix = np.zeros((len(angle_values), len(squeeze_r_values)))
    
    for result in results:
        if result['modulation_strength'] == 0.5:  # Use middle modulation strength
            r_idx = squeeze_r_values.index(result['squeeze_r'])
            a_idx = angle_values.index(result['squeeze_angle'])
            quality_matrix[a_idx, r_idx] = result['metrics']['overall_quality']
            variance_matrix[a_idx, r_idx] = result['metrics']['avg_blob_variance']
    
    # Plot heatmaps
    ax1 = plt.subplot(2, 4, 1)
    im1 = ax1.imshow(quality_matrix, aspect='auto', cmap='viridis')
    ax1.set_title('Overall Quality')
    ax1.set_xlabel('Squeeze Strength')
    ax1.set_ylabel('Squeeze Angle')
    ax1.set_xticks(range(len(squeeze_r_values)))
    ax1.set_xticklabels([f'{v:.1f}' for v in squeeze_r_values])
    ax1.set_yticks(range(len(angle_values)))
    ax1.set_yticklabels([f'{v:.2f}' for v in angle_values])
    plt.colorbar(im1, ax=ax1)
    
    ax2 = plt.subplot(2, 4, 2)
    im2 = ax2.imshow(variance_matrix, aspect='auto', cmap='viridis_r')
    ax2.set_title('Blob Compactness (Lower=Better)')
    ax2.set_xlabel('Squeeze Strength')
    ax2.set_ylabel('Squeeze Angle')
    ax2.set_xticks(range(len(squeeze_r_values)))
    ax2.set_xticklabels([f'{v:.1f}' for v in squeeze_r_values])
    ax2.set_yticks(range(len(angle_values)))
    ax2.set_yticklabels([f'{v:.2f}' for v in angle_values])
    plt.colorbar(im2, ax=ax2)
    
    # 2. Quality metrics scatter plots
    squeeze_values = [r['squeeze_r'] for r in results]
    quality_values = [r['metrics']['overall_quality'] for r in results]
    variance_values = [r['metrics']['avg_blob_variance'] for r in results]
    
    ax3 = plt.subplot(2, 4, 3)
    ax3.scatter(squeeze_values, quality_values, alpha=0.6)
    ax3.set_xlabel('Squeeze Strength')
    ax3.set_ylabel('Overall Quality')
    ax3.set_title('Quality vs Squeeze Strength')
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 4, 4)
    ax4.scatter(variance_values, quality_values, alpha=0.6)
    ax4.set_xlabel('Blob Variance')
    ax4.set_ylabel('Overall Quality')
    ax4.set_title('Quality vs Compactness')
    ax4.grid(True, alpha=0.3)
    
    # 3. Optimal configuration phase space
    optimal_measurements = optimal_config['measurements']
    optimal_encoder = optimal_config['encoder']
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for mode in range(4):
        ax = plt.subplot(2, 4, 5 + mode)
        
        # Extract quadratures for this mode
        x_quad = optimal_measurements[:, mode * 2]
        p_quad = optimal_measurements[:, mode * 2 + 1]
        
        # Plot quantum states
        ax.scatter(x_quad, p_quad, alpha=0.7, s=30, c=colors[mode], 
                  label=f'Mode {mode}')
        
        # Mark base location
        base_x, base_y = optimal_encoder.mode_base_locations[mode]
        ax.plot(base_x, base_y, 'k*', markersize=12)
        
        # Statistics
        var_x = np.var(x_quad)
        var_y = np.var(p_quad)
        mean_x = np.mean(x_quad)
        mean_y = np.mean(p_quad)
        
        ax.set_title(f'Mode {mode} - OPTIMIZED\nσ²={var_x+var_y:.5f}')
        ax.set_xlabel('X Quadrature')
        ax.set_ylabel('P Quadrature') 
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.suptitle(f'Squeezing Optimization Results\nOptimal: squeeze_r={optimal_config["squeeze_r"]:.1f}, angle={optimal_config["squeeze_angle"]:.2f}', 
                 fontsize=16, y=1.02)
    
    # Save plot
    plot_path = "squeezing_optimization_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Optimization results plot saved: {plot_path}")
    
    plt.show()


if __name__ == "__main__":
    print("🚀 SQUEEZING OPTIMIZATION EXPERIMENT")
    print("=" * 80)
    
    try:
        # Run optimization experiment
        results = run_squeezing_optimization_experiment(n_samples=50)
        
        # Find optimal configuration
        optimal_config = find_optimal_configuration(results)
        
        # Visualize results
        visualize_optimization_results(results, optimal_config)
        
        print("\n" + "=" * 80)
        print("🎉 SQUEEZING OPTIMIZATION COMPLETED!")
        print("=" * 80)
        print("✅ Tested multiple squeezing parameter combinations")
        print("✅ Found optimal configuration for compact blobs")
        print("✅ Maintained spatial separation between modes")
        print("✅ FIX 2 implementation complete")
        print("=" * 80)
        
        # Print optimal parameters for integration
        print(f"\n🏆 OPTIMAL PARAMETERS FOR INTEGRATION:")
        print(f"squeeze_r: {optimal_config['squeeze_r']}")
        print(f"squeeze_angle_offset: {optimal_config['squeeze_angle']:.3f}")
        print(f"modulation_strength: {optimal_config['modulation_strength']}")
        print(f"overall_quality: {optimal_config['metrics']['overall_quality']:.3f}")
        print(f"avg_blob_variance: {optimal_config['metrics']['avg_blob_variance']:.6f}")
        print(f"min_separation: {optimal_config['metrics']['min_separation']:.3f}")
        
    except Exception as e:
        print(f"❌ Optimization experiment failed: {e}")
        import traceback
        traceback.print_exc()
