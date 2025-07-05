"""
Quantum Mode Diagnostics

Deep analysis tool for understanding quantum circuit behavior in cluster generation.
Analyzes mode outputs, parameter evolution, and input-to-output mapping to identify
why the generator creates linear interpolation instead of discrete clusters.

Focus Areas:
1. Mode-by-mode output analysis 
2. Parameter evolution tracking
3. Input latent to cluster assignment mapping
4. Mode specialization assessment
5. Quantum circuit input integration validation
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import time
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.data_generators import BimodalDataGenerator


def create_well_conditioned_matrices(seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Create well-conditioned matrices from breakthrough analysis."""
    np.random.seed(seed)
    
    raw_encoder = np.random.randn(2, 2)
    raw_decoder = np.random.randn(2, 2)
    
    # Condition encoder
    U_enc, s_enc, Vt_enc = np.linalg.svd(raw_encoder)
    s_enc_conditioned = np.maximum(s_enc, 0.1 * np.max(s_enc))
    encoder_conditioned = U_enc @ np.diag(s_enc_conditioned) @ Vt_enc
    
    # Condition decoder
    U_dec, s_dec, Vt_dec = np.linalg.svd(raw_decoder)
    s_dec_conditioned = np.maximum(s_dec, 0.1 * np.max(s_dec))
    decoder_conditioned = U_dec @ np.diag(s_dec_conditioned) @ Vt_dec
    
    return encoder_conditioned.astype(np.float32), decoder_conditioned.astype(np.float32)


class QuantumModeAnalyzer:
    """
    Deep analysis of quantum mode behavior for cluster generation debugging.
    """
    
    def __init__(self, n_modes: int = 4, layers: int = 2):
        """Initialize quantum mode analyzer."""
        self.n_modes = n_modes
        self.layers = layers
        
        # Create well-conditioned matrices
        encoder_matrix, decoder_matrix = create_well_conditioned_matrices(42)
        self.static_encoder = tf.constant(encoder_matrix, dtype=tf.float32)
        self.static_decoder = tf.constant(decoder_matrix, dtype=tf.float32)
        
        # Quantum parameters (same as SF GAN v0.1)
        self.quantum_params = tf.Variable(
            tf.random.normal([n_modes * layers], stddev=0.1, seed=42),
            name="quantum_parameters"
        )
        
        self.mode_mixing = tf.Variable(
            tf.random.normal([n_modes, 2], stddev=0.05, seed=43),
            name="mode_mixing"
        )
        
        # Diagnostic storage
        self.mode_outputs_history = []
        self.parameter_history = []
        self.input_mapping_history = []
        
        print(f"Quantum Mode Analyzer initialized:")
        print(f"  Modes: {n_modes}, Layers: {layers}")
        print(f"  Focus: Understanding linear interpolation vs discrete clusters")
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Return trainable parameters."""
        return [self.quantum_params, self.mode_mixing]
    
    def analyze_single_sample(self, z_sample: tf.Tensor, detailed: bool = True) -> Dict:
        """
        Detailed analysis of how a single latent sample is processed.
        
        Args:
            z_sample: Single latent vector [1, latent_dim]
            detailed: Whether to return detailed intermediate results
            
        Returns:
            Dictionary with complete processing breakdown
        """
        # Step 1: Encoding analysis
        encoded = tf.matmul(z_sample, self.static_encoder)  # [1, 2]
        
        # Step 2: Mode-by-mode quantum processing analysis
        mode_analysis = {}
        mode_activations = []
        
        for mode in range(self.n_modes):
            # Get mode-specific parameters
            mode_param = self.quantum_params[mode * self.layers:(mode + 1) * self.layers]
            
            # CRITICAL: Analyze this step - source of linear interpolation?
            sample_influence = tf.reduce_sum(encoded) + tf.reduce_sum(mode_param)
            
            # Mode mixing analysis
            mode_mix = self.mode_mixing[mode]
            mode_activation = tf.sin(sample_influence + mode_mix[0]) * tf.cos(mode_mix[1])
            mode_activations.append(mode_activation)
            
            # Store detailed mode analysis
            mode_analysis[f'mode_{mode}'] = {
                'mode_param': mode_param.numpy(),
                'sample_influence': float(sample_influence),
                'mode_mix': mode_mix.numpy(),
                'activation': float(mode_activation),
                'sin_component': float(tf.sin(sample_influence + mode_mix[0])),
                'cos_component': float(tf.cos(mode_mix[1]))
            }
        
        # Step 3: Mode combination analysis
        quantum_output = tf.stack(mode_activations, axis=0)  # [n_modes]
        
        # CRITICAL: This averaging might be causing the interpolation!
        quantum_2d = tf.reduce_mean(tf.reshape(quantum_output, [-1, 2]), axis=0, keepdims=True)  # [1, 2]
        
        # Step 4: Noise and decoding
        noise = tf.random.normal(tf.shape(quantum_2d), stddev=0.1, seed=42)  # Fixed seed for analysis
        quantum_diverse = quantum_2d + noise
        
        # Step 5: Final output
        output = tf.matmul(quantum_diverse, self.static_decoder)  # [1, 2]
        
        analysis = {
            'input_latent': z_sample.numpy(),
            'encoded': encoded.numpy(),
            'mode_analysis': mode_analysis,
            'mode_activations': [float(a) for a in mode_activations],
            'quantum_output_raw': quantum_output.numpy(),
            'quantum_2d': quantum_2d.numpy(),
            'quantum_diverse': quantum_diverse.numpy(),
            'final_output': output.numpy(),
            'processing_summary': {
                'encoding_norm': float(tf.norm(encoded)),
                'quantum_output_std': float(tf.math.reduce_std(quantum_output)),
                'mode_activation_range': [float(tf.reduce_min(quantum_output)), float(tf.reduce_max(quantum_output))],
                'final_output_norm': float(tf.norm(output))
            }
        }
        
        return analysis
    
    def batch_mode_analysis(self, batch_size: int = 32) -> Dict:
        """
        Analyze mode behavior across a batch of diverse inputs.
        
        Args:
            batch_size: Number of samples to analyze
            
        Returns:
            Comprehensive batch analysis
        """
        print(f"Running batch mode analysis on {batch_size} samples...")
        
        # Generate diverse latent inputs
        z_batch = tf.random.normal([batch_size, 2], seed=42)
        
        batch_analysis = {
            'samples': [],
            'mode_specialization': {},
            'input_output_mapping': {},
            'cluster_assignment_analysis': {}
        }
        
        # Process each sample
        all_outputs = []
        all_mode_activations = []
        
        for i in range(batch_size):
            z_sample = z_batch[i:i+1]
            sample_analysis = self.analyze_single_sample(z_sample, detailed=False)
            
            batch_analysis['samples'].append(sample_analysis)
            all_outputs.append(sample_analysis['final_output'][0])
            all_mode_activations.append(sample_analysis['mode_activations'])
        
        all_outputs = np.array(all_outputs)  # [batch_size, 2]
        all_mode_activations = np.array(all_mode_activations)  # [batch_size, n_modes]
        
        # Mode specialization analysis
        for mode in range(self.n_modes):
            mode_values = all_mode_activations[:, mode]
            batch_analysis['mode_specialization'][f'mode_{mode}'] = {
                'mean': float(np.mean(mode_values)),
                'std': float(np.std(mode_values)),
                'range': [float(np.min(mode_values)), float(np.max(mode_values))],
                'unique_values': len(np.unique(np.round(mode_values, 4)))
            }
        
        # Input-output mapping analysis
        batch_analysis['input_output_mapping'] = {
            'input_range': [list(np.min(z_batch.numpy(), axis=0)), list(np.max(z_batch.numpy(), axis=0))],
            'output_range': [list(np.min(all_outputs, axis=0)), list(np.max(all_outputs, axis=0))],
            'output_std': list(np.std(all_outputs, axis=0)),
            'linear_correlation': float(np.corrcoef(z_batch.numpy()[:, 0], all_outputs[:, 0])[0, 1])
        }
        
        # Cluster assignment analysis
        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(all_outputs)
            
            batch_analysis['cluster_assignment_analysis'] = {
                'detected_clusters': len(np.unique(cluster_labels)),
                'cluster_balance': [int(np.sum(cluster_labels == 0)), int(np.sum(cluster_labels == 1))],
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'inertia': float(kmeans.inertia_),
                'is_linear_pattern': self._detect_linear_pattern(all_outputs)
            }
        except Exception as e:
            batch_analysis['cluster_assignment_analysis'] = {'error': str(e)}
        
        return batch_analysis
    
    def _detect_linear_pattern(self, outputs: np.ndarray) -> Dict:
        """Detect if outputs form a linear pattern instead of clusters."""
        # Fit a line to the outputs
        X = outputs[:, 0].reshape(-1, 1)
        y = outputs[:, 1]
        
        # Simple linear regression
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        
        numerator = np.sum((X.flatten() - X_mean) * (y - y_mean))
        denominator = np.sum((X.flatten() - X_mean) ** 2)
        
        if denominator > 1e-10:
            slope = numerator / denominator
            intercept = y_mean - slope * X_mean
            
            # Calculate R-squared
            y_pred = slope * X.flatten() + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-10))
            
            return {
                'is_linear': r_squared > 0.8,
                'r_squared': float(r_squared),
                'slope': float(slope),
                'intercept': float(intercept),
                'line_equation': f'y = {slope:.3f}x + {intercept:.3f}'
            }
        else:
            return {'is_linear': False, 'r_squared': 0.0}
    
    def identify_interpolation_source(self) -> Dict:
        """
        Identify the exact source of linear interpolation in the quantum circuit.
        """
        print("Identifying source of linear interpolation...")
        
        # Test with extreme latent inputs
        test_inputs = [
            tf.constant([[-2.0, -2.0]], dtype=tf.float32),  # Extreme negative
            tf.constant([[0.0, 0.0]], dtype=tf.float32),    # Center
            tf.constant([[2.0, 2.0]], dtype=tf.float32)     # Extreme positive
        ]
        
        interpolation_analysis = {
            'extreme_inputs_test': [],
            'parameter_sensitivity': {},
            'averaging_effect': {}
        }
        
        for i, z_input in enumerate(test_inputs):
            analysis = self.analyze_single_sample(z_input)
            interpolation_analysis['extreme_inputs_test'].append({
                'input': analysis['input_latent'].tolist(),
                'output': analysis['final_output'].tolist(),
                'mode_activations': analysis['mode_activations']
            })
        
        # Test parameter sensitivity
        original_params = self.quantum_params.numpy().copy()
        
        # Perturb each parameter and see output change
        for param_idx in range(len(original_params)):
            # Small perturbation
            perturbed_params = original_params.copy()
            perturbed_params[param_idx] += 0.1
            self.quantum_params.assign(perturbed_params)
            
            # Test with center input
            center_analysis = self.analyze_single_sample(tf.constant([[0.0, 0.0]], dtype=tf.float32))
            
            interpolation_analysis['parameter_sensitivity'][f'param_{param_idx}'] = {
                'output_change': center_analysis['final_output'].tolist(),
                'mode_affected': param_idx // self.layers
            }
        
        # Restore original parameters
        self.quantum_params.assign(original_params)
        
        # Test averaging effect
        # What if we don't average the modes?
        z_test = tf.constant([[1.0, -1.0]], dtype=tf.float32)
        encoded = tf.matmul(z_test, self.static_encoder)
        
        mode_activations = []
        for mode in range(self.n_modes):
            mode_param = self.quantum_params[mode * self.layers:(mode + 1) * self.layers]
            sample_influence = tf.reduce_sum(encoded) + tf.reduce_sum(mode_param)
            mode_mix = self.mode_mixing[mode]
            mode_activation = tf.sin(sample_influence + mode_mix[0]) * tf.cos(mode_mix[1])
            mode_activations.append(float(mode_activation))
        
        quantum_output = tf.stack([tf.constant(a) for a in mode_activations], axis=0)
        
        # Compare averaging vs max vs other combinations
        averaged = tf.reduce_mean(tf.reshape(quantum_output, [-1, 2]), axis=0)
        max_selected = tf.reduce_max(tf.reshape(quantum_output, [-1, 2]), axis=0)
        
        interpolation_analysis['averaging_effect'] = {
            'raw_mode_activations': mode_activations,
            'averaged_result': averaged.numpy().tolist(),
            'max_selected_result': max_selected.numpy().tolist(),
            'averaging_vs_max_diff': float(tf.norm(averaged - max_selected))
        }
        
        return interpolation_analysis
    
    def visualize_mode_analysis(self, analysis: Dict, save_dir: str = "results/quantum_mode_diagnostics"):
        """Create comprehensive visualizations of mode analysis."""
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Quantum Mode Diagnostics - Source of Linear Interpolation', fontsize=16, fontweight='bold')
        
        # Plot 1: Mode specialization
        mode_stds = [analysis['mode_specialization'][f'mode_{i}']['std'] for i in range(self.n_modes)]
        mode_ranges = [analysis['mode_specialization'][f'mode_{i}']['range'][1] - 
                      analysis['mode_specialization'][f'mode_{i}']['range'][0] for i in range(self.n_modes)]
        
        axes[0, 0].bar(range(self.n_modes), mode_stds, alpha=0.7, label='Std Dev')
        axes[0, 0].set_title('Mode Specialization (Std Dev)')
        axes[0, 0].set_xlabel('Mode Index')
        axes[0, 0].set_ylabel('Standard Deviation')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Input-output correlation
        inputs = np.array([s['input_latent'][0] for s in analysis['samples']])
        outputs = np.array([s['final_output'][0] for s in analysis['samples']])
        
        axes[0, 1].scatter(inputs[:, 0], outputs[:, 0], alpha=0.6, s=20, label='X dimension')
        axes[0, 1].scatter(inputs[:, 1], outputs[:, 1], alpha=0.6, s=20, label='Y dimension')
        axes[0, 1].set_title(f'Input-Output Correlation\nR² = {analysis["input_output_mapping"]["linear_correlation"]:.3f}')
        axes[0, 1].set_xlabel('Input Value')
        axes[0, 1].set_ylabel('Output Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Output distribution
        if 'cluster_assignment_analysis' in analysis and 'is_linear_pattern' in analysis['cluster_assignment_analysis']:
            linear_info = analysis['cluster_assignment_analysis']['is_linear_pattern']
            axes[0, 2].scatter(outputs[:, 0], outputs[:, 1], alpha=0.6, s=20)
            axes[0, 2].set_title(f'Output Distribution\nLinear: {linear_info.get("is_linear", False)}\nR² = {linear_info.get("r_squared", 0):.3f}')
            axes[0, 2].set_xlabel('X Output')
            axes[0, 2].set_ylabel('Y Output')
            axes[0, 2].grid(True, alpha=0.3)
            
            # Add fitted line if linear
            if linear_info.get('is_linear', False):
                x_line = np.linspace(outputs[:, 0].min(), outputs[:, 0].max(), 100)
                y_line = linear_info['slope'] * x_line + linear_info['intercept']
                axes[0, 2].plot(x_line, y_line, 'r--', alpha=0.8, label='Fitted Line')
                axes[0, 2].legend()
        
        # Plot 4: Mode activation patterns
        mode_matrix = np.array([s['mode_activations'] for s in analysis['samples']])
        im = axes[1, 0].imshow(mode_matrix.T, aspect='auto', cmap='viridis')
        axes[1, 0].set_title('Mode Activation Patterns')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Mode Index')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Plot 5: Parameter influence analysis
        param_evolution = []
        for sample in analysis['samples'][:10]:  # First 10 samples
            param_influences = []
            for mode in range(self.n_modes):
                param_influences.append(sample['mode_analysis'][f'mode_{mode}']['sample_influence'])
            param_evolution.append(param_influences)
        
        param_matrix = np.array(param_evolution)
        for mode in range(self.n_modes):
            axes[1, 1].plot(param_matrix[:, mode], 'o-', label=f'Mode {mode}')
        axes[1, 1].set_title('Parameter Influence Evolution')
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Sample Influence')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Summary statistics
        stats_text = f"""Mode Diagnostics Summary:

Linear Pattern: {analysis['cluster_assignment_analysis'].get('is_linear_pattern', {}).get('is_linear', 'Unknown')}
R²: {analysis['cluster_assignment_analysis'].get('is_linear_pattern', {}).get('r_squared', 0):.3f}

Detected Clusters: {analysis['cluster_assignment_analysis'].get('detected_clusters', 'Unknown')}
Target Clusters: 2

Output Range:
X: [{analysis['input_output_mapping']['output_range'][0][0]:.3f}, {analysis['input_output_mapping']['output_range'][1][0]:.3f}]
Y: [{analysis['input_output_mapping']['output_range'][0][1]:.3f}, {analysis['input_output_mapping']['output_range'][1][1]:.3f}]

Mode Specialization:
Avg Std: {np.mean(mode_stds):.4f}
Max Range: {np.max(mode_ranges):.4f}

PROBLEM SOURCE:
{"Linear interpolation detected!" if analysis['cluster_assignment_analysis'].get('is_linear_pattern', {}).get('is_linear', False) else "Cluster formation detected"}
"""
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, "quantum_mode_diagnostics.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Quantum mode diagnostics saved: {save_path}")
        return save_path


def main():
    """Main quantum mode diagnostics execution."""
    print("🔬 QUANTUM MODE DIAGNOSTICS")
    print("=" * 60)
    print("Deep analysis of quantum circuit behavior in cluster generation")
    print("Focus: Understanding why linear interpolation occurs instead of discrete clusters")
    print("=" * 60)
    
    # Suppress TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    try:
        # Create analyzer
        analyzer = QuantumModeAnalyzer(n_modes=4, layers=2)
        
        # Run comprehensive analysis
        print("\n1. Running batch mode analysis...")
        batch_analysis = analyzer.batch_mode_analysis(batch_size=50)
        
        print("\n2. Identifying interpolation source...")
        interpolation_analysis = analyzer.identify_interpolation_source()
        
        print("\n3. Creating visualizations...")
        viz_path = analyzer.visualize_mode_analysis(batch_analysis)
        
        # Analysis summary
        print("\n" + "=" * 60)
        print("🎯 QUANTUM MODE DIAGNOSTIC RESULTS")
        print("=" * 60)
        
        linear_pattern = batch_analysis['cluster_assignment_analysis'].get('is_linear_pattern', {})
        is_linear = linear_pattern.get('is_linear', False)
        r_squared = linear_pattern.get('r_squared', 0)
        
        print(f"Linear Pattern Analysis:")
        print(f"  Is Linear: {is_linear}")
        print(f"  R-squared: {r_squared:.4f}")
        print(f"  Line equation: {linear_pattern.get('line_equation', 'N/A')}")
        
        print(f"\nMode Specialization:")
        for mode in range(4):
            mode_data = batch_analysis['mode_specialization'][f'mode_{mode}']
            print(f"  Mode {mode}: std={mode_data['std']:.4f}, range={mode_data['range']}")
        
        print(f"\nCluster Detection:")
        cluster_data = batch_analysis['cluster_assignment_analysis']
        print(f"  Detected clusters: {cluster_data.get('detected_clusters', 'N/A')}")
        print(f"  Target clusters: 2")
        print(f"  Cluster balance: {cluster_data.get('cluster_balance', 'N/A')}")
        
        # Critical findings
        print(f"\n🚨 CRITICAL FINDINGS:")
        if is_linear and r_squared > 0.8:
            print(f"  ❌ CONFIRMED: Generator creates linear interpolation (R²={r_squared:.3f})")
            print(f"  ❌ Problem: Continuous manifold instead of discrete clusters")
            print(f"  📍 Root cause: Quantum circuit averaging creates smooth transitions")
        else:
            print(f"  ✅ Generator shows some cluster formation")
        
        print(f"\n🔧 RECOMMENDED FIXES:")
        print(f"  1. Replace mode averaging with discrete mode selection")
        print(f"  2. Add cluster assignment loss to encourage discrete outputs")
        print(f"  3. Implement mode specialization (each mode learns one cluster)")
        print(f"  4. Add input-dependent quantum parameter modulation")
        
        print(f"\n📊 Visualizations saved: {viz_path}")
        
        return {
            'batch_analysis': batch_analysis,
            'interpolation_analysis': interpolation_analysis,
            'is_linear': is_linear,
            'r_squared': r_squared,
            'visualization_path': viz_path
        }
        
    except Exception as e:
        print(f"❌ Quantum mode diagnostics failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
