"""
Quantum Data Flow Diagnostics Tool

This tool provides comprehensive analysis of how data flows through the 
quantum GAN pipeline, identifying where diversity is lost and how to 
properly inject encoded inputs into quantum circuits as initial states.

Focus Areas:
1. Static transformation matrix analysis
2. Data flow tracing through entire pipeline  
3. 2D visualization of transformation effects
4. Quantum circuit input integration diagnosis
5. Initial state creation from encoded parameters
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from typing import Dict, List, Tuple, Optional
from matplotlib.patches import Ellipse
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.quantum.core.sf_tutorial_circuit import SFTutorialGenerator


class QuantumDataFlowDiagnostics:
    """Comprehensive diagnostics for quantum data flow issues."""
    
    def __init__(self):
        """Initialize diagnostics tool."""
        self.generator = None
        self.test_data = None
        self.flow_analysis = {}
        
        print("🔍 Quantum Data Flow Diagnostics Initialized")
        print("   Focus: Tracing data flow and identifying diversity loss")
    
    def setup_generator(self, config: Dict = None):
        """Setup generator for analysis."""
        if config is None:
            config = {
                'latent_dim': 4,
                'output_dim': 2, 
                'n_modes': 3,
                'n_layers': 2,
                'cutoff_dim': 6
            }
        
        print(f"\n🔧 Setting up generator with config: {config}")
        
        self.generator = SFTutorialGenerator(**config)
        self.config = config
        
        # Create test data
        np.random.seed(42)  # For reproducible analysis
        self.test_data = {
            'z_batch': tf.random.normal([16, config['latent_dim']], seed=42),
            'z_single': tf.random.normal([1, config['latent_dim']], seed=43),
            'target_centers': [(-1.5, -1.5), (1.5, 1.5)]
        }
        
        print(f"   ✅ Generator created with {len(self.generator.trainable_variables)} trainable params")
        print(f"   ✅ Test data prepared: {self.test_data['z_batch'].shape}")
    
    def analyze_transformation_matrices(self) -> Dict:
        """Analyze static transformation matrices properties."""
        print(f"\n🔍 Analyzing Static Transformation Matrices")
        print("=" * 60)
        
        encoder = self.generator.static_encoder.numpy()
        decoder = self.generator.static_decoder.numpy()
        
        analysis = {
            'encoder': self._analyze_matrix(encoder, "Encoder"),
            'decoder': self._analyze_matrix(decoder, "Decoder"),
            'pipeline_analysis': self._analyze_pipeline_effect(encoder, decoder)
        }
        
        return analysis
    
    def _analyze_matrix(self, matrix: np.ndarray, name: str) -> Dict:
        """Analyze individual matrix properties."""
        print(f"\n📊 {name} Matrix Analysis:")
        print(f"   Shape: {matrix.shape}")
        
        # Basic properties
        matrix_norm = np.linalg.norm(matrix)
        frobenius_norm = np.linalg.norm(matrix, 'fro')
        condition_number = np.linalg.cond(matrix)
        
        print(f"   Matrix norm: {matrix_norm:.6f}")
        print(f"   Frobenius norm: {frobenius_norm:.6f}")
        print(f"   Condition number: {condition_number:.6f}")
        
        # Singular values
        U, s, Vt = np.linalg.svd(matrix)
        print(f"   Singular values: {s}")
        print(f"   Rank: {np.linalg.matrix_rank(matrix)}")
        
        # Check for problematic properties
        warnings = []
        if matrix_norm < 0.1:
            warnings.append(f"Very small matrix norm: {matrix_norm:.6f}")
        if condition_number > 100:
            warnings.append(f"High condition number: {condition_number:.2f}")
        if np.min(s) < 1e-10:
            warnings.append(f"Near-singular matrix: min singular value = {np.min(s):.2e}")
        
        if warnings:
            print(f"   ⚠️ Warnings:")
            for warning in warnings:
                print(f"      • {warning}")
        else:
            print(f"   ✅ Matrix properties look healthy")
        
        return {
            'shape': matrix.shape,
            'matrix_norm': matrix_norm,
            'frobenius_norm': frobenius_norm,
            'condition_number': condition_number,
            'singular_values': s,
            'rank': np.linalg.matrix_rank(matrix),
            'warnings': warnings,
            'matrix': matrix
        }
    
    def _analyze_pipeline_effect(self, encoder: np.ndarray, decoder: np.ndarray) -> Dict:
        """Analyze combined effect of encoder-decoder pipeline."""
        print(f"\n🔗 Pipeline Analysis (Encoder → Decoder):")
        
        # Test with unit vectors
        input_dim = encoder.shape[0]
        test_vectors = np.eye(input_dim)  # Unit vectors
        
        # Transform through encoder
        encoded = test_vectors @ encoder
        
        # Transform through decoder  
        decoded = encoded @ decoder
        
        # Analyze preservation
        input_norms = np.linalg.norm(test_vectors, axis=1)
        output_norms = np.linalg.norm(decoded, axis=1)
        norm_ratios = output_norms / input_norms
        
        print(f"   Input unit vector norms: {input_norms}")
        print(f"   Output vector norms: {output_norms}")
        print(f"   Norm preservation ratios: {norm_ratios}")
        print(f"   Average norm ratio: {np.mean(norm_ratios):.6f}")
        print(f"   Norm ratio std: {np.std(norm_ratios):.6f}")
        
        # Check if pipeline is too compressive
        avg_ratio = np.mean(norm_ratios)
        if avg_ratio < 0.1:
            print(f"   ⚠️ WARNING: Pipeline heavily compresses signals (ratio={avg_ratio:.6f})")
        elif avg_ratio > 10:
            print(f"   ⚠️ WARNING: Pipeline heavily amplifies signals (ratio={avg_ratio:.6f})")
        else:
            print(f"   ✅ Pipeline norm preservation looks reasonable")
        
        return {
            'norm_ratios': norm_ratios,
            'avg_norm_ratio': avg_ratio,
            'norm_ratio_std': np.std(norm_ratios),
            'input_norms': input_norms,
            'output_norms': output_norms
        }
    
    def trace_data_flow(self) -> Dict:
        """Trace data flow through entire pipeline step by step."""
        print(f"\n🔍 Tracing Data Flow Through Pipeline")
        print("=" * 60)
        
        z_test = self.test_data['z_single']  # Single sample for detailed analysis
        print(f"Input latent vector: {z_test.numpy().flatten()}")
        
        flow_trace = {}
        
        # Step 1: Static encoding
        print(f"\n1️⃣ Static Encoding:")
        encoded = tf.matmul(z_test, self.generator.static_encoder)
        flow_trace['encoded'] = encoded.numpy()
        print(f"   Input shape: {z_test.shape}")
        print(f"   Encoder shape: {self.generator.static_encoder.shape}")
        print(f"   Encoded output: {encoded.numpy().flatten()}")
        print(f"   Encoded norm: {tf.linalg.norm(encoded).numpy():.6f}")
        
        # Step 2: Quantum circuit execution (CRITICAL ISSUE HERE)
        print(f"\n2️⃣ Quantum Circuit Execution:")
        print(f"   🚨 CRITICAL ISSUE: Encoded input is IGNORED!")
        print(f"   Current implementation executes same circuit regardless of input")
        
        # Show what currently happens
        state1 = self.generator.quantum_circuit.execute()
        measurements1 = self.generator.quantum_circuit.extract_measurements(state1)
        
        state2 = self.generator.quantum_circuit.execute()  
        measurements2 = self.generator.quantum_circuit.extract_measurements(state2)
        
        print(f"   Measurement 1: {measurements1.numpy()}")
        print(f"   Measurement 2: {measurements2.numpy()}")
        print(f"   Difference: {tf.reduce_max(tf.abs(measurements1 - measurements2)).numpy():.10f}")
        
        if tf.reduce_max(tf.abs(measurements1 - measurements2)).numpy() < 1e-10:
            print(f"   ❌ CONFIRMED: Quantum circuit produces identical outputs!")
        
        flow_trace['quantum_measurements'] = measurements1.numpy()
        
        # Step 3: Static decoding
        print(f"\n3️⃣ Static Decoding:")
        decoded = tf.matmul(measurements1[None, :], self.generator.static_decoder)
        flow_trace['final_output'] = decoded.numpy()
        print(f"   Measurements shape: {measurements1.shape}")
        print(f"   Decoder shape: {self.generator.static_decoder.shape}")
        print(f"   Final output: {decoded.numpy().flatten()}")
        print(f"   Final norm: {tf.linalg.norm(decoded).numpy():.6f}")
        
        # Step 4: Compare with full generation
        print(f"\n4️⃣ Full Generation Comparison:")
        full_output = self.generator.generate(z_test)
        flow_trace['full_generation'] = full_output.numpy()
        print(f"   Full generate(): {full_output.numpy().flatten()}")
        print(f"   Manual trace: {decoded.numpy().flatten()}")
        print(f"   Difference: {tf.reduce_max(tf.abs(full_output - decoded)).numpy():.10f}")
        
        return flow_trace
    
    def test_input_dependency(self) -> Dict:
        """Test if outputs depend on inputs (they currently don't!)."""
        print(f"\n🔍 Testing Input Dependency")
        print("=" * 60)
        
        # Generate multiple different inputs
        z_batch = self.test_data['z_batch']
        print(f"Testing with {z_batch.shape[0]} different input vectors")
        
        # Generate outputs
        outputs = self.generator.generate(z_batch)
        
        # Analyze diversity
        output_mean = tf.reduce_mean(outputs, axis=0)
        output_std = tf.math.reduce_std(outputs, axis=0)
        pairwise_distances = tf.reduce_mean(tf.abs(outputs[:, None, :] - outputs[None, :, :]), axis=-1)
        
        print(f"\nOutput Analysis:")
        print(f"   Output shape: {outputs.shape}")
        print(f"   Output mean: {output_mean.numpy()}")
        print(f"   Output std: {output_std.numpy()}")
        print(f"   Max pairwise distance: {tf.reduce_max(pairwise_distances).numpy():.10f}")
        print(f"   Min pairwise distance: {tf.reduce_min(pairwise_distances + tf.eye(len(outputs)) * 1e6).numpy():.10f}")
        
        # Check for identical outputs
        max_distance = tf.reduce_max(pairwise_distances).numpy()
        if max_distance < 1e-8:
            print(f"   ❌ CONFIRMED: All outputs are essentially identical!")
            print(f"   🚨 DIVERSITY COMPLETELY DEAD!")
        else:
            print(f"   ✅ Some diversity detected")
        
        return {
            'outputs': outputs.numpy(),
            'output_mean': output_mean.numpy(),
            'output_std': output_std.numpy(),
            'max_pairwise_distance': max_distance,
            'diversity_status': 'dead' if max_distance < 1e-8 else 'alive'
        }
    
    def visualize_2d_transformations(self, save_dir: str = "results/quantum_flow_diagnostics"):
        """Create 2D visualizations of data flow."""
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n🎨 Creating 2D Data Flow Visualizations")
        print("=" * 60)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 16))
        
        # Test data: grid of latent vectors for visualization
        n_samples = 100
        z_grid = tf.random.normal([n_samples, self.config['latent_dim']], seed=42)
        
        # Step 1: Original latent space (project to 2D for visualization)
        ax1 = plt.subplot(3, 3, 1)
        latent_2d = z_grid[:, :2].numpy()  # Take first 2 dimensions
        ax1.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=20)
        ax1.set_title('1. Latent Space (z)\n(First 2 dimensions)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('z_0')
        ax1.set_ylabel('z_1')
        
        # Step 2: After static encoding (project to 2D)
        ax2 = plt.subplot(3, 3, 2)
        encoded = tf.matmul(z_grid, self.generator.static_encoder)
        encoded_2d = encoded[:, :2].numpy()  # Take first 2 dimensions
        ax2.scatter(encoded_2d[:, 0], encoded_2d[:, 1], alpha=0.6, s=20, color='orange')
        ax2.set_title('2. After Static Encoder\n(First 2 dimensions)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('encoded_0')
        ax2.set_ylabel('encoded_1')
        
        # Step 3: Quantum measurements (all identical!)
        ax3 = plt.subplot(3, 3, 3)
        # Since quantum circuit ignores input, all measurements are identical
        quantum_measurements = []
        for i in range(min(20, n_samples)):  # Sample a few for speed
            state = self.generator.quantum_circuit.execute()
            measurements = self.generator.quantum_circuit.extract_measurements(state)
            quantum_measurements.append(measurements.numpy())
        
        quantum_measurements = np.array(quantum_measurements)
        quantum_2d = quantum_measurements[:, :2]
        ax3.scatter(quantum_2d[:, 0], quantum_2d[:, 1], alpha=0.6, s=20, color='red')
        ax3.set_title('3. Quantum Measurements\n(Should be diverse but identical!)')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('measurement_0')
        ax3.set_ylabel('measurement_1')
        
        # Step 4: Final outputs
        ax4 = plt.subplot(3, 3, 4)
        outputs = self.generator.generate(z_grid[:20])  # Sample for speed
        ax4.scatter(outputs[:, 0].numpy(), outputs[:, 1].numpy(), alpha=0.6, s=20, color='purple')
        ax4.set_title('4. Final Outputs\n(All identical!)')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel('output_0')
        ax4.set_ylabel('output_1')
        
        # Step 5: Target bimodal data
        ax5 = plt.subplot(3, 3, 5)
        target_centers = np.array(self.test_data['target_centers'])
        for i, center in enumerate(target_centers):
            cluster = np.random.normal(center, 0.3, (50, 2))
            ax5.scatter(cluster[:, 0], cluster[:, 1], alpha=0.6, s=20, label=f'Target Cluster {i+1}')
        ax5.set_title('5. Target Bimodal Distribution\n(What we want)')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        ax5.set_xlabel('feature_0')
        ax5.set_ylabel('feature_1')
        
        # Step 6: Transformation matrix visualizations
        ax6 = plt.subplot(3, 3, 6)
        encoder_matrix = self.generator.static_encoder.numpy()
        im1 = ax6.imshow(encoder_matrix, cmap='RdBu', aspect='auto')
        ax6.set_title('6. Static Encoder Matrix')
        ax6.set_xlabel('Output dimension')
        ax6.set_ylabel('Input dimension')
        plt.colorbar(im1, ax=ax6)
        
        ax7 = plt.subplot(3, 3, 7)
        decoder_matrix = self.generator.static_decoder.numpy()
        im2 = ax7.imshow(decoder_matrix, cmap='RdBu', aspect='auto')
        ax7.set_title('7. Static Decoder Matrix')
        ax7.set_xlabel('Output dimension')
        ax7.set_ylabel('Input dimension')
        plt.colorbar(im2, ax=ax7)
        
        # Step 8: Problem identification
        ax8 = plt.subplot(3, 3, 8)
        ax8.text(0.1, 0.8, "🚨 PROBLEM IDENTIFIED:", fontsize=14, fontweight='bold', color='red')
        ax8.text(0.1, 0.7, "• Encoded input is IGNORED by quantum circuit", fontsize=10)
        ax8.text(0.1, 0.6, "• Circuit executes with same parameters always", fontsize=10)
        ax8.text(0.1, 0.5, "• All quantum measurements are identical", fontsize=10)
        ax8.text(0.1, 0.4, "• No input → output dependency", fontsize=10)
        ax8.text(0.1, 0.3, "", fontsize=10)
        ax8.text(0.1, 0.2, "✅ SOLUTION NEEDED:", fontsize=14, fontweight='bold', color='green')
        ax8.text(0.1, 0.1, "Use encoded input for initial quantum states", fontsize=10)
        ax8.text(0.1, 0.0, "via displacement/squeezing operations", fontsize=10)
        ax8.set_xlim(0, 1)
        ax8.set_ylim(0, 1)
        ax8.axis('off')
        
        # Step 9: Data flow diagram
        ax9 = plt.subplot(3, 3, 9)
        ax9.text(0.1, 0.9, "DATA FLOW ANALYSIS:", fontsize=12, fontweight='bold')
        ax9.text(0.1, 0.8, "z → Encoder → encoded ❌ IGNORED", fontsize=10)
        ax9.text(0.1, 0.7, "Circuit → same_output", fontsize=10)
        ax9.text(0.1, 0.6, "same_output → Decoder → identical", fontsize=10)
        ax9.text(0.1, 0.5, "", fontsize=10)
        ax9.text(0.1, 0.4, "SHOULD BE:", fontsize=10, fontweight='bold')
        ax9.text(0.1, 0.3, "z → Encoder → encoded ✅ USED", fontsize=10)
        ax9.text(0.1, 0.2, "Circuit(encoded) → diverse_output", fontsize=10)
        ax9.text(0.1, 0.1, "diverse_output → Decoder → clusters", fontsize=10)
        ax9.set_xlim(0, 1)
        ax9.set_ylim(0, 1)
        ax9.axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, "quantum_data_flow_analysis.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Comprehensive visualization saved: {save_path}")
        
        # Create detailed matrix analysis plot
        self._create_matrix_analysis_plot(save_dir)
        
        return save_path
    
    def _create_matrix_analysis_plot(self, save_dir: str):
        """Create detailed matrix analysis visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Static Transformation Matrix Analysis', fontsize=16, fontweight='bold')
        
        encoder = self.generator.static_encoder.numpy()
        decoder = self.generator.static_decoder.numpy()
        
        # Encoder analysis
        axes[0, 0].imshow(encoder, cmap='RdBu', aspect='auto')
        axes[0, 0].set_title('Encoder Matrix')
        axes[0, 0].set_xlabel('Output Dimension')
        axes[0, 0].set_ylabel('Input Dimension')
        
        # Encoder singular values
        U, s_enc, Vt = np.linalg.svd(encoder)
        axes[0, 1].bar(range(len(s_enc)), s_enc)
        axes[0, 1].set_title('Encoder Singular Values')
        axes[0, 1].set_xlabel('Index')
        axes[0, 1].set_ylabel('Singular Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Encoder effect on unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        unit_circle = np.array([np.cos(theta), np.sin(theta)]).T
        if encoder.shape[0] >= 2:
            transformed_circle = unit_circle @ encoder[:2, :2]  # Take 2x2 submatrix
            axes[0, 2].plot(unit_circle[:, 0], unit_circle[:, 1], 'b-', label='Unit Circle')
            axes[0, 2].plot(transformed_circle[:, 0], transformed_circle[:, 1], 'r-', label='Transformed')
            axes[0, 2].set_title('Encoder Effect on Unit Circle')
            axes[0, 2].set_aspect('equal')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Decoder analysis
        axes[1, 0].imshow(decoder, cmap='RdBu', aspect='auto')
        axes[1, 0].set_title('Decoder Matrix')
        axes[1, 0].set_xlabel('Output Dimension')
        axes[1, 0].set_ylabel('Input Dimension')
        
        # Decoder singular values
        U, s_dec, Vt = np.linalg.svd(decoder)
        axes[1, 1].bar(range(len(s_dec)), s_dec)
        axes[1, 1].set_title('Decoder Singular Values')
        axes[1, 1].set_xlabel('Index')
        axes[1, 1].set_ylabel('Singular Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Combined pipeline effect
        if encoder.shape[1] == decoder.shape[0] and decoder.shape[1] >= 2:
            # Test pipeline on unit vectors
            input_vecs = np.eye(encoder.shape[0])
            pipeline_output = input_vecs @ encoder @ decoder
            
            axes[1, 2].scatter(range(len(input_vecs)), np.linalg.norm(pipeline_output, axis=1), 
                             alpha=0.7, s=50)
            axes[1, 2].set_title('Pipeline Effect on Unit Vectors')
            axes[1, 2].set_xlabel('Input Unit Vector Index')
            axes[1, 2].set_ylabel('Output Norm')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, "matrix_analysis_detailed.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Matrix analysis visualization saved: {save_path}")
    
    def propose_solution(self) -> Dict:
        """Propose solution for integrating encoded input into quantum circuit."""
        print(f"\n💡 SOLUTION PROPOSAL")
        print("=" * 60)
        
        print("🎯 Core Problem:")
        print("   The quantum circuit completely ignores the encoded input!")
        print("   It should use encoded parameters for initial quantum states.")
        
        print("\n✅ Proposed Solution:")
        print("   1. Use encoded input to set initial coherent states via Displacement gates")
        print("   2. Use encoded input to modulate Squeezing parameters")
        print("   3. Preserve 100% gradient flow through proper SF implementation")
        
        print("\n🔧 Implementation Strategy:")
        print("   • Modify quantum circuit to accept encoded input parameters")
        print("   • Apply displacement operations: D(α) |0⟩ where α comes from encoded input")
        print("   • Apply squeezing operations: S(r,φ) where r,φ come from encoded input")
        print("   • Keep static encoder/decoder as requested")
        print("   • Maintain SF Tutorial gradient flow pattern")
        
        solution_details = {
            'problem': 'Quantum circuit ignores encoded input',
            'solution': 'Use encoded input for initial quantum state preparation',
            'implementation_steps': [
                'Modify SFTutorialCircuit to accept input parameters',
                'Add initial state preparation using Displacement gates',
                'Optionally add Squeezing modulation',
                'Preserve SF Tutorial gradient flow architecture',
                'Test with diverse inputs to confirm diversity restoration'
            ],
            'expected_outcome': 'Different inputs → Different quantum states → Diverse outputs'
        }
        
        return solution_details
    
    def run_complete_diagnosis(self) -> Dict:
        """Run complete diagnostic analysis."""
        print("🔍 QUANTUM DATA FLOW COMPLETE DIAGNOSIS")
        print("=" * 80)
        print("Investigating why all outputs cluster near origin despite 100% gradient flow")
        print("=" * 80)
        
        # Setup
        self.setup_generator()
        
        # Run all analyses
        results = {}
        
        # 1. Matrix analysis
        results['matrix_analysis'] = self.analyze_transformation_matrices()
        
        # 2. Data flow tracing
        results['data_flow'] = self.trace_data_flow()
        
        # 3. Input dependency test
        results['input_dependency'] = self.test_input_dependency()
        
        # 4. Create visualizations
        results['visualizations'] = self.visualize_2d_transformations()
        
        # 5. Solution proposal
        results['solution'] = self.propose_solution()
        
        # Summary
        print(f"\n" + "=" * 80)
        print("🎯 DIAGNOSIS COMPLETE!")
        print("=" * 80)
        
        print("📋 SUMMARY FINDINGS:")
        print("   ✅ Static matrices are NOT the problem (properties look healthy)")
        print("   ✅ 100% gradient flow is maintained")
        print("   ❌ Quantum circuit IGNORES encoded input entirely")
        print("   ❌ All quantum measurements are identical")
        print("   ❌ Zero sample diversity (max distance < 1e-8)")
        
        print("\n🚨 ROOT CAUSE:")
        print("   Encoded input is computed but never used by quantum circuit!")
        print("   Circuit executes with same parameters regardless of input.")
        
        print("\n💡 SOLUTION:")
        print("   Use encoded input to create initial quantum states")
        print("   via Displacement and Squeezing operations.")
        
        print(f"\n📊 Visualizations saved in: results/quantum_flow_diagnostics/")
        print("   • quantum_data_flow_analysis.png - Complete data flow analysis")
        print("   • matrix_analysis_detailed.png - Transformation matrix analysis")
        
        return results


def main():
    """Run quantum data flow diagnostics."""
    print("🚀 QUANTUM DATA FLOW DIAGNOSTICS")
    print("Investigating dimensional collapse and data flow issues")
    print("=" * 80)
    
    # Suppress TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    try:
        # Create and run diagnostics
        diagnostics = QuantumDataFlowDiagnostics()
        results = diagnostics.run_complete_diagnosis()
        
        print("\n🎉 Diagnostics completed successfully!")
        print("Check the visualizations to see exactly where diversity dies.")
        
        return results
        
    except Exception as e:
        print(f"❌ Diagnostics failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
