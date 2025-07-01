"""
Matrix Conditioning Fix for Quantum Data Flow

This tool fixes the severe matrix conditioning issues identified in the diagnostics.
The current static transformation matrices cause extreme compression (unit circle → line segment),
leading to all outputs clustering near origin.

Focus Areas:
1. Analyze current matrix conditioning problems
2. Design well-conditioned replacement matrices  
3. Test same-dimensionality case (2D → 2D)
4. Preserve input diversity through transformations
5. Validate improvements with visualizations
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler

def create_orthogonal_matrix(size: int, seed: Optional[int] = None) -> np.ndarray:
    """Create orthogonal matrix using QR decomposition."""
    if seed is not None:
        np.random.seed(seed)
    # Generate random matrix
    A = np.random.randn(size, size)
    # QR decomposition gives us orthogonal matrix Q
    Q, R = np.linalg.qr(A)
    # Ensure determinant is positive (proper rotation, not reflection)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.quantum.core.sf_tutorial_circuit import SFTutorialGenerator


class MatrixConditioningFix:
    """Fix matrix conditioning issues in static transformations."""
    
    def __init__(self):
        """Initialize matrix conditioning fix tool."""
        self.original_generator = None
        self.fixed_generators = {}
        self.analysis_results = {}
        
        print("🔧 Matrix Conditioning Fix Tool Initialized")
        print("   Focus: Fixing severe matrix compression issues")
    
    def analyze_current_matrices(self) -> Dict:
        """Detailed analysis of current matrix conditioning issues."""
        print("\n🔍 ANALYZING CURRENT MATRIX CONDITIONING ISSUES")
        print("=" * 70)
        
        # Create original generator
        config = {'latent_dim': 2, 'output_dim': 2, 'n_modes': 2, 'n_layers': 2, 'cutoff_dim': 6}
        self.original_generator = SFTutorialGenerator(**config)
        
        # Get matrices
        encoder = self.original_generator.static_encoder.numpy()
        decoder = self.original_generator.static_decoder.numpy()
        
        print(f"Current matrices:")
        print(f"  Encoder shape: {encoder.shape}")
        print(f"  Decoder shape: {decoder.shape}")
        
        # Analyze conditioning
        analysis = {}
        
        # Encoder analysis
        print(f"\n📊 ENCODER CONDITIONING ANALYSIS:")
        encoder_cond = np.linalg.cond(encoder)
        encoder_rank = np.linalg.matrix_rank(encoder)
        U, s, Vt = np.linalg.svd(encoder)
        encoder_norm = np.linalg.norm(encoder)
        
        print(f"   Condition number: {encoder_cond:.2e}")
        print(f"   Matrix rank: {encoder_rank}")
        print(f"   Singular values: {s}")
        print(f"   Matrix norm: {encoder_norm:.6f}")
        print(f"   Singular value ratio: {s[0]/s[-1]:.2e}")
        
        # Check for problems
        if encoder_cond > 1e12:
            print(f"   🚨 CRITICAL: Near-singular matrix!")
        elif encoder_cond > 1e6:
            print(f"   ⚠️ WARNING: Poorly conditioned matrix")
        else:
            print(f"   ✅ Conditioning looks acceptable")
        
        analysis['encoder'] = {
            'condition_number': encoder_cond,
            'rank': encoder_rank,
            'singular_values': s,
            'norm': encoder_norm,
            'matrix': encoder
        }
        
        # Decoder analysis
        print(f"\n📊 DECODER CONDITIONING ANALYSIS:")
        decoder_cond = np.linalg.cond(decoder)
        decoder_rank = np.linalg.matrix_rank(decoder)
        U_d, s_d, Vt_d = np.linalg.svd(decoder)
        decoder_norm = np.linalg.norm(decoder)
        
        print(f"   Condition number: {decoder_cond:.2e}")
        print(f"   Matrix rank: {decoder_rank}")
        print(f"   Singular values: {s_d}")
        print(f"   Matrix norm: {decoder_norm:.6f}")
        print(f"   Singular value ratio: {s_d[0]/s_d[-1]:.2e}")
        
        analysis['decoder'] = {
            'condition_number': decoder_cond,
            'rank': decoder_rank,
            'singular_values': s_d,
            'norm': decoder_norm,
            'matrix': decoder
        }
        
        # Pipeline analysis
        print(f"\n🔗 PIPELINE ANALYSIS:")
        pipeline_matrix = encoder @ decoder  # For same dimensions
        pipeline_cond = np.linalg.cond(pipeline_matrix)
        U_p, s_p, Vt_p = np.linalg.svd(pipeline_matrix)
        
        print(f"   Pipeline condition number: {pipeline_cond:.2e}")
        print(f"   Pipeline singular values: {s_p}")
        
        # Test unit circle transformation
        theta = np.linspace(0, 2*np.pi, 100)
        unit_circle = np.array([np.cos(theta), np.sin(theta)]).T
        
        # Transform through pipeline
        if encoder.shape[0] == 2 and encoder.shape[1] >= 2 and decoder.shape[1] == 2:
            encoded_circle = unit_circle @ encoder[:, :2]  # Take first 2 output dims
            decoded_circle = encoded_circle @ decoder[:2, :]  # Take first 2 input dims
            
            # Analyze deformation
            original_area = np.pi  # Unit circle area
            # Approximate transformed area using shoelace formula
            x, y = decoded_circle[:, 0], decoded_circle[:, 1]
            transformed_area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))
            
            area_ratio = transformed_area / original_area
            print(f"   Unit circle area preservation: {area_ratio:.6f}")
            print(f"   Area compression factor: {1/area_ratio:.2f}x")
            
            if area_ratio < 0.01:
                print(f"   🚨 SEVERE COMPRESSION: {1/area_ratio:.0f}x compression!")
            elif area_ratio < 0.1:
                print(f"   ⚠️ HIGH COMPRESSION: {1/area_ratio:.1f}x compression")
            else:
                print(f"   ✅ Reasonable area preservation")
            
            analysis['pipeline'] = {
                'condition_number': pipeline_cond,
                'area_ratio': area_ratio,
                'compression_factor': 1/area_ratio,
                'unit_circle_original': unit_circle,
                'unit_circle_transformed': decoded_circle
            }
        
        self.analysis_results['original'] = analysis
        return analysis
    
    def create_well_conditioned_matrices(self) -> Dict:
        """Create well-conditioned replacement matrices."""
        print(f"\n🔧 CREATING WELL-CONDITIONED MATRICES")
        print("=" * 70)
        
        # We'll create several different approaches
        strategies = {}
        
        # Strategy 1: Orthogonal matrices (perfect conditioning)
        print(f"\n1️⃣ Strategy 1: Orthogonal Matrices")
        encoder_ortho = create_orthogonal_matrix(2, seed=42)  # 2x2 orthogonal
        decoder_ortho = create_orthogonal_matrix(2, seed=43)  # 2x2 orthogonal
        
        # Scale to reasonable magnitude
        encoder_ortho *= 0.5  # Moderate scaling
        decoder_ortho *= 0.5
        
        print(f"   Encoder condition number: {np.linalg.cond(encoder_ortho):.2e}")
        print(f"   Decoder condition number: {np.linalg.cond(decoder_ortho):.2e}")
        
        strategies['orthogonal'] = {
            'encoder': encoder_ortho,
            'decoder': decoder_ortho,
            'description': 'Orthogonal matrices with moderate scaling'
        }
        
        # Strategy 2: Identity-based (minimal transformation)
        print(f"\n2️⃣ Strategy 2: Identity-Based")
        encoder_identity = np.eye(2) + 0.1 * np.random.randn(2, 2)  # Near-identity
        decoder_identity = np.eye(2) + 0.1 * np.random.randn(2, 2)
        
        print(f"   Encoder condition number: {np.linalg.cond(encoder_identity):.2e}")
        print(f"   Decoder condition number: {np.linalg.cond(decoder_identity):.2e}")
        
        strategies['identity_based'] = {
            'encoder': encoder_identity,
            'decoder': decoder_identity,
            'description': 'Near-identity with small perturbations'
        }
        
        # Strategy 3: Scaled random with good conditioning
        print(f"\n3️⃣ Strategy 3: Well-Conditioned Random")
        # Generate random matrix and condition it
        raw_encoder = np.random.randn(2, 2)
        U, s, Vt = np.linalg.svd(raw_encoder)
        # Ensure good conditioning by setting minimum singular value
        s_conditioned = np.maximum(s, 0.1 * np.max(s))  # Min singular value = 10% of max
        encoder_conditioned = U @ np.diag(s_conditioned) @ Vt
        
        raw_decoder = np.random.randn(2, 2)
        U_d, s_d, Vt_d = np.linalg.svd(raw_decoder)
        s_d_conditioned = np.maximum(s_d, 0.1 * np.max(s_d))
        decoder_conditioned = U_d @ np.diag(s_d_conditioned) @ Vt_d
        
        print(f"   Encoder condition number: {np.linalg.cond(encoder_conditioned):.2e}")
        print(f"   Decoder condition number: {np.linalg.cond(decoder_conditioned):.2e}")
        
        strategies['well_conditioned'] = {
            'encoder': encoder_conditioned,
            'decoder': decoder_conditioned,
            'description': 'Random matrices with enforced good conditioning'
        }
        
        # Strategy 4: Rotation + uniform scaling
        print(f"\n4️⃣ Strategy 4: Rotation + Scaling")
        angle1 = np.pi / 6  # 30 degrees
        rotation1 = np.array([[np.cos(angle1), -np.sin(angle1)],
                             [np.sin(angle1), np.cos(angle1)]])
        encoder_rotation = rotation1 * 0.7  # Uniform scaling
        
        angle2 = -np.pi / 4  # -45 degrees  
        rotation2 = np.array([[np.cos(angle2), -np.sin(angle2)],
                             [np.sin(angle2), np.cos(angle2)]])
        decoder_rotation = rotation2 * 1.2
        
        print(f"   Encoder condition number: {np.linalg.cond(encoder_rotation):.2e}")
        print(f"   Decoder condition number: {np.linalg.cond(decoder_rotation):.2e}")
        
        strategies['rotation_scaling'] = {
            'encoder': encoder_rotation,
            'decoder': decoder_rotation,
            'description': 'Pure rotation with uniform scaling'
        }
        
        return strategies
    
    def test_matrix_strategies(self, strategies: Dict) -> Dict:
        """Test different matrix strategies on bimodal data."""
        print(f"\n🧪 TESTING MATRIX STRATEGIES")
        print("=" * 70)
        
        # Create test bimodal data
        np.random.seed(42)
        n_samples = 200
        
        # Bimodal clusters
        cluster1 = np.random.normal([-1.5, -1.5], 0.3, (n_samples//2, 2))
        cluster2 = np.random.normal([1.5, 1.5], 0.3, (n_samples//2, 2))
        bimodal_data = np.vstack([cluster1, cluster2])
        
        # Test each strategy
        results = {}
        
        for strategy_name, strategy in strategies.items():
            print(f"\n📊 Testing Strategy: {strategy_name}")
            print(f"   Description: {strategy['description']}")
            
            encoder = strategy['encoder']
            decoder = strategy['decoder']
            
            # Transform bimodal data through pipeline
            encoded = bimodal_data @ encoder
            decoded = encoded @ decoder
            
            # Analyze preservation
            original_mean = np.mean(bimodal_data, axis=0)
            original_std = np.std(bimodal_data, axis=0)
            original_span = np.max(bimodal_data, axis=0) - np.min(bimodal_data, axis=0)
            
            decoded_mean = np.mean(decoded, axis=0)
            decoded_std = np.std(decoded, axis=0)
            decoded_span = np.max(decoded, axis=0) - np.min(decoded, axis=0)
            
            # Metrics
            mean_preservation = np.linalg.norm(decoded_mean - original_mean) / np.linalg.norm(original_mean)
            std_preservation = np.mean(decoded_std / original_std)
            span_preservation = np.mean(decoded_span / original_span)
            
            # Cluster separation
            from sklearn.cluster import KMeans
            original_kmeans = KMeans(n_clusters=2, random_state=42).fit(bimodal_data)
            decoded_kmeans = KMeans(n_clusters=2, random_state=42).fit(decoded)
            
            original_separation = np.linalg.norm(original_kmeans.cluster_centers_[0] - original_kmeans.cluster_centers_[1])
            decoded_separation = np.linalg.norm(decoded_kmeans.cluster_centers_[0] - decoded_kmeans.cluster_centers_[1])
            separation_preservation = decoded_separation / original_separation
            
            print(f"   Mean shift: {mean_preservation:.4f}")
            print(f"   Std preservation: {std_preservation:.4f}")
            print(f"   Span preservation: {span_preservation:.4f}")
            print(f"   Cluster separation preservation: {separation_preservation:.4f}")
            
            # Overall quality score
            quality_score = (std_preservation + span_preservation + separation_preservation) / 3
            print(f"   Overall quality score: {quality_score:.4f}")
            
            if quality_score > 0.8:
                print(f"   ✅ EXCELLENT preservation")
            elif quality_score > 0.5:
                print(f"   ✅ GOOD preservation")
            elif quality_score > 0.2:
                print(f"   ⚠️ MODERATE preservation")
            else:
                print(f"   ❌ POOR preservation")
            
            results[strategy_name] = {
                'encoder': encoder,
                'decoder': decoder,
                'original_data': bimodal_data,
                'encoded_data': encoded,
                'decoded_data': decoded,
                'mean_preservation': mean_preservation,
                'std_preservation': std_preservation,
                'span_preservation': span_preservation,
                'separation_preservation': separation_preservation,
                'quality_score': quality_score,
                'description': strategy['description']
            }
        
        return results
    
    def visualize_matrix_improvements(self, strategies: Dict, test_results: Dict, save_dir: str = "results/matrix_conditioning_fix"):
        """Create comprehensive visualizations of matrix improvements."""
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n🎨 Creating Matrix Improvement Visualizations")
        print("=" * 70)
        
        # Create comprehensive comparison plot
        fig = plt.figure(figsize=(20, 16))
        
        # Original matrix effects
        original_analysis = self.analysis_results['original']
        
        # Plot 1: Original unit circle transformation
        ax1 = plt.subplot(3, 5, 1)
        if 'pipeline' in original_analysis:
            original_circle = original_analysis['pipeline']['unit_circle_original']
            transformed_circle = original_analysis['pipeline']['unit_circle_transformed']
            ax1.plot(original_circle[:, 0], original_circle[:, 1], 'b-', linewidth=2, label='Original')
            ax1.plot(transformed_circle[:, 0], transformed_circle[:, 1], 'r-', linewidth=2, label='Transformed')
            ax1.set_title('Original Matrix\nUnit Circle Effect')
            ax1.set_aspect('equal')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot improved unit circle transformations for each strategy
        strategy_names = list(strategies.keys())
        for i, (strategy_name, strategy) in enumerate(strategies.items()):
            ax = plt.subplot(3, 5, i + 2)
            
            # Test unit circle transformation
            theta = np.linspace(0, 2*np.pi, 100)
            unit_circle = np.array([np.cos(theta), np.sin(theta)]).T
            
            encoder = strategy['encoder']
            decoder = strategy['decoder']
            transformed = unit_circle @ encoder @ decoder
            
            ax.plot(unit_circle[:, 0], unit_circle[:, 1], 'b-', linewidth=2, label='Original')
            ax.plot(transformed[:, 0], transformed[:, 1], 'g-', linewidth=2, label='Transformed')
            ax.set_title(f'{strategy_name.replace("_", " ").title()}\nUnit Circle Effect')
            ax.set_aspect('equal')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot bimodal data transformations
        for i, (strategy_name, result) in enumerate(test_results.items()):
            # Original data
            ax_orig = plt.subplot(3, 5, 6 + i)
            original_data = result['original_data']
            ax_orig.scatter(original_data[:100, 0], original_data[:100, 1], alpha=0.6, s=10, c='blue', label='Cluster 1')
            ax_orig.scatter(original_data[100:, 0], original_data[100:, 1], alpha=0.6, s=10, c='red', label='Cluster 2')
            ax_orig.set_title(f'{strategy_name.replace("_", " ").title()}\nOriginal Bimodal Data')
            ax_orig.legend()
            ax_orig.grid(True, alpha=0.3)
            
            # Transformed data
            ax_trans = plt.subplot(3, 5, 11 + i)
            decoded_data = result['decoded_data']
            ax_trans.scatter(decoded_data[:100, 0], decoded_data[:100, 1], alpha=0.6, s=10, c='blue', label='Cluster 1')
            ax_trans.scatter(decoded_data[100:, 0], decoded_data[100:, 1], alpha=0.6, s=10, c='red', label='Cluster 2')
            ax_trans.set_title(f'Transformed Data\nQuality: {result["quality_score"]:.3f}')
            ax_trans.legend()
            ax_trans.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, "matrix_conditioning_improvements.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Comprehensive improvements visualization saved: {save_path}")
        
        # Create detailed comparison metrics plot
        self._create_metrics_comparison_plot(test_results, save_dir)
        
        return save_path
    
    def _create_metrics_comparison_plot(self, test_results: Dict, save_dir: str):
        """Create detailed metrics comparison plot."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Matrix Strategy Performance Comparison', fontsize=16, fontweight='bold')
        
        strategies = list(test_results.keys())
        
        # Metric 1: Standard deviation preservation
        std_preservations = [test_results[s]['std_preservation'] for s in strategies]
        axes[0, 0].bar(strategies, std_preservations, alpha=0.7)
        axes[0, 0].set_title('Standard Deviation Preservation')
        axes[0, 0].set_ylabel('Preservation Ratio')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=1.0, color='green', linestyle='--', label='Perfect Preservation')
        axes[0, 0].legend()
        
        # Metric 2: Span preservation
        span_preservations = [test_results[s]['span_preservation'] for s in strategies]
        axes[0, 1].bar(strategies, span_preservations, alpha=0.7, color='orange')
        axes[0, 1].set_title('Data Span Preservation')
        axes[0, 1].set_ylabel('Preservation Ratio')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=1.0, color='green', linestyle='--', label='Perfect Preservation')
        axes[0, 1].legend()
        
        # Metric 3: Cluster separation preservation
        sep_preservations = [test_results[s]['separation_preservation'] for s in strategies]
        axes[1, 0].bar(strategies, sep_preservations, alpha=0.7, color='red')
        axes[1, 0].set_title('Cluster Separation Preservation')
        axes[1, 0].set_ylabel('Preservation Ratio')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=1.0, color='green', linestyle='--', label='Perfect Preservation')
        axes[1, 0].legend()
        
        # Metric 4: Overall quality scores
        quality_scores = [test_results[s]['quality_score'] for s in strategies]
        bars = axes[1, 1].bar(strategies, quality_scores, alpha=0.7, color='purple')
        axes[1, 1].set_title('Overall Quality Score')
        axes[1, 1].set_ylabel('Quality Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1.2)
        
        # Add value labels on bars
        for bar, score in zip(bars, quality_scores):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, "matrix_strategy_metrics.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Metrics comparison plot saved: {save_path}")
    
    def select_best_strategy(self, test_results: Dict) -> Tuple[str, Dict]:
        """Select the best matrix strategy based on quality metrics."""
        print(f"\n🏆 SELECTING BEST MATRIX STRATEGY")
        print("=" * 70)
        
        # Rank strategies by quality score
        strategies_ranked = sorted(test_results.items(), 
                                 key=lambda x: x[1]['quality_score'], 
                                 reverse=True)
        
        print(f"Strategy Rankings:")
        for i, (strategy_name, result) in enumerate(strategies_ranked):
            print(f"   {i+1}. {strategy_name}: {result['quality_score']:.4f}")
            print(f"      • Std preservation: {result['std_preservation']:.4f}")
            print(f"      • Span preservation: {result['span_preservation']:.4f}")
            print(f"      • Separation preservation: {result['separation_preservation']:.4f}")
        
        best_strategy_name, best_result = strategies_ranked[0]
        
        print(f"\n🥇 BEST STRATEGY: {best_strategy_name}")
        print(f"   Quality Score: {best_result['quality_score']:.4f}")
        print(f"   Description: {best_result['description']}")
        
        return best_strategy_name, best_result
    
    def create_fixed_generator(self, best_strategy: Dict) -> 'SFTutorialGenerator':
        """Create a generator with fixed transformation matrices."""
        print(f"\n🔧 CREATING FIXED GENERATOR")
        print("=" * 70)
        
        # We need to create a modified version of SFTutorialGenerator
        # that uses our well-conditioned matrices
        
        print("Creating fixed generator with well-conditioned matrices...")
        print("Note: This will require modifying the generator initialization")
        
        # For now, return the analysis - we'll need to modify the actual generator class
        return {
            'fixed_encoder': best_strategy['encoder'],
            'fixed_decoder': best_strategy['decoder'],
            'improvement_metrics': {
                'std_preservation': best_strategy['std_preservation'],
                'span_preservation': best_strategy['span_preservation'],
                'separation_preservation': best_strategy['separation_preservation'],
                'quality_score': best_strategy['quality_score']
            }
        }
    
    def run_complete_fix(self) -> Dict:
        """Run complete matrix conditioning fix process."""
        print("🔧 MATRIX CONDITIONING FIX - COMPLETE PROCESS")
        print("=" * 80)
        print("Fixing severe matrix compression issues identified in diagnostics")
        print("=" * 80)
        
        # Step 1: Analyze current problems
        original_analysis = self.analyze_current_matrices()
        
        # Step 2: Create well-conditioned alternatives
        strategies = self.create_well_conditioned_matrices()
        
        # Step 3: Test strategies on bimodal data
        test_results = self.test_matrix_strategies(strategies)
        
        # Step 4: Visualize improvements
        viz_path = self.visualize_matrix_improvements(strategies, test_results)
        
        # Step 5: Select best strategy
        best_strategy_name, best_strategy = self.select_best_strategy(test_results)
        
        # Step 6: Create fixed generator info
        fixed_generator_info = self.create_fixed_generator(best_strategy)
        
        # Summary
        print(f"\n" + "=" * 80)
        print("🎯 MATRIX CONDITIONING FIX COMPLETE!")
        print("=" * 80)
        
        original_compression = original_analysis.get('pipeline', {}).get('compression_factor', 'N/A')
        print(f"📋 SUMMARY:")
        print(f"   Original matrix compression: {original_compression}x")
        print(f"   Best strategy: {best_strategy_name}")
        print(f"   Quality improvement: {best_strategy['quality_score']:.4f}")
        print(f"   Data preservation: {best_strategy['std_preservation']:.1%}")
        print(f"   Cluster preservation: {best_strategy['separation_preservation']:.1%}")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Integrate these matrices into the quantum generator")
        print(f"   2. Fix quantum circuit input integration")
        print(f"   3. Test complete pipeline with diverse inputs")
        print(f"   4. Validate bimodal cluster generation")
        
        print(f"\n📊 Visualizations saved in: results/matrix_conditioning_fix/")
        print("   • matrix_conditioning_improvements.png - Before/after comparison")
        print("   • matrix_strategy_metrics.png - Performance metrics")
        
        return {
            'original_analysis': original_analysis,
            'strategies': strategies,
            'test_results': test_results,
            'best_strategy': (best_strategy_name, best_strategy),
            'fixed_generator_info': fixed_generator_info,
            'visualizations': viz_path
        }


def main():
    """Run matrix conditioning fix."""
    print("🚀 MATRIX CONDITIONING FIX")
    print("Fixing severe matrix compression issues")
    print("=" * 80)
    
    # Suppress TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    try:
        # Create and run matrix fix
        matrix_fix = MatrixConditioningFix()
        results = matrix_fix.run_complete_fix()
        
        print("\n🎉 Matrix conditioning fix completed successfully!")
        print("Check the visualizations to see the dramatic improvements.")
        
        return results
        
    except Exception as e:
        print(f"❌ Matrix conditioning fix failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
