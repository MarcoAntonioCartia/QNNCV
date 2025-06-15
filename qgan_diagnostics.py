# advanced_qgan_diagnostic.py
"""Advanced QGAN diagnostic with proper dimensional handling."""

import sys
import os

# Add src to path
current_dir = os.path.dirname(__file__)
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import wasserstein_distance

def create_bimodal_target():
    """Create the target bimodal distribution."""
    np.random.seed(42)
    cluster1 = np.random.normal([-0.8, -0.8], 0.15, (250, 2))
    cluster2 = np.random.normal([0.8, 0.8], 0.15, (250, 2))
    return tf.constant(np.vstack([cluster1, cluster2]), dtype=tf.float32)

def create_dimensional_adapter():
    """Create a proper 4D → 2D adapter."""
    class StaticAdapter:
        def __init__(self):
            # Create fixed transformation matrix for 4D → 2D
            # This should preserve multimodal structure
            self.transform_matrix = tf.constant([
                [0.7, 0.3, 0.0, 0.0],  # X1 = 0.7*mode0 + 0.3*mode1
                [0.0, 0.0, 0.7, 0.3]   # X2 = 0.7*mode2 + 0.3*mode3
            ], dtype=tf.float32)
        
        def __call__(self, x):
            # x shape: (batch, 4) → (batch, 2)
            return tf.matmul(x, tf.transpose(self.transform_matrix))
    
    return StaticAdapter()

def analyze_quantum_state_diversity(generator, n_tests=10):
    """Analyze how diverse the quantum states are."""
    print("🔬 Quantum State Diversity Analysis")
    print("-" * 40)
    
    diverse_inputs = [
        tf.random.normal([1, 4], stddev=0.5),
        tf.random.normal([1, 4], stddev=1.0),
        tf.random.normal([1, 4], stddev=2.0),
        tf.constant([[-2.0, -2.0, 2.0, 2.0]], dtype=tf.float32),
        tf.constant([[2.0, 2.0, -2.0, -2.0]], dtype=tf.float32),
    ]
    
    quantum_outputs = []
    for i, z in enumerate(diverse_inputs):
        # Get raw quantum output (4D)
        quantum_out = generator.generate(z)
        quantum_outputs.append(quantum_out.numpy().flatten())
        print(f"  Input {i}: {z.numpy().flatten()}")
        print(f"  Output {i}: {quantum_out.numpy().flatten()}")
    
    # Compute diversity
    quantum_array = np.array(quantum_outputs)
    diversity_score = np.std(quantum_array, axis=0).mean()
    print(f"\n✓ Quantum output diversity: {diversity_score:.4f}")
    
    return diversity_score, quantum_outputs

def test_measurement_strategy_impact(generator, adapter, n_samples=200):
    """Test how measurement strategy affects output diversity."""
    print("\n🎯 Measurement Strategy Impact Analysis")
    print("-" * 50)
    
    # Generate diverse latent inputs
    z_samples = tf.random.normal([n_samples, 4], stddev=1.0)
    
    # Get raw quantum measurements (4D)
    raw_quantum = generator.generate(z_samples)
    
    # Apply dimensional adapter (4D → 2D)
    adapted_output = adapter(raw_quantum)
    
    print(f"Raw quantum shape: {raw_quantum.shape}")
    print(f"Adapted output shape: {adapted_output.shape}")
    print(f"Raw quantum range: [{tf.reduce_min(raw_quantum):.3f}, {tf.reduce_max(raw_quantum):.3f}]")
    print(f"Adapted range: [{tf.reduce_min(adapted_output):.3f}, {tf.reduce_max(adapted_output):.3f}]")
    
    # Analyze per-dimension
    raw_stds = tf.math.reduce_std(raw_quantum, axis=0)
    adapted_stds = tf.math.reduce_std(adapted_output, axis=0)
    
    print(f"\nRaw quantum std per mode: {raw_stds.numpy()}")
    print(f"Adapted std per dimension: {adapted_stds.numpy()}")
    
    # Information preservation
    raw_total_var = tf.reduce_sum(tf.square(raw_stds))
    adapted_total_var = tf.reduce_sum(tf.square(adapted_stds))
    info_preservation = adapted_total_var / raw_total_var
    
    print(f"Information preservation: {info_preservation.numpy():.3f}")
    
    return raw_quantum, adapted_output, info_preservation

def detect_multimodal_structure(data, max_clusters=5):
    """Detect multimodal structure in data."""
    print("\n🔍 Multimodal Structure Detection")
    print("-" * 40)
    
    data_np = data.numpy() if hasattr(data, 'numpy') else data
    
    best_score = -1
    best_n_clusters = 1
    best_labels = np.zeros(len(data_np))
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_np)
        
        if len(np.unique(labels)) > 1:
            score = silhouette_score(data_np, labels)
            print(f"  {n_clusters} clusters: silhouette = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_labels = labels
    
    print(f"✓ Best clustering: {best_n_clusters} clusters (score: {best_score:.4f})")
    
    # Analyze cluster properties
    for i in range(best_n_clusters):
        cluster_data = data_np[best_labels == i]
        cluster_center = np.mean(cluster_data, axis=0)
        cluster_size = len(cluster_data)
        print(f"  Cluster {i}: center={cluster_center}, size={cluster_size}")
    
    return best_n_clusters, best_score, best_labels

def comprehensive_mode_collapse_test():
    """Comprehensive test for mode collapse issues."""
    print("🔬 COMPREHENSIVE QGAN MODE COLLAPSE DIAGNOSTIC")
    print("=" * 60)
    
    # Create target data
    real_data = create_bimodal_target()
    adapter = create_dimensional_adapter()
    
    print(f"Target data: {real_data.shape}")
    target_clusters, target_score, _ = detect_multimodal_structure(real_data)
    
    # Import quantum generator
    try:
        from models.generators.quantum_sf_generator import QuantumSFGenerator
        
        # Test different generator configurations
        configs = [
            {"n_modes": 2, "layers": 1, "name": "Simple (2 modes, 1 layer)"},
            {"n_modes": 4, "layers": 2, "name": "Medium (4 modes, 2 layers)"},
            {"n_modes": 4, "layers": 4, "name": "Complex (4 modes, 4 layers)"},
        ]
        
        results = {}
        
        for config in configs:
            print(f"\n{'='*20} {config['name']} {'='*20}")
            
            try:
                # Create generator
                generator = QuantumSFGenerator(
                    n_modes=config['n_modes'], 
                    latent_dim=4, 
                    layers=config['layers']
                )
                
                print(f"✓ Generator created: {len(generator.trainable_variables)} parameters")
                
                # Test quantum state diversity
                diversity_score, _ = analyze_quantum_state_diversity(generator)
                
                # Test measurement strategy
                raw_quantum, adapted_output, info_preservation = test_measurement_strategy_impact(
                    generator, adapter, n_samples=300
                )
                
                # Detect structure in generated data
                gen_clusters, gen_score, gen_labels = detect_multimodal_structure(adapted_output)
                
                # Compute mode collapse metrics
                cluster_ratio = gen_clusters / target_clusters
                quality_ratio = gen_score / max(target_score, 0.1)
                mode_collapse_score = (cluster_ratio + quality_ratio) / 2
                
                # Store results
                results[config['name']] = {
                    'quantum_diversity': diversity_score,
                    'info_preservation': info_preservation.numpy(),
                    'generated_clusters': gen_clusters,
                    'cluster_quality': gen_score,
                    'mode_collapse_score': mode_collapse_score,
                    'raw_quantum_data': raw_quantum,
                    'adapted_output': adapted_output
                }
                
                print(f"\n📊 RESULTS for {config['name']}:")
                print(f"   Quantum diversity: {diversity_score:.4f}")
                print(f"   Info preservation: {info_preservation.numpy():.4f}")
                print(f"   Generated clusters: {gen_clusters} (target: {target_clusters})")
                print(f"   Cluster quality: {gen_score:.4f} (target: {target_score:.4f})")
                print(f"   Mode collapse score: {mode_collapse_score:.4f}")
                
                if mode_collapse_score < 0.3:
                    print("   🚨 SEVERE MODE COLLAPSE")
                elif mode_collapse_score < 0.6:
                    print("   ⚠️  MODERATE MODE COLLAPSE")
                else:
                    print("   ✅ GOOD MULTIMODAL STRUCTURE")
                
            except Exception as e:
                print(f"❌ Failed for {config['name']}: {e}")
                continue
        
        # Generate comparison visualization
        create_comparison_visualization(real_data, results)
        
        # Generate recommendations
        generate_recommendations(results, target_clusters)
        
        return results
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_comparison_visualization(real_data, results):
    """Create visualization comparing all configurations."""
    try:
        n_configs = len(results)
        fig, axes = plt.subplots(2, n_configs + 1, figsize=(4 * (n_configs + 1), 8))
        
        # Plot real data
        real_np = real_data.numpy()
        axes[0, 0].scatter(real_np[:, 0], real_np[:, 1], alpha=0.6, s=20, color='blue')
        axes[0, 0].set_title('Real Data (Target)')
        axes[0, 0].set_xlabel('X1')
        axes[0, 0].set_ylabel('X2')
        
        axes[1, 0].hist2d(real_np[:, 0], real_np[:, 1], bins=20, alpha=0.7)
        axes[1, 0].set_title('Real Data Density')
        
        # Plot each configuration
        for i, (name, result) in enumerate(results.items()):
            col = i + 1
            adapted_data = result['adapted_output'].numpy()
            
            # Scatter plot
            axes[0, col].scatter(adapted_data[:, 0], adapted_data[:, 1], 
                               alpha=0.6, s=20, color='red')
            axes[0, col].set_title(f'{name}\nScore: {result["mode_collapse_score"]:.3f}')
            axes[0, col].set_xlabel('X1')
            axes[0, col].set_ylabel('X2')
            
            # Density plot
            axes[1, col].hist2d(adapted_data[:, 0], adapted_data[:, 1], 
                              bins=20, alpha=0.7)
            axes[1, col].set_title(f'Generated Density')
        
        plt.tight_layout()
        plt.savefig('qgan_mode_collapse_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Visualization saved as 'qgan_mode_collapse_analysis.png'")
        
    except Exception as e:
        print(f"Visualization failed: {e}")

def generate_recommendations(results, target_clusters):
    """Generate specific recommendations based on results."""
    print(f"\n{'='*60}")
    print("🎯 RECOMMENDATIONS TO IMPROVE MULTIMODAL GENERATION")
    print(f"{'='*60}")
    
    best_config = max(results.items(), key=lambda x: x[1]['mode_collapse_score'])
    worst_config = min(results.items(), key=lambda x: x[1]['mode_collapse_score'])
    
    print(f"🏆 Best configuration: {best_config[0]} (score: {best_config[1]['mode_collapse_score']:.3f})")
    print(f"📉 Worst configuration: {worst_config[0]} (score: {worst_config[1]['mode_collapse_score']:.3f})")
    
    # Specific recommendations
    recommendations = []
    
    avg_info_preservation = np.mean([r['info_preservation'] for r in results.values()])
    if avg_info_preservation < 0.5:
        recommendations.append("🔧 CRITICAL: Improve dimensional adapter - too much information loss")
    
    avg_quantum_diversity = np.mean([r['quantum_diversity'] for r in results.values()])
    if avg_quantum_diversity < 0.1:
        recommendations.append("⚡ Increase quantum circuit expressivity (more layers/modes)")
    
    max_clusters = max([r['generated_clusters'] for r in results.values()])
    if max_clusters < target_clusters:
        recommendations.append("🎯 Implement advanced measurement strategy for multimodal preservation")
    
    if not recommendations:
        recommendations.append("✅ Configuration looks reasonable - try longer training")
    
    print("\n📋 Action Items:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    # Specific technical recommendations
    print(f"\n🔧 Technical Improvements:")
    print(f"   • Use {best_config[0].split('(')[1].split(')')[0]} configuration")
    print(f"   • Implement heterodyne measurement (X+iP) for complex amplitudes")
    print(f"   • Add cross-mode entangling operations")
    print(f"   • Use quantum Wasserstein loss (already implemented)")
    print(f"   • Initialize with larger parameter variance")

if __name__ == "__main__":
    results = comprehensive_mode_collapse_test()
    
    if results:
        print(f"\n🎉 Diagnostic completed! Check 'qgan_mode_collapse_analysis.png' for visual results.")
    else:
        print(f"\n❌ Diagnostic failed - check error messages above.")