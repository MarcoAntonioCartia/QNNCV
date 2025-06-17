"""
Simple Measurement Diagnosis

Direct test to answer your key questions:
1. Does oscillatory measurement help with bimodal generation?
2. Can we use a single parameter tensor effectively?
3. Are we inherently splitting modes or creating artificial separation?
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class SimpleGenerator:
    """Simple generator with single parameter tensor."""
    
    def __init__(self, n_params=50):
        # SINGLE PARAMETER TENSOR (as you requested)
        self.weights = tf.Variable(
            tf.random.normal([n_params], stddev=0.5),
            name="single_tensor"
        )
        print(f"✅ Single tensor: {self.weights.shape[0]} parameters")
    
    @property
    def trainable_variables(self):
        return [self.weights]
    
    def generate(self, n_samples, strategy):
        """Generate samples with different measurement strategies."""
        samples = []
        
        for i in range(n_samples):
            # Use different parts of parameter tensor
            idx = i % self.weights.shape[0]
            
            if strategy == 'oscillatory':
                # Your current oscillatory approach
                mean_n = tf.abs(self.weights[idx])
                var_n = tf.abs(self.weights[(idx + 1) % self.weights.shape[0]])
                
                x_phase = mean_n * np.pi * 2
                x = tf.sin(x_phase) * 3.0 + tf.cos(tf.sqrt(var_n + 1e-8)) * 0.5
                
                y_phase = (mean_n + 0.25) * np.pi * 2
                y = tf.cos(y_phase) * 3.0 + tf.sin(tf.sqrt(var_n + 1e-8)) * 0.5
                
            elif strategy == 'linear':
                # Simple linear mapping
                x = self.weights[idx] * 2.0
                y = self.weights[(idx + 1) % self.weights.shape[0]] * 2.0
                
            elif strategy == 'nonlinear':
                # Simple nonlinear but non-oscillatory
                x = tf.tanh(self.weights[idx]) * 3.0
                y = tf.tanh(self.weights[(idx + 1) % self.weights.shape[0]]) * 3.0
                
            elif strategy == 'raw':
                # Direct parameter usage
                x = self.weights[idx]
                y = self.weights[(idx + 1) % self.weights.shape[0]]
            
            samples.append(tf.stack([x, y]))
        
        return tf.stack(samples)

def analyze_bimodal_quality(samples, name):
    """Analyze how bimodal the distribution is."""
    data = samples.numpy()
    
    # K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_
    
    # Calculate metrics
    n1 = np.sum(labels == 0)
    n2 = np.sum(labels == 1)
    balance = min(n1, n2) / len(data)
    separation = np.linalg.norm(centers[0] - centers[1])
    
    # Intra-cluster variance
    var1 = np.mean(np.var(data[labels == 0], axis=0)) if n1 > 0 else 0
    var2 = np.mean(np.var(data[labels == 1], axis=0)) if n2 > 0 else 0
    avg_var = (var1 + var2) / 2
    
    print(f"\n{name}:")
    print(f"  Balance: {balance:.3f} (0.5 = perfect)")
    print(f"  Separation: {separation:.3f}")
    print(f"  Avg variance: {avg_var:.3f}")
    print(f"  Bimodal: {'YES' if balance > 0.2 and separation > 1.0 else 'NO'}")
    
    return {
        'balance': balance,
        'separation': separation,
        'variance': avg_var,
        'is_bimodal': balance > 0.2 and separation > 1.0,
        'data': data,
        'labels': labels
    }

def test_gradients(generator, strategy):
    """Test gradient flow."""
    with tf.GradientTape() as tape:
        samples = generator.generate(10, strategy)
        # Loss encouraging bimodal distribution
        loss = tf.reduce_mean(tf.square(samples - tf.constant([1.0, -1.0])))
    
    grads = tape.gradient(loss, generator.trainable_variables)
    grad_norm = tf.norm(grads[0]) if grads[0] is not None else 0.0
    
    print(f"  Gradient norm: {grad_norm:.6f}")
    return float(grad_norm) > 1e-8

def main_test():
    """Run the main diagnostic test."""
    print("=" * 60)
    print("MEASUREMENT STRATEGY DIAGNOSIS")
    print("=" * 60)
    
    # Test different parameter counts and strategies
    param_counts = [20, 50, 100]
    strategies = ['raw', 'linear', 'nonlinear', 'oscillatory']
    
    results = {}
    
    for n_params in param_counts:
        print(f"\n--- TESTING {n_params} PARAMETERS ---")
        
        generator = SimpleGenerator(n_params)
        param_results = {}
        
        for strategy in strategies:
            print(f"\n🧪 {strategy}:")
            
            # Generate samples
            samples = generator.generate(200, strategy)
            
            # Analyze quality
            analysis = analyze_bimodal_quality(samples, strategy)
            
            # Test gradients
            has_grads = test_gradients(generator, strategy)
            
            # Score
            score = analysis['balance'] + analysis['separation'] - analysis['variance'] * 0.5
            print(f"  Score: {score:.3f}")
            
            param_results[strategy] = {
                **analysis,
                'has_gradients': has_grads,
                'score': score
            }
        
        results[n_params] = param_results
    
    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS & ANSWERS")
    print("=" * 60)
    
    # Find best configuration
    best_score = -10
    best_config = None
    
    for n_params, strategies_results in results.items():
        for strategy, result in strategies_results.items():
            if result['score'] > best_score:
                best_score = result['score']
                best_config = (n_params, strategy)
    
    print(f"\n🏆 BEST: {best_config[1]} with {best_config[0]} parameters (score: {best_score:.3f})")
    
    # Answer key questions
    print(f"\n❓ Q1: Does oscillatory measurement help?")
    osc_scores = []
    other_scores = []
    
    for n_params, strategies_results in results.items():
        for strategy, result in strategies_results.items():
            if strategy == 'oscillatory':
                osc_scores.append(result['score'])
            else:
                other_scores.append(result['score'])
    
    osc_avg = np.mean(osc_scores) if osc_scores else 0
    other_avg = np.mean(other_scores) if other_scores else 0
    
    print(f"  Oscillatory avg: {osc_avg:.3f}")
    print(f"  Others avg: {other_avg:.3f}")
    print(f"  📊 ANSWER: Oscillatory {'HELPS' if osc_avg > other_avg else 'HURTS'} with bimodal generation")
    
    print(f"\n❓ Q2: Single parameter tensor working?")
    best_result = results[best_config[0]][best_config[1]]
    print(f"  Best config uses single tensor with {best_config[0]} parameters")
    print(f"  Has gradients: {best_result['has_gradients']}")
    print(f"  📊 ANSWER: {'YES' if best_result['has_gradients'] else 'NO'} - single tensor works!")
    
    print(f"\n❓ Q3: Mode separation real or artificial?")
    # Check correlation between params and separation
    separations = []
    param_counts_list = []
    
    for n_params, strategies_results in results.items():
        avg_sep = np.mean([r['separation'] for r in strategies_results.values()])
        separations.append(avg_sep)
        param_counts_list.append(n_params)
    
    correlation = np.corrcoef(param_counts_list, separations)[0, 1] if len(separations) > 1 else 0
    print(f"  Param-separation correlation: {correlation:.3f}")
    print(f"  📊 ANSWER: {'ARTIFICIAL' if abs(correlation) > 0.7 else 'INHERENT'} separation")
    
    print(f"\n📋 RECOMMENDATIONS:")
    print(f"  1. Use '{best_config[1]}' measurement strategy")
    print(f"  2. Single tensor with ~{best_config[0]} parameters")
    print(f"  3. {'Continue with oscillatory' if osc_avg > other_avg else 'Drop oscillatory mapping'}")
    print(f"  4. Focus on training stability next")
    
    # Create visualization
    create_plots(results, best_config)
    
    return results, best_config

def create_plots(results, best_config):
    """Create comparison plots."""
    strategies = ['raw', 'linear', 'nonlinear', 'oscillatory']
    param_counts = list(results.keys())
    
    fig, axes = plt.subplots(len(param_counts), len(strategies), 
                            figsize=(16, 4*len(param_counts)))
    
    if len(param_counts) == 1:
        axes = axes.reshape(1, -1)
    
    for i, n_params in enumerate(param_counts):
        for j, strategy in enumerate(strategies):
            ax = axes[i, j]
            result = results[n_params][strategy]
            
            data = result['data']
            labels = result['labels']
            
            # Scatter plot
            ax.scatter(data[:, 0], data[:, 1], c=labels, alpha=0.6, s=20)
            
            # Mark if this is the best config
            title = f"{strategy}\n{n_params}p"
            if (n_params, strategy) == best_config:
                title = f"⭐ {title} ⭐"
            
            ax.set_title(title)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('measurement_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Plots saved to 'measurement_comparison.png'")

if __name__ == "__main__":
    results, best_config = main_test()
    print(f"\n✅ Diagnosis complete! Check the plots for visual comparison.") 