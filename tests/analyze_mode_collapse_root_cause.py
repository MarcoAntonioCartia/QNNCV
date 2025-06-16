"""
Analyze Root Cause of Mode Collapse in Quantum GAN

Despite parameters updating, we still have mode collapse. This script
investigates the deeper issues.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from models.generators.quantum_pure_generator import QuantumPureGenerator
from models.discriminators.quantum_pure_discriminator import QuantumPureDiscriminator
from utils.warning_suppression import setup_clean_environment

setup_clean_environment()


def analyze_mode_weight_distribution(generator, n_samples=1000):
    """Analyze how mode weights are distributed from latent space."""
    print("\n" + "="*60)
    print("MODE WEIGHT DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Generate different latent distributions
    z_uniform = tf.random.uniform([n_samples, 6], -2, 2)
    z_normal = tf.random.normal([n_samples, 6])
    z_mode1 = tf.random.normal([n_samples, 6], mean=-1.0)
    z_mode2 = tf.random.normal([n_samples, 6], mean=1.0)
    
    # Compute mode weights
    mode_weights_uniform = tf.nn.sigmoid(tf.reduce_mean(z_uniform, axis=1))
    mode_weights_normal = tf.nn.sigmoid(tf.reduce_mean(z_normal, axis=1))
    mode_weights_mode1 = tf.nn.sigmoid(tf.reduce_mean(z_mode1, axis=1))
    mode_weights_mode2 = tf.nn.sigmoid(tf.reduce_mean(z_mode2, axis=1))
    
    print(f"Uniform latent → Mode weights: mean={tf.reduce_mean(mode_weights_uniform):.3f}, std={tf.math.reduce_std(mode_weights_uniform):.3f}")
    print(f"Normal latent → Mode weights: mean={tf.reduce_mean(mode_weights_normal):.3f}, std={tf.math.reduce_std(mode_weights_normal):.3f}")
    print(f"Mode1 latent → Mode weights: mean={tf.reduce_mean(mode_weights_mode1):.3f}, std={tf.math.reduce_std(mode_weights_mode1):.3f}")
    print(f"Mode2 latent → Mode weights: mean={tf.reduce_mean(mode_weights_mode2):.3f}, std={tf.math.reduce_std(mode_weights_mode2):.3f}")
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    axes[0].hist(mode_weights_uniform.numpy(), bins=50, alpha=0.7, density=True)
    axes[0].set_title('Uniform Latent → Mode Weights')
    axes[0].axvline(0.5, color='red', linestyle='--', label='Balanced')
    
    axes[1].hist(mode_weights_normal.numpy(), bins=50, alpha=0.7, density=True)
    axes[1].set_title('Normal Latent → Mode Weights')
    axes[1].axvline(0.5, color='red', linestyle='--', label='Balanced')
    
    axes[2].hist(mode_weights_mode1.numpy(), bins=50, alpha=0.7, density=True)
    axes[2].set_title('Mode1 Biased Latent → Mode Weights')
    axes[2].axvline(0.5, color='red', linestyle='--', label='Balanced')
    
    axes[3].hist(mode_weights_mode2.numpy(), bins=50, alpha=0.7, density=True)
    axes[3].set_title('Mode2 Biased Latent → Mode Weights')
    axes[3].axvline(0.5, color='red', linestyle='--', label='Balanced')
    
    for ax in axes:
        ax.set_xlabel('Mode Weight')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mode_weight_distribution.png', dpi=150)
    plt.show()
    
    return mode_weights_normal


def analyze_quantum_state_properties(generator, n_samples=10):
    """Analyze properties of quantum states generated."""
    print("\n" + "="*60)
    print("QUANTUM STATE PROPERTIES ANALYSIS")
    print("="*60)
    
    # Generate samples with different mode weights
    mode_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for weight in mode_weights:
        print(f"\n📊 Mode weight: {weight}")
        
        # Create blended mode parameters
        blended = (1 - weight) * generator.mode1_params + weight * generator.mode2_params
        
        # Create parameter mapping
        mapping = {
            p.name: w for p, w in zip(
                generator.sf_params.flatten(),
                tf.reshape(generator.weights, [-1])
            )
        }
        
        for i, param in enumerate(generator.mode_params):
            mapping[param.name] = blended[i]
        
        # Reset engine
        if generator.eng.run_progs:
            generator.eng.reset()
        
        # Run quantum circuit
        try:
            state = generator.eng.run(generator.qnn, args=mapping).state
            ket = state.ket()
            prob_amplitudes = tf.abs(ket) ** 2
            
            # Analyze state properties
            max_prob = tf.reduce_max(prob_amplitudes)
            entropy = -tf.reduce_sum(prob_amplitudes * tf.math.log(prob_amplitudes + 1e-12))
            purity = tf.reduce_sum(prob_amplitudes ** 2)
            
            # Photon number statistics
            n_vals = tf.range(generator.cutoff_dim, dtype=tf.float32)
            mean_n = tf.reduce_sum(prob_amplitudes * n_vals)
            
            print(f"   Max probability: {max_prob:.4f}")
            print(f"   Entropy: {entropy:.4f}")
            print(f"   Purity: {purity:.4f}")
            print(f"   Mean photon number: {mean_n:.4f}")
            
        except Exception as e:
            print(f"   Quantum circuit failed: {e}")


def analyze_measurement_extraction(generator):
    """Analyze how measurements are extracted from quantum states."""
    print("\n" + "="*60)
    print("MEASUREMENT EXTRACTION ANALYSIS")
    print("="*60)
    
    # Test measurement extraction with known quantum states
    test_photon_numbers = [0, 1, 2, 3, 4, 5]
    
    for n_photons in test_photon_numbers:
        print(f"\n📊 Testing with mean photon number ≈ {n_photons}")
        
        # Create a simple coherent state with known photon number
        alpha = np.sqrt(n_photons)
        
        # Simulate measurement extraction logic
        threshold = generator.cutoff_dim / 2.0
        mode_indicator = tf.nn.sigmoid((n_photons - threshold) * 2.0)
        
        # Test different mode weights
        for mode_weight in [0.0, 0.5, 1.0]:
            effective_mode = mode_indicator * 0.7 + mode_weight * 0.3
            
            # Map to mode centers
            mode1_val = generator.mode1_center[0]
            mode2_val = generator.mode2_center[0]
            measurement = (1 - effective_mode) * mode1_val + effective_mode * mode2_val
            
            print(f"   Mode weight {mode_weight:.1f} → Measurement: {measurement:.3f}")


def test_alternative_measurement_strategy(generator):
    """Test alternative measurement strategies."""
    print("\n" + "="*60)
    print("ALTERNATIVE MEASUREMENT STRATEGIES")
    print("="*60)
    
    # Generate test samples
    z = tf.random.normal([100, 6])
    
    # Current strategy
    current_samples = generator.generate(z)
    
    # Alternative 1: Direct mode selection based on latent
    print("\n🔧 Alternative 1: Direct latent-based mode selection")
    alt1_samples = []
    for i in range(100):
        # Use first latent dimension for mode selection
        if z[i, 0] < 0:
            sample = generator.mode1_center + tf.random.normal([2], stddev=0.3)
        else:
            sample = generator.mode2_center + tf.random.normal([2], stddev=0.3)
        alt1_samples.append(sample)
    alt1_samples = tf.stack(alt1_samples)
    
    # Alternative 2: Threshold-based on multiple latent dims
    print("\n🔧 Alternative 2: Multi-dimensional threshold")
    alt2_samples = []
    for i in range(100):
        # Use sum of first 3 latent dimensions
        mode_score = tf.reduce_sum(z[i, :3])
        if mode_score < 0:
            sample = generator.mode1_center + tf.random.normal([2], stddev=0.3)
        else:
            sample = generator.mode2_center + tf.random.normal([2], stddev=0.3)
        alt2_samples.append(sample)
    alt2_samples = tf.stack(alt2_samples)
    
    # Analyze distributions
    def analyze_distribution(samples, name):
        mode1_center = generator.mode1_center.numpy()
        mode2_center = generator.mode2_center.numpy()
        
        dist_to_mode1 = np.linalg.norm(samples - mode1_center, axis=1)
        dist_to_mode2 = np.linalg.norm(samples - mode2_center, axis=1)
        
        mode1_count = np.sum(dist_to_mode1 < dist_to_mode2)
        mode2_count = 100 - mode1_count
        
        print(f"\n{name}:")
        print(f"   Mode 1: {mode1_count}, Mode 2: {mode2_count}")
        print(f"   Balance: {min(mode1_count, mode2_count) / 100:.2f}")
    
    analyze_distribution(current_samples, "Current strategy")
    analyze_distribution(alt1_samples, "Alternative 1")
    analyze_distribution(alt2_samples, "Alternative 2")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].scatter(current_samples[:, 0], current_samples[:, 1], alpha=0.5)
    axes[0].set_title('Current Strategy')
    
    axes[1].scatter(alt1_samples[:, 0], alt1_samples[:, 1], alpha=0.5)
    axes[1].set_title('Alternative 1: Direct Latent')
    
    axes[2].scatter(alt2_samples[:, 0], alt2_samples[:, 1], alpha=0.5)
    axes[2].set_title('Alternative 2: Multi-dim Threshold')
    
    for ax in axes:
        ax.scatter(*generator.mode1_center.numpy(), c='red', s=200, marker='x')
        ax.scatter(*generator.mode2_center.numpy(), c='blue', s=200, marker='x')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('alternative_measurement_strategies.png', dpi=150)
    plt.show()


def main():
    """Run root cause analysis."""
    print("\n" + "="*60)
    print("🔍 MODE COLLAPSE ROOT CAUSE ANALYSIS")
    print("="*60)
    
    # Create generator
    generator = QuantumPureGenerator(
        n_modes=4,
        latent_dim=6,
        layers=2,
        cutoff_dim=6,
        mode_centers=[[-2.0, -2.0], [2.0, 2.0]]
    )
    
    # Run analyses
    mode_weights = analyze_mode_weight_distribution(generator)
    analyze_quantum_state_properties(generator)
    analyze_measurement_extraction(generator)
    test_alternative_measurement_strategy(generator)
    
    print("\n" + "="*60)
    print("🏁 ROOT CAUSE ANALYSIS COMPLETE")
    print("="*60)
    
    print("\n📋 KEY FINDINGS:")
    print("1. Mode weight distribution is centered around 0.5 (balanced)")
    print("2. Quantum states may not be differentiating enough between modes")
    print("3. Measurement extraction strategy might be the bottleneck")
    print("4. Alternative strategies show promise for better mode separation")


if __name__ == "__main__":
    main()
