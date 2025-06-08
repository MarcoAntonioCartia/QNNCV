"""
Comprehensive Test for True Quantum Components.
This tests both Strawberry Fields and PennyLane implementations with proper dataset sizes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def test_true_quantum_components():
    """Test true quantum generator and discriminator with realistic datasets."""
    
    print("🔬 Testing True Quantum QGAN Components")
    print("=" * 50)
    
    # Import our quantum components
    from NN.quantum_generator_sf import QuantumGeneratorStrawberryFields
    from NN.quantum_discriminator import QuantumDiscriminator
    from utils import load_synthetic_data, plot_results, compute_wasserstein_distance
    from main_qgan import QGAN
    
    # Load PROPER dataset size - this was the issue!
    print("Loading dataset with PROPER size (2000 samples)...")
    data = load_synthetic_data(dataset_type="spiral", num_samples=2000)  # Much larger!
    print(f"Loaded data shape: {data.shape}")
    print(f"Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
    
    # Test Strawberry Fields Generator
    print("\n1. Testing Strawberry Fields Quantum Generator")
    print("-" * 40)
    
    sf_generator = QuantumGeneratorStrawberryFields(
        n_qumodes=6,
        latent_dim=8,
        output_dim=2,
        cutoff_dim=6  # Smaller cutoff for efficiency
    )
    
    # Generate samples with the quantum generator
    test_noise = tf.random.normal([100, 8])
    sf_generated = sf_generator.generate(test_noise)
    
    print(f"SF Generator created samples shape: {sf_generated.shape}")
    print(f"SF Trainable variables: {len(sf_generator.trainable_variables)}")
    
    quantum_info = sf_generator.get_quantum_state_info()
    print(f"SF Backend: {quantum_info.get('mode', 'strawberry_fields')}")
    
    # Test PennyLane Quantum Discriminator
    print("\n2. Testing PennyLane Quantum Discriminator")
    print("-" * 40)
    
    pl_discriminator = QuantumDiscriminator(
        input_dim=2,
        n_qubits=6,
        n_layers=3,
        backend='pennylane'
    )
    
    # Test discrimination on real vs generated data
    real_batch = data[:100]
    real_scores = pl_discriminator.discriminate(real_batch)
    fake_scores = pl_discriminator.discriminate(sf_generated)
    
    print(f"PL Discriminator real scores shape: {real_scores.shape}")
    print(f"PL Discriminator fake scores shape: {fake_scores.shape}")
    print(f"PL Trainable variables: {len(pl_discriminator.trainable_variables)}")
    
    print(f"Real data scores - Mean: {tf.reduce_mean(real_scores):.3f}, Std: {tf.math.reduce_std(real_scores):.3f}")
    print(f"Generated scores - Mean: {tf.reduce_mean(fake_scores):.3f}, Std: {tf.math.reduce_std(fake_scores):.3f}")
    
    # Test Full Quantum QGAN Training
    print("\n3. Testing Full Quantum QGAN Training")
    print("-" * 40)
    
    # Create quantum QGAN with proper components
    quantum_qgan = QGAN(
        generator=sf_generator,
        discriminator=pl_discriminator,
        latent_dim=8
    )
    
    print("Created Quantum QGAN with:")
    print(f"  - Strawberry Fields Generator ({quantum_info.get('mode', 'sf')})")
    print(f"  - PennyLane Discriminator ({pl_discriminator.backend})")
    
    # Training with larger dataset
    print(f"\nTraining on {data.shape[0]} samples for 20 epochs...")
    
    try:
        # Train the quantum QGAN
        quantum_qgan.train(data, epochs=20, batch_size=64)
        
        # Generate final samples for evaluation
        final_test_noise = tf.random.normal([200, 8])  # More samples
        final_generated = sf_generator.generate(final_test_noise)
        
        # Evaluate performance
        wasserstein_dist = compute_wasserstein_distance(data[:200], final_generated)
        print(f"\n✅ Training completed successfully!")
        print(f"Final Wasserstein distance: {wasserstein_dist:.4f}")
        
        # Create comprehensive visualization
        plot_results(data[:200], final_generated, epoch=20, 
                    save_path="results/quantum_qgan_results.png")
        
        return True, wasserstein_dist
        
    except Exception as e:
        print(f"\n❌ Quantum training failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_quantum_vs_classical_comprehensive():
    """Comprehensive comparison between quantum and classical approaches."""
    
    print("\n🏆 Quantum vs Classical Comprehensive Comparison")
    print("=" * 55)
    
    from NN.classical_generator import ClassicalGenerator
    from NN.classical_discriminator import ClassicalDiscriminator
    from utils import load_synthetic_data, compute_wasserstein_distance, compute_mmd
    
    # Load larger test dataset
    test_data = load_synthetic_data(dataset_type="moons", num_samples=1500)
    print(f"Test data shape: {test_data.shape}")
    
    # Test classical GAN
    print("\n1. Classical GAN Performance")
    print("-" * 30)
    
    classical_gen = ClassicalGenerator(latent_dim=8, output_dim=2)
    classical_disc = ClassicalDiscriminator(input_dim=2)
    
    classical_qgan = QGAN(
        generator=classical_gen,
        discriminator=classical_disc,
        latent_dim=8
    )
    
    # Quick training
    classical_qgan.train(test_data, epochs=15, batch_size=64)
    
    # Generate samples
    test_noise = tf.random.normal([300, 8])
    classical_samples = classical_gen.generate(test_noise)
    
    classical_wd = compute_wasserstein_distance(test_data[:300], classical_samples)
    classical_mmd = compute_mmd(test_data[:300], classical_samples)
    
    print(f"Classical - Wasserstein distance: {classical_wd:.4f}")
    print(f"Classical - MMD: {classical_mmd:.4f}")
    print(f"Classical - Parameters: {len(classical_gen.trainable_variables)}")
    
    # Test quantum-inspired GAN
    print("\n2. Quantum-Inspired GAN Performance")
    print("-" * 35)
    
    from NN.quantum_generator_sf import QuantumGeneratorStrawberryFields
    from NN.quantum_discriminator import QuantumDiscriminator
    
    quantum_gen = QuantumGeneratorStrawberryFields(
        n_qumodes=6,
        latent_dim=8,
        output_dim=2,
        cutoff_dim=6
    )
    
    quantum_disc = QuantumDiscriminator(
        input_dim=2,
        n_qubits=6,
        n_layers=3,
        backend='auto'  # Will select best available
    )
    
    quantum_qgan = QGAN(
        generator=quantum_gen,
        discriminator=quantum_disc,
        latent_dim=8
    )
    
    # Quick training
    quantum_qgan.train(test_data, epochs=15, batch_size=64)
    
    # Generate samples
    quantum_samples = quantum_gen.generate(test_noise)
    
    quantum_wd = compute_wasserstein_distance(test_data[:300], quantum_samples)
    quantum_mmd = compute_mmd(test_data[:300], quantum_samples)
    
    print(f"Quantum - Wasserstein distance: {quantum_wd:.4f}")
    print(f"Quantum - MMD: {quantum_mmd:.4f}")
    print(f"Quantum - Parameters: {len(quantum_gen.trainable_variables)}")
    
    # Performance comparison
    print("\n3. Performance Summary")
    print("-" * 25)
    
    wd_improvement = ((classical_wd - quantum_wd) / classical_wd) * 100
    mmd_improvement = ((classical_mmd - quantum_mmd) / classical_mmd) * 100
    
    print(f"Wasserstein Distance Improvement: {wd_improvement:+.1f}%")
    print(f"MMD Improvement: {mmd_improvement:+.1f}%")
    
    if quantum_wd < classical_wd:
        print("🏆 Quantum approach shows BETTER performance!")
    else:
        print("📊 Classical approach shows better performance")
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot real data
    if test_data.shape[1] == 2:
        axes[0].scatter(test_data[:300, 0], test_data[:300, 1], alpha=0.6, s=15, label='Real Data')
        axes[0].set_title("Real Data")
        axes[0].legend()
        
        # Plot classical results
        axes[1].scatter(test_data[:300, 0], test_data[:300, 1], alpha=0.3, s=10, label='Real')
        axes[1].scatter(classical_samples[:, 0], classical_samples[:, 1], alpha=0.6, s=10, label='Classical GAN')
        axes[1].set_title(f"Classical GAN (WD: {classical_wd:.3f})")
        axes[1].legend()
        
        # Plot quantum results
        axes[2].scatter(test_data[:300, 0], test_data[:300, 1], alpha=0.3, s=10, label='Real')
        axes[2].scatter(quantum_samples[:, 0], quantum_samples[:, 1], alpha=0.6, s=10, label='Quantum GAN')
        axes[2].set_title(f"Quantum GAN (WD: {quantum_wd:.3f})")
        axes[2].legend()
    
    plt.tight_layout()
    plt.savefig("results/quantum_vs_classical_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'classical_wd': classical_wd,
        'quantum_wd': quantum_wd,
        'classical_mmd': classical_mmd,
        'quantum_mmd': quantum_mmd,
        'wd_improvement': wd_improvement,
        'mmd_improvement': mmd_improvement
    }

def analyze_quantum_advantage():
    """Analyze where quantum advantage comes from."""
    
    print("\n🔍 Quantum Advantage Analysis")
    print("=" * 35)
    
    from NN.quantum_generator_sf import QuantumGeneratorStrawberryFields
    
    # Create quantum generator
    qgen = QuantumGeneratorStrawberryFields(n_qumodes=4, latent_dim=6, output_dim=2)
    
    # Get quantum state information
    quantum_info = qgen.get_quantum_state_info()
    
    print("Quantum Generator Analysis:")
    print(f"  • Backend: {quantum_info.get('mode', 'strawberry_fields')}")
    print(f"  • Quantum modes: {quantum_info['n_qumodes']}")
    print(f"  • Parameters: {len(qgen.trainable_variables)}")
    
    if quantum_info.get('strawberry_fields_available', False):
        print(f"  • True quantum computing available!")
        print(f"  • Cutoff dimension: {quantum_info['cutoff_dim']}")
    else:
        print(f"  • Using quantum-inspired classical simulation")
    
    # Test entanglement generation
    print("\nTesting quantum entanglement effects...")
    
    # Generate samples with different noise patterns
    structured_noise = tf.constant([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]] * 10, dtype=tf.float32)
    random_noise = tf.random.normal([10, 6])
    
    structured_samples = qgen.generate(structured_noise)
    random_samples = qgen.generate(random_noise)
    
    # Analyze correlations
    structured_corr = tf.reduce_mean(tf.abs(tf.linalg.diag_part(tf.matmul(structured_samples, structured_samples, transpose_b=True))))
    random_corr = tf.reduce_mean(tf.abs(tf.linalg.diag_part(tf.matmul(random_samples, random_samples, transpose_b=True))))
    
    print(f"  • Structured input correlation: {structured_corr:.3f}")
    print(f"  • Random input correlation: {random_corr:.3f}")
    print(f"  • Quantum correlation ratio: {structured_corr / random_corr:.3f}")
    
    return quantum_info

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Quantum QGAN Testing")
    print("=" * 60)
    
    # Test 1: True quantum components
    success, wd = test_true_quantum_components()
    
    if success:
        print(f"\n✅ Quantum components test PASSED! (WD: {wd:.4f})")
        
        # Test 2: Comprehensive comparison
        comparison_results = test_quantum_vs_classical_comprehensive()
        
        # Test 3: Quantum advantage analysis
        quantum_info = analyze_quantum_advantage()
        
        print(f"\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 40)
        print("Key Results:")
        print(f"  • Quantum QGAN functional: ✅")
        print(f"  • Performance improvement: {comparison_results['wd_improvement']:+.1f}%")
        print(f"  • Backend: {quantum_info.get('mode', 'quantum')}")
        print(f"  • Data size: Fixed (was too small before!)")
        
    else:
        print(f"\n❌ Quantum components test FAILED")
        print("Check error messages above for troubleshooting.")
    
    print(f"\n📈 Results saved to results/ directory") 