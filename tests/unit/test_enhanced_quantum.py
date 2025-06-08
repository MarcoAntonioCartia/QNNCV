"""
Comprehensive test script for enhanced quantum QGAN components.
Tests the sophisticated quantum discriminator, generator, and training framework.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
import numpy as np
from utils import load_synthetic_data, plot_results, compute_wasserstein_distance
from main_qgan import QGAN

def test_enhanced_quantum_discriminator():
    """Test the enhanced quantum discriminator with sophisticated circuit."""
    print("Testing Enhanced Quantum Discriminator")
    print("======================================")
    
    try:
        from NN.quantum_continuous_discriminator import QuantumContinuousDiscriminator
        
        # Initialize enhanced discriminator
        discriminator = QuantumContinuousDiscriminator(
            n_qumodes=4, 
            input_dim=4, 
            cutoff_dim=10
        )
        
        print(f"  Discriminator initialized successfully")
        print(f"  - Quantum modes: {discriminator.n_qumodes}")
        print(f"  - Input dimension: {discriminator.input_dim}")
        print(f"  - Cutoff dimension: {discriminator.cutoff_dim}")
        
        # Test parameter structure
        trainable_vars = discriminator.trainable_variables
        print(f"  - Trainable variables: {len(trainable_vars)}")
        
        # Test single sample discrimination
        x_single = tf.random.normal([4])
        try:
            prob_single = discriminator.discriminate_single(x_single)
            print(f"  Single sample discrimination: {prob_single.numpy():.4f}")
        except Exception as e:
            print(f"  Single sample test failed: {e}")
            return False
        
        # Test batch discrimination
        x_batch = tf.random.normal([3, 4])
        try:
            probs_batch = discriminator.discriminate(x_batch)
            print(f" Batch discrimination shape: {probs_batch.shape}")
            print(f"  - Probabilities: {probs_batch.numpy().flatten()}")
        except Exception as e:
            print(f" Batch discrimination failed: {e}")
            return False
        
        # Test gradient flow
        try:
            with tf.GradientTape() as tape:
                output = discriminator.discriminate(x_batch)
                loss = tf.reduce_mean(output)
            
            gradients = tape.gradient(loss, discriminator.trainable_variables)
            non_none_grads = [g for g in gradients if g is not None]
            print(f" Gradient flow: {len(non_none_grads)}/{len(gradients)} gradients computed")
        except Exception as e:
            print(f" Gradient flow test failed: {e}")
            return False
        
        print(" Enhanced quantum discriminator tests passed!")
        return True
        
    except ImportError as e:
        print(f" Import failed (likely missing Strawberry Fields): {e}")
        return False
    except Exception as e:
        print(f" Discriminator test failed: {e}")
        return False

def test_enhanced_quantum_generator():
    """Test the enhanced quantum generator with sophisticated circuit."""
    print("\nTesting Enhanced Quantum Generator")
    print("==================================")
    
    try:
        from NN.quantum_continuous_generator_enhanced import QuantumContinuousGeneratorEnhanced
        
        # Initialize enhanced generator
        generator = QuantumContinuousGeneratorEnhanced(
            n_qumodes=4, 
            latent_dim=8, 
            cutoff_dim=10
        )
        
        print(f" Generator initialized successfully")
        print(f"  - Quantum modes: {generator.n_qumodes}")
        print(f"  - Latent dimension: {generator.latent_dim}")
        print(f"  - Cutoff dimension: {generator.cutoff_dim}")
        
        # Test parameter structure
        trainable_vars = generator.trainable_variables
        print(f"  - Trainable variables: {len(trainable_vars)}")
        
        # Test single sample generation
        z_single = tf.random.normal([8])
        try:
            sample_single = generator.generate_single(z_single)
            print(f" Single sample generation shape: {sample_single.shape}")
        except Exception as e:
            print(f" Single sample test failed: {e}")
            return False
        
        # Test batch generation
        z_batch = tf.random.normal([3, 8])
        try:
            samples_batch = generator.generate(z_batch)
            print(f" Batch generation shape: {samples_batch.shape}")
            print(f"  - Sample statistics: mean={tf.reduce_mean(samples_batch):.4f}, "
                  f"std={tf.math.reduce_std(samples_batch):.4f}")
        except Exception as e:
            print(f" Batch generation failed: {e}")
            return False
        
        # Test gradient flow
        try:
            with tf.GradientTape() as tape:
                output = generator.generate(z_batch)
                loss = tf.reduce_mean(tf.square(output))
            
            gradients = tape.gradient(loss, generator.trainable_variables)
            non_none_grads = [g for g in gradients if g is not None]
            print(f" Gradient flow: {len(non_none_grads)}/{len(gradients)} gradients computed")
        except Exception as e:
            print(f" Gradient flow test failed: {e}")
            return False
        
        print(" Enhanced quantum generator tests passed!")
        return True
        
    except ImportError as e:
        print(f" Import failed (likely missing Strawberry Fields): {e}")
        return False
    except Exception as e:
        print(f" Generator test failed: {e}")
        return False

def test_quantum_inspired_fallback():
    """Test the quantum-inspired fallback generator."""
    print("\nTesting Quantum-Inspired Fallback Generator")
    print("===========================================")
    
    try:
        from NN.quantum_continuous_generator_enhanced import QuantumContinuousGeneratorSimple
        
        # Initialize fallback generator
        generator = QuantumContinuousGeneratorSimple(n_qumodes=4, latent_dim=8)
        
        print(f" Fallback generator initialized successfully")
        print(f"  - Quantum modes: {generator.n_qumodes}")
        print(f"  - Latent dimension: {generator.latent_dim}")
        
        # Test generation
        z_batch = tf.random.normal([5, 8])
        samples = generator.generate(z_batch)
        
        print(f" Fallback generation shape: {samples.shape}")
        print(f"  - Trainable variables: {len(generator.trainable_variables)}")
        
        # Test gradient flow
        with tf.GradientTape() as tape:
            output = generator.generate(z_batch)
            loss = tf.reduce_mean(tf.square(output))
        
        gradients = tape.gradient(loss, generator.trainable_variables)
        non_none_grads = [g for g in gradients if g is not None]
        print(f" Fallback gradient flow: {len(non_none_grads)}/{len(gradients)} gradients computed")
        
        print(" Quantum-inspired fallback tests passed!")
        return True
        
    except Exception as e:
        print(f" Fallback test failed: {e}")
        return False

def test_enhanced_training_framework():
    """Test the enhanced QGAN training framework."""
    print("\nTesting Enhanced QGAN Training Framework")
    print("========================================")
    
    try:
        # Use classical components for reliable testing
        from NN.classical_generator import ClassicalGenerator
        from NN.classical_discriminator import ClassicalDiscriminator
        
        # Create test data
        data = load_synthetic_data(dataset_type="moons", num_samples=200)
        print(f" Test data loaded: {data.shape}")
        
        # Initialize components
        generator = ClassicalGenerator(latent_dim=4, output_dim=2)
        discriminator = ClassicalDiscriminator(input_dim=2)
        
        # Initialize enhanced QGAN
        qgan = QGAN(
            generator=generator,
            discriminator=discriminator,
            latent_dim=4,
            generator_lr=0.001,
            discriminator_lr=0.001,
            gradient_clip_norm=1.0
        )
        
        print(f" Enhanced QGAN initialized")
        print(f"  - Gradient clipping norm: {qgan.gradient_clip_norm}")
        
        # Test single training step
        real_samples = data[:16]  # Small batch for testing
        try:
            metrics = qgan.train_step(real_samples, use_wasserstein=False)
            print(f" Training step completed")
            print(f"  - D loss: {metrics['d_loss']:.4f}")
            print(f"  - G loss: {metrics['g_loss']:.4f}")
            print(f"  - Stability metric: {metrics['stability_metric']:.4f}")
        except Exception as e:
            print(f" Training step failed: {e}")
            return False
        
        # Test short training run
        print("Running short training test (5 epochs)...")
        try:
            history = qgan.train(
                data, 
                epochs=5, 
                batch_size=32, 
                use_wasserstein=False,
                verbose=True,
                save_interval=2
            )
            print(f" Training completed successfully")
            print(f"  - Final D loss: {history['d_loss'][-1]:.4f}")
            print(f"  - Final G loss: {history['g_loss'][-1]:.4f}")
        except Exception as e:
            print(f" Training failed: {e}")
            return False
        
        print(" Enhanced training framework tests passed!")
        return True
        
    except Exception as e:
        print(f" Training framework test failed: {e}")
        return False

def test_quantum_qgan_integration():
    """Test integration of quantum components with enhanced training."""
    print("\nTesting Quantum QGAN Integration")
    print("================================")
    
    try:
        # Try to use quantum components
        from NN.quantum_continuous_generator_enhanced import QuantumContinuousGeneratorSimple
        from NN.classical_discriminator import ClassicalDiscriminator
        
        # Create test data
        data = load_synthetic_data(dataset_type="circles", num_samples=100)
        print(f" Test data loaded: {data.shape}")
        
        # Initialize hybrid QGAN (quantum generator + classical discriminator)
        generator = QuantumContinuousGeneratorSimple(n_qumodes=2, latent_dim=6)
        discriminator = ClassicalDiscriminator(input_dim=2)
        
        qgan = QGAN(
            generator=generator,
            discriminator=discriminator,
            latent_dim=6,
            generator_lr=0.0005,  # Lower LR for quantum stability
            discriminator_lr=0.001,
            gradient_clip_norm=0.5  # Tighter clipping for quantum
        )
        
        print(f" Hybrid quantum QGAN initialized")
        
        # Test training
        print("Running hybrid quantum training test (3 epochs)...")
        try:
            history = qgan.train(
                data, 
                epochs=3, 
                batch_size=16, 
                use_wasserstein=False,
                verbose=True,
                save_interval=1
            )
            print(f" Hybrid quantum training completed")
            print(f"  - Generator type: Quantum-inspired")
            print(f"  - Discriminator type: Classical")
            print(f"  - Final stability: {history['stability_metric'][-1]:.4f}")
        except Exception as e:
            print(f" Hybrid training failed: {e}")
            return False
        
        # Test sample quality
        z_test = tf.random.normal([50, 6])
        generated_samples = generator.generate(z_test)
        
        # Compute quality metric
        wd = compute_wasserstein_distance(data[:50], generated_samples)
        print(f" Sample quality (Wasserstein distance): {wd:.4f}")
        
        print(" Quantum QGAN integration tests passed!")
        return True
        
    except ImportError as e:
        print(f" Quantum components not available: {e}")
        return False
    except Exception as e:
        print(f" Integration test failed: {e}")
        return False

def run_comprehensive_tests():
    """Run all enhanced quantum QGAN tests."""
    print("Enhanced Quantum QGAN Test Suite")
    print("================================")
    print()
    
    test_results = []
    
    # Test individual components
    test_results.append(("Enhanced Quantum Discriminator", test_enhanced_quantum_discriminator()))
    test_results.append(("Enhanced Quantum Generator", test_enhanced_quantum_generator()))
    test_results.append(("Quantum-Inspired Fallback", test_quantum_inspired_fallback()))
    test_results.append(("Enhanced Training Framework", test_enhanced_training_framework()))
    test_results.append(("Quantum QGAN Integration", test_quantum_qgan_integration()))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<35} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n All tests passed! Enhanced quantum QGAN is ready for use.")
        print("\nNext steps:")
        print("1. Install quantum dependencies: pip install strawberryfields")
        print("2. Test with true quantum implementations")
        print("3. Run molecular generation experiments")
        print("4. Benchmark quantum vs classical performance")
    else:
        print(f"\n  {total - passed} tests failed. Check the implementations.")
        print("\nRecommendations:")
        print("1. Install missing dependencies")
        print("2. Check quantum circuit implementations")
        print("3. Verify gradient flow through quantum components")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_tests()
    
    if success:
        print("\n" + "="*50)
        print("ENHANCED QUANTUM QGAN READY FOR DEPLOYMENT")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("TESTS FAILED - REVIEW IMPLEMENTATIONS")
        print("="*50)
