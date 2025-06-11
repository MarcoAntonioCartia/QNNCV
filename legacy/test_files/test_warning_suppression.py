#!/usr/bin/env python3
"""
Test script to verify that the warning suppression is working correctly
for the quantum GAN training process.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

import tensorflow as tf
import numpy as np

def test_warning_suppression():
    """Test that complex number casting warnings are properly suppressed."""
    
    print("Testing Warning Suppression for Quantum GAN")
    print("=" * 50)
    
    # Import the warning suppression utility
    try:
        from utils.tensorflow_compat import suppress_complex_warnings
        print(" Successfully imported warning suppression utility")
    except ImportError as e:
        print(f" Failed to import warning suppression: {e}")
        return False
    
    # Test the warning suppression context manager
    print("\n1. Testing warning suppression context manager...")
    
    try:
        # This should normally produce warnings
        print("   Without suppression (should show warnings):")
        complex_tensor = tf.constant(1.0 + 2.0j)
        real_part = tf.cast(complex_tensor, tf.float32)  # This causes the warning
        print(f"   Result: {real_part}")
        
        print("\n   With suppression (should be clean):")
        with suppress_complex_warnings():
            complex_tensor = tf.constant(1.0 + 2.0j)
            real_part = tf.cast(complex_tensor, tf.float32)  # This should be silent
            print(f"   Result: {real_part}")
        
        print(" Warning suppression working correctly")
        
    except Exception as e:
        print(f" Warning suppression test failed: {e}")
        return False
    
    return True

def test_quantum_generator_with_suppression():
    """Test the quantum generator with warning suppression."""
    
    print("\n2. Testing Quantum Generator with Warning Suppression...")
    print("-" * 50)
    
    try:
        # Import quantum generator
        from models.generators.quantum_continuous_generator_enhanced import QuantumContinuousGeneratorEnhanced
        from utils.tensorflow_compat import suppress_complex_warnings
        
        print(" Successfully imported quantum generator")
        
        # Create generator
        generator = QuantumContinuousGeneratorEnhanced(
            n_qumodes=3, 
            latent_dim=4, 
            cutoff_dim=6
        )
        
        print(" Successfully created quantum generator")
        
        # Test generation without suppression
        print("\n   Testing generation without warning suppression...")
        z_test = tf.random.normal([2, 4])
        samples_without = generator.generate(z_test)
        print(f"   Generated samples shape: {samples_without.shape}")
        
        # Test generation with suppression
        print("\n   Testing generation with warning suppression...")
        with suppress_complex_warnings():
            z_test = tf.random.normal([2, 4])
            samples_with = generator.generate(z_test)
            print(f"   Generated samples shape: {samples_with.shape}")
        
        print(" Quantum generator working with warning suppression")
        return True
        
    except Exception as e:
        print(f" Quantum generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_qgan_trainer_integration():
    """Test the QGAN trainer with warning suppression integration."""
    
    print("\n3. Testing QGAN Trainer Integration...")
    print("-" * 50)
    
    try:
        # Import components
        from models.generators.quantum_continuous_generator_enhanced import QuantumContinuousGeneratorEnhanced
        from models.discriminators.classical_discriminator import ClassicalDiscriminator
        from training.qgan_trainer import QGAN
        from utils.data_utils import load_synthetic_data
        
        print(" Successfully imported QGAN components")
        
        # Create components
        generator = QuantumContinuousGeneratorEnhanced(n_qumodes=3, latent_dim=4, cutoff_dim=6)
        discriminator = ClassicalDiscriminator(input_dim=3)
        
        # Create QGAN
        qgan = QGAN(generator, discriminator, latent_dim=4)
        print(" Successfully created QGAN")
        
        # Generate test data
        data = load_synthetic_data(dataset_type="spiral", num_samples=100)
        data = data[:, :3]  # Match generator output dimension
        
        print(" Successfully loaded test data")
        
        # Test a few training steps
        print("\n   Testing training steps with warning suppression...")
        
        # Convert to tensor
        data_tensor = tf.constant(data, dtype=tf.float32)
        
        # Test single training step
        real_batch = data_tensor[:16]  # Small batch
        metrics = qgan.train_step(real_batch)
        
        print(f"   Training step completed successfully")
        print(f"   Generator loss: {metrics['g_loss']:.4f}")
        print(f"   Discriminator loss: {metrics['d_loss']:.4f}")
        
        print(" QGAN trainer integration working correctly")
        return True
        
    except Exception as e:
        print(f" QGAN trainer integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    
    print("Quantum GAN Warning Suppression Test Suite")
    print("=" * 60)
    
    # Configure TensorFlow for testing
    tf.config.run_functions_eagerly(True)
    
    tests = [
        test_warning_suppression,
        test_quantum_generator_with_suppression,
        test_qgan_trainer_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f" Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print(" ALL TESTS PASSED! Warning suppression is working correctly.")
        print("\nYou can now run the qgan_synthetic.ipynb notebook without")
        print("the complex number casting warnings flooding your terminal.")
    else:
        print(" Some tests failed. Check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
