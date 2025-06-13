"""
Test script for enhanced quantum discriminator with encoding strategies.

This script tests the enhanced discriminator functionality including:
1. Multiple encoding strategies
2. Feature extraction methods
3. Backward compatibility
4. Gradient computation
5. Integration with existing training
"""

import numpy as np
import tensorflow as tf
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_enhanced_discriminator():
    """Test the enhanced quantum discriminator with different encoding strategies."""
    print("=" * 70)
    print("TESTING ENHANCED QUANTUM DISCRIMINATOR")
    print("=" * 70)
    
    # Test results storage
    test_results = {}
    
    # Test 1: Basic discriminator (backward compatibility)
    print("\n1. Testing Basic Discriminator (Backward Compatibility)")
    print("-" * 50)
    
    try:
        from models.discriminators.quantum_sf_discriminator import QuantumSFDiscriminator
        
        # Create basic discriminator
        basic_discriminator = QuantumSFDiscriminator(
            n_modes=2,
            input_dim=2,
            layers=2,
            cutoff_dim=6
        )
        
        # Test basic functionality
        x_test = tf.random.normal([4, 2])
        probabilities = basic_discriminator.discriminate(x_test)
        
        print(f"✓ Basic discriminator created successfully")
        print(f"  Input shape: {x_test.shape}")
        print(f"  Output shape: {probabilities.shape}")
        print(f"  Probability range: [{tf.reduce_min(probabilities):.3f}, {tf.reduce_max(probabilities):.3f}]")
        print(f"  Trainable variables: {len(basic_discriminator.trainable_variables)}")
        
        test_results['basic_discriminator'] = {
            'success': True,
            'output_shape': probabilities.shape,
            'trainable_vars': len(basic_discriminator.trainable_variables)
        }
        
    except Exception as e:
        print(f"✗ Basic discriminator test failed: {e}")
        test_results['basic_discriminator'] = {'success': False, 'error': str(e)}
    
    # Test 2: Enhanced discriminator with different encoding strategies
    print("\n2. Testing Enhanced Discriminator with Encoding Strategies")
    print("-" * 50)
    
    encoding_strategies = ['coherent_state', 'direct_displacement', 'angle_encoding', 'classical_neural']
    
    for strategy in encoding_strategies:
        print(f"\nTesting {strategy} encoding:")
        try:
            # Create enhanced discriminator
            enhanced_discriminator = QuantumSFDiscriminator(
                n_modes=2,
                input_dim=2,
                layers=2,
                cutoff_dim=6,
                encoding_strategy=strategy,
                feature_extraction='multi_mode',
                enable_batch_processing=True
            )
            
            # Test discrimination
            x_test = tf.random.normal([3, 2])
            probabilities = enhanced_discriminator.discriminate(x_test)
            
            print(f"  ✓ {strategy} discriminator successful")
            print(f"    Output shape: {probabilities.shape}")
            print(f"    Probability range: [{tf.reduce_min(probabilities):.3f}, {tf.reduce_max(probabilities):.3f}]")
            
            # Test gradient computation
            with tf.GradientTape() as tape:
                x = tf.random.normal([2, 2])
                output = enhanced_discriminator.discriminate(x)
                loss = tf.reduce_mean(tf.square(output - 0.5))
            
            gradients = tape.gradient(loss, enhanced_discriminator.trainable_variables)
            non_none_grads = [g for g in gradients if g is not None]
            
            print(f"    Gradients: {len(non_none_grads)}/{len(gradients)} computed")
            print(f"    Loss: {loss:.4f}")
            
            test_results[f'{strategy}_discriminator'] = {
                'success': True,
                'output_shape': probabilities.shape,
                'gradients': len(non_none_grads),
                'loss': float(loss)
            }
            
        except Exception as e:
            print(f"  ✗ {strategy} discriminator failed: {e}")
            test_results[f'{strategy}_discriminator'] = {'success': False, 'error': str(e)}
    
    # Test 3: Feature extraction methods
    print("\n3. Testing Feature Extraction Methods")
    print("-" * 50)
    
    feature_methods = ['multi_mode', 'quantum_observables', 'default']
    
    for method in feature_methods:
        print(f"\nTesting {method} feature extraction:")
        try:
            # Create discriminator with specific feature extraction
            discriminator = QuantumSFDiscriminator(
                n_modes=2,
                input_dim=2,
                layers=2,
                cutoff_dim=6,
                encoding_strategy='coherent_state',
                feature_extraction=method
            )
            
            # Test feature extraction
            x_test = tf.random.normal([2, 2])
            probabilities = discriminator.discriminate(x_test)
            
            print(f"  ✓ {method} feature extraction successful")
            print(f"    Output shape: {probabilities.shape}")
            
            test_results[f'{method}_features'] = {
                'success': True,
                'output_shape': probabilities.shape
            }
            
        except Exception as e:
            print(f"  ✗ {method} feature extraction failed: {e}")
            test_results[f'{method}_features'] = {'success': False, 'error': str(e)}
    
    # Test 4: Batch processing optimization
    print("\n4. Testing Batch Processing Optimization")
    print("-" * 50)
    
    try:
        discriminator = QuantumSFDiscriminator(
            n_modes=2,
            input_dim=2,
            layers=2,
            cutoff_dim=6,
            enable_batch_processing=True
        )
        
        # Test different batch sizes
        batch_sizes = [1, 4, 8]
        batch_times = {}
        
        for batch_size in batch_sizes:
            x_batch = tf.random.normal([batch_size, 2])
            
            import time
            start_time = time.time()
            probabilities = discriminator.discriminate(x_batch)
            batch_time = time.time() - start_time
            
            batch_times[batch_size] = batch_time
            print(f"  Batch size {batch_size}: {batch_time:.4f}s ({batch_time/batch_size:.4f}s per sample)")
        
        print(f"✓ Batch processing test successful")
        
        test_results['batch_processing'] = {
            'success': True,
            'batch_times': batch_times
        }
        
    except Exception as e:
        print(f"✗ Batch processing test failed: {e}")
        test_results['batch_processing'] = {'success': False, 'error': str(e)}
    
    # Test 5: Integration with training (basic compatibility test)
    print("\n5. Testing Training Integration")
    print("-" * 50)
    
    try:
        # Create generator and discriminator for training test
        from models.generators.quantum_sf_generator import QuantumSFGenerator
        
        generator = QuantumSFGenerator(
            n_modes=2,
            latent_dim=4,
            layers=2,
            cutoff_dim=6,
            encoding_strategy='coherent_state'
        )
        
        discriminator = QuantumSFDiscriminator(
            n_modes=2,
            input_dim=2,
            layers=2,
            cutoff_dim=6,
            encoding_strategy='coherent_state'
        )
        
        # Test training step simulation
        real_data = tf.random.normal([4, 2])
        z = tf.random.normal([4, 4])
        
        # Generator forward pass
        fake_data = generator.generate(z)
        
        # Discriminator forward pass
        real_output = discriminator.discriminate(real_data)
        fake_output = discriminator.discriminate(fake_data)
        
        # Simple loss computation
        d_loss = -tf.reduce_mean(tf.math.log(real_output + 1e-8) + tf.math.log(1 - fake_output + 1e-8))
        g_loss = -tf.reduce_mean(tf.math.log(fake_output + 1e-8))
        
        print(f"✓ Training integration test successful")
        print(f"  Real data shape: {real_data.shape}")
        print(f"  Fake data shape: {fake_data.shape}")
        print(f"  Real output shape: {real_output.shape}")
        print(f"  Fake output shape: {fake_output.shape}")
        print(f"  Discriminator loss: {d_loss:.4f}")
        print(f"  Generator loss: {g_loss:.4f}")
        
        test_results['training_integration'] = {
            'success': True,
            'd_loss': float(d_loss),
            'g_loss': float(g_loss)
        }
        
    except Exception as e:
        print(f"✗ Training integration test failed: {e}")
        test_results['training_integration'] = {'success': False, 'error': str(e)}
    
    # Test Summary
    print("\n" + "=" * 70)
    print("ENHANCED DISCRIMINATOR TEST SUMMARY")
    print("=" * 70)
    
    successful_tests = [name for name, result in test_results.items() if result.get('success', False)]
    failed_tests = [name for name, result in test_results.items() if not result.get('success', False)]
    
    print(f"\nTest Results:")
    print(f"  Successful: {len(successful_tests)}/{len(test_results)}")
    print(f"  Failed: {len(failed_tests)}/{len(test_results)}")
    
    if successful_tests:
        print(f"\n✓ Successful tests:")
        for test in successful_tests:
            print(f"    - {test}")
    
    if failed_tests:
        print(f"\n✗ Failed tests:")
        for test in failed_tests:
            error = test_results[test].get('error', 'Unknown error')
            print(f"    - {test}: {error}")
    
    # Overall assessment
    success_rate = len(successful_tests) / len(test_results) * 100
    
    print(f"\nOverall Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 Enhanced discriminator is working well!")
        print("   Ready for advanced quantum GAN experiments.")
    elif success_rate >= 60:
        print("✓ Enhanced discriminator is mostly functional.")
        print("   Minor issues may need attention.")
    else:
        print("⚠ Enhanced discriminator needs debugging.")
        print("   Review failed tests before proceeding.")
    
    return test_results

if __name__ == "__main__":
    # Run the test
    results = test_enhanced_discriminator()
    
    print(f"\n" + "=" * 70)
    print("ENHANCED DISCRIMINATOR TESTING COMPLETE")
    print("=" * 70)
