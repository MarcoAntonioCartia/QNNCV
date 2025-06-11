"""
Test script to verify the quantum GAN AutoGraph fixes.

This script tests the modified quantum generator and training framework
to ensure the AutoGraph compatibility issues have been resolved.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

# Import TensorFlow with error handling
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    TF_AVAILABLE = True
except ImportError as e:
    print(f"TensorFlow not available: {e}")
    TF_AVAILABLE = False
    sys.exit(1)

# Configure TensorFlow for quantum operations
from utils.tensorflow_compat import configure_tensorflow_for_quantum, test_tensorflow_quantum_compatibility

print("Configuring TensorFlow for quantum operations...")
config_status = configure_tensorflow_for_quantum()
print(f"Configuration status: {config_status}")

# Test TensorFlow compatibility
print("\nTesting TensorFlow quantum compatibility...")
compat_results = test_tensorflow_quantum_compatibility()
for key, value in compat_results.items():
    print(f"  {key}: {value}")

# Test quantum generator
print("\n" + "="*60)
print("TESTING QUANTUM GENERATOR")
print("="*60)

try:
    from models.generators.quantum_continuous_generator_enhanced import QuantumContinuousGeneratorEnhanced
    
    print("Creating quantum generator...")
    generator = QuantumContinuousGeneratorEnhanced(
        n_qumodes=2, 
        latent_dim=4,
        cutoff_dim=5
    )
    
    print("Testing generator with small batch...")
    z_test = tf.random.normal([2, 4])
    
    # Test generation
    print("Generating samples...")
    samples = generator.generate(z_test)
    print(f"Generated samples shape: {samples.shape}")
    print(f"Sample values: {samples.numpy()}")
    
    print("✓ Quantum generator test PASSED")
    
except Exception as e:
    print(f"✗ Quantum generator test FAILED: {e}")
    print("Trying fallback generator...")
    
    try:
        from models.generators.quantum_continuous_generator_enhanced import QuantumContinuousGeneratorSimple
        
        generator = QuantumContinuousGeneratorSimple(n_qumodes=2, latent_dim=4)
        z_test = tf.random.normal([2, 4])
        samples = generator.generate(z_test)
        print(f"Fallback generator samples shape: {samples.shape}")
        print("✓ Fallback generator test PASSED")
        
    except Exception as e2:
        print(f"✗ Fallback generator test FAILED: {e2}")

# Test discriminator
print("\n" + "="*60)
print("TESTING QUANTUM DISCRIMINATOR")
print("="*60)

try:
    from models.discriminators.quantum_continuous_discriminator import QuantumContinuousDiscriminator
    
    print("Creating quantum discriminator...")
    discriminator = QuantumContinuousDiscriminator(
        n_qumodes=2,
        input_dim=2,
        cutoff_dim=5
    )
    
    print("Testing discriminator...")
    test_data = tf.random.normal([2, 2])
    probs = discriminator.discriminate(test_data)
    print(f"Discriminator output shape: {probs.shape}")
    print(f"Probability values: {probs.numpy()}")
    
    print("✓ Quantum discriminator test PASSED")
    
except Exception as e:
    print(f"✗ Quantum discriminator test FAILED: {e}")

# Test QGAN training framework
print("\n" + "="*60)
print("TESTING QGAN TRAINING FRAMEWORK")
print("="*60)

try:
    from training.qgan_trainer import QGAN
    
    # Use simple generators for testing if quantum ones failed
    if 'generator' not in locals():
        from models.generators.quantum_continuous_generator_enhanced import QuantumContinuousGeneratorSimple
        generator = QuantumContinuousGeneratorSimple(n_qumodes=2, latent_dim=4)
    
    if 'discriminator' not in locals():
        # Create a simple classical discriminator for testing
        class SimpleDiscriminator:
            def __init__(self):
                self.network = tf.keras.Sequential([
                    tf.keras.layers.Dense(16, activation='relu'),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
                # Build the network
                dummy_input = tf.zeros((1, 2))
                _ = self.network(dummy_input)
            
            @property
            def trainable_variables(self):
                return self.network.trainable_variables
            
            def discriminate(self, x):
                return self.network(x)
        
        discriminator = SimpleDiscriminator()
    
    print("Creating QGAN...")
    qgan = QGAN(
        generator=generator,
        discriminator=discriminator,
        latent_dim=4,
        generator_lr=0.001,
        discriminator_lr=0.001
    )
    
    print("Testing training step...")
    # Create small test dataset
    test_data = tf.random.normal([8, 2])
    
    # Test single training step
    metrics = qgan.train_step(test_data, use_wasserstein=False)
    print(f"Training step metrics: {list(metrics.keys())}")
    
    for key, value in metrics.items():
        if hasattr(value, 'numpy'):
            print(f"  {key}: {value.numpy():.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("✓ QGAN training framework test PASSED")
    
except Exception as e:
    print(f"✗ QGAN training framework test FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test short training run
print("\n" + "="*60)
print("TESTING SHORT TRAINING RUN")
print("="*60)

try:
    if 'qgan' in locals():
        print("Running 3 epochs of training...")
        
        # Generate synthetic spiral data
        t = np.linspace(0, 2*np.pi, 100)
        x = t * np.cos(t) * 0.1
        y = t * np.sin(t) * 0.1
        training_data = tf.constant(np.column_stack([x, y]), dtype=tf.float32)
        
        # Run short training
        history = qgan.train(
            data=training_data,
            epochs=3,
            batch_size=16,
            use_wasserstein=False,
            verbose=True,
            save_interval=1
        )
        
        print(f"Training history keys: {list(history.keys())}")
        print("✓ Short training run PASSED")
        
    else:
        print("Skipping training run - QGAN not available")
        
except Exception as e:
    print(f"✗ Short training run FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print("The quantum GAN AutoGraph fixes have been tested.")
print("If all tests passed, the notebook should now work without AutoGraph errors.")
print("If some tests failed, check the error messages above for debugging.")
