"""
Debug script to isolate the gradient issue in quantum GAN training.
"""

import sys
import os
sys.path.append('src')

import tensorflow as tf
import numpy as np
from models.generators.quantum_differentiable_generator import QuantumDifferentiableGenerator

def test_generator_gradients():
    """Test generator gradients in isolation."""
    print("=" * 60)
    print("GRADIENT DEBUG TEST")
    print("=" * 60)
    
    # Create generator
    print("Creating quantum generator...")
    generator = QuantumDifferentiableGenerator(
        n_qumodes=2, 
        latent_dim=8, 
        cutoff_dim=4, 
        use_quantum=True
    )
    
    print(f"Generator created with {len(generator.trainable_variables)} trainable variables")
    
    # Test 1: Simple gradient computation
    print("\n1. Testing simple gradient computation...")
    with tf.GradientTape() as tape:
        z = tf.random.normal([4, 8])
        output = generator.generate(z)
        loss = tf.reduce_mean(tf.square(output))
    
    gradients = tape.gradient(loss, generator.trainable_variables)
    non_none_grads = [g for g in gradients if g is not None]
    
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.numpy():.6f}")
    print(f"Gradients: {len(non_none_grads)}/{len(gradients)} variables have gradients")
    
    # Check each gradient
    for i, (var, grad) in enumerate(zip(generator.trainable_variables, gradients)):
        if grad is not None:
            grad_norm = tf.norm(grad).numpy()
            print(f"  Variable {i} ({var.name}): gradient norm = {grad_norm:.8f}")
        else:
            print(f"  Variable {i} ({var.name}): gradient is None ❌")
    
    # Test 2: Multiple gradient computations
    print("\n2. Testing multiple gradient computations...")
    for test_num in range(3):
        with tf.GradientTape() as tape:
            z = tf.random.normal([2, 8])
            output = generator.generate(z)
            loss = tf.reduce_mean(tf.square(output - 1.0))  # Different target
        
        gradients = tape.gradient(loss, generator.trainable_variables)
        non_none_grads = [g for g in gradients if g is not None]
        
        print(f"  Test {test_num + 1}: {len(non_none_grads)}/{len(gradients)} gradients, loss = {loss.numpy():.6f}")
    
    # Test 3: Optimizer application
    print("\n3. Testing optimizer gradient application...")
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    
    # Store initial weights
    initial_weights = [tf.identity(var) for var in generator.trainable_variables]
    
    # Compute gradients and apply
    with tf.GradientTape() as tape:
        z = tf.random.normal([4, 8])
        output = generator.generate(z)
        loss = tf.reduce_mean(tf.square(output - 0.5))
    
    gradients = tape.gradient(loss, generator.trainable_variables)
    
    # Check if any gradients are None before applying
    none_count = sum(1 for g in gradients if g is None)
    if none_count > 0:
        print(f"❌ ERROR: {none_count} gradients are None - optimizer will fail!")
        return False
    
    # Apply gradients
    try:
        optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
        print("✅ Optimizer successfully applied gradients")
        
        # Check if weights actually changed
        weights_changed = False
        for initial, current in zip(initial_weights, generator.trainable_variables):
            if not tf.reduce_all(tf.equal(initial, current)):
                weights_changed = True
                break
        
        if weights_changed:
            print("✅ Weights successfully updated")
            return True
        else:
            print("❌ Weights did not change")
            return False
            
    except Exception as e:
        print(f"❌ Optimizer failed: {e}")
        return False

def test_training_step_simulation():
    """Simulate the exact training step from QGAN."""
    print("\n" + "=" * 60)
    print("TRAINING STEP SIMULATION")
    print("=" * 60)
    
    # Create generator
    generator = QuantumDifferentiableGenerator(
        n_qumodes=2, 
        latent_dim=8, 
        cutoff_dim=4, 
        use_quantum=True
    )
    
    # Create fake discriminator for testing
    class FakeDiscriminator:
        def discriminate(self, x):
            # Simple linear discriminator for testing
            return tf.reduce_mean(x, axis=1, keepdims=True)
    
    discriminator = FakeDiscriminator()
    
    # Simulate training step
    batch_size = 4
    latent_dim = 8
    
    print("Simulating generator training step...")
    
    # Create fake real samples
    real_samples = tf.random.normal([batch_size, 2])
    
    # Generator training step
    z = tf.random.normal([batch_size, latent_dim])
    
    with tf.GradientTape() as g_tape:
        fake_samples = generator.generate(z)
        fake_output = discriminator.discriminate(fake_samples)
        
        # Simple generator loss (want discriminator to output high values)
        g_loss = -tf.reduce_mean(fake_output)
    
    # Compute gradients
    g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
    
    # Check gradients
    non_none_grads = [g for g in g_gradients if g is not None]
    
    print(f"Fake samples shape: {fake_samples.shape}")
    print(f"Generator loss: {g_loss.numpy():.6f}")
    print(f"Generator gradients: {len(non_none_grads)}/{len(g_gradients)} variables have gradients")
    
    if len(non_none_grads) == len(g_gradients):
        print("✅ All generator variables have gradients in training simulation")
        return True
    else:
        print("❌ Some generator variables missing gradients in training simulation")
        return False

if __name__ == "__main__":
    # Run tests
    test1_passed = test_generator_gradients()
    test2_passed = test_training_step_simulation()
    
    print("\n" + "=" * 60)
    print("GRADIENT DEBUG SUMMARY")
    print("=" * 60)
    print(f"Generator gradient test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Training step simulation: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n✅ Gradients are working correctly!")
        print("The issue might be elsewhere in the training pipeline.")
    else:
        print("\n❌ Gradient issue confirmed!")
        print("The problem is in the generator's gradient computation.")
