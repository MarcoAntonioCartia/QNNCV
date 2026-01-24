"""
Test Quantum SF Generator and Discriminator - Repo Classes
===========================================================

This test verifies that the actual QuantumSFGenerator and QuantumSFDiscriminator
classes from the repo work correctly with the fixed gradient flow patterns.

Run from repo root:
    python tests/test_repo_classes.py
"""

import sys
import os

# Add src to path (works whether run from root or tests/)
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(repo_root, 'src'))

# Apply SciPy compatibility patch FIRST
from utils.scipy_compat import _patch_scipy_simps
_patch_scipy_simps()

import numpy as np
import tensorflow as tf
from models.generators.quantum_sf_generator import QuantumSFGenerator
from models.discriminators.quantum_sf_discriminator import QuantumSFDiscriminator


def test_1_initialization():
    """Test that both models can be initialized with 1 mode, 1 layer."""
    print("="*70)
    print("TEST 1: Model Initialization (1 mode, 1 layer)")
    print("="*70)
    
    try:
        print("\nInitializing Generator...")
        generator = QuantumSFGenerator(
            latent_dim=2,
            output_dim=1, 
            n_modes=1,
            n_layers=1,
            cutoff_dim=6
        )
        
        print("\nInitializing Discriminator...")
        discriminator = QuantumSFDiscriminator(
            input_dim=1,
            n_modes=1,
            n_layers=1,
            cutoff_dim=6
        )
        
        print("\n✓ Both models initialized successfully")
        print(f"  Generator params: {generator.num_params}")
        print(f"  Discriminator params: {discriminator.num_params}")
        
        return True, generator, discriminator
        
    except Exception as e:
        print(f"\n✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_2_forward_pass(generator, discriminator):
    """Test forward passes through both models."""
    print("\n" + "="*70)
    print("TEST 2: Forward Pass")
    print("="*70)
    
    try:
        batch_size = 4
        
        # Test generator
        print("\nTesting Generator forward pass...")
        z = tf.random.normal([batch_size, 2])
        gen_output = generator.generate(z)
        print(f"  Input shape: {z.shape}")
        print(f"  Output shape: {gen_output.shape}")
        print(f"  Output values (first 2): {gen_output.numpy()[:2, 0]}")
        
        # Test discriminator  
        print("\nTesting Discriminator forward pass...")
        x = tf.random.normal([batch_size, 1])
        disc_output = discriminator.discriminate(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {disc_output.shape}")
        print(f"  Output values: {disc_output.numpy().flatten()}")
        
        print("\n✓ Forward passes successful")
        return True
        
    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_gradient_flow(generator, discriminator):
    """Test that gradients flow correctly through both models."""
    print("\n" + "="*70)
    print("TEST 3: Gradient Flow Verification")
    print("="*70)
    
    try:
        batch_size = 4
        
        # Test generator gradients
        print("\nTesting Generator gradients...")
        with tf.GradientTape() as tape:
            z = tf.random.normal([batch_size, 2])
            gen_output = generator.generate(z)
            g_loss = tf.reduce_mean(tf.square(gen_output - 1.0))
        
        g_grads = tape.gradient(g_loss, generator.trainable_variables)
        
        if g_grads is None or all(g is None for g in g_grads):
            print("  ✗ Generator gradients are None!")
            return False
        
        # Check quantum weights gradient
        quantum_grad = g_grads[0]  # weights
        print(f"  ✓ Generator gradients computed")
        print(f"    Quantum weights gradient shape: {quantum_grad.shape}")
        print(f"    Quantum weights gradient norm: {tf.norm(quantum_grad).numpy():.6f}")
        print(f"    Non-zero gradients: {tf.reduce_sum(tf.cast(quantum_grad != 0, tf.int32)).numpy()}/{np.prod(quantum_grad.shape)}")
        
        # Test discriminator gradients
        print("\nTesting Discriminator gradients...")
        with tf.GradientTape() as tape:
            x = tf.random.normal([batch_size, 1])
            disc_output = discriminator.discriminate(x)
            d_loss = tf.reduce_mean(tf.square(disc_output - 0.5))
        
        d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
        
        if d_grads is None or all(g is None for g in d_grads):
            print("  ✗ Discriminator gradients are None!")
            return False
        
        # Check quantum weights gradient
        quantum_grad = d_grads[0]  # weights
        print(f"  ✓ Discriminator gradients computed")
        print(f"    Quantum weights gradient shape: {quantum_grad.shape}")
        print(f"    Quantum weights gradient norm: {tf.norm(quantum_grad).numpy():.6f}")
        print(f"    Non-zero gradients: {tf.reduce_sum(tf.cast(quantum_grad != 0, tf.int32)).numpy()}/{np.prod(quantum_grad.shape)}")
        
        print("\n✓ Gradient flow verification successful")
        return True
        
    except Exception as e:
        print(f"\n✗ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_adversarial_gradients(generator, discriminator):
    """Test gradients in adversarial setting (GAN training step)."""
    print("\n" + "="*70)
    print("TEST 4: Adversarial Gradient Flow")
    print("="*70)
    
    try:
        batch_size = 4
        
        # Generate fake data
        z = tf.random.normal([batch_size, 2])
        
        # Test discriminator update with both real and fake
        print("\nTesting Discriminator adversarial gradients...")
        with tf.GradientTape() as tape:
            fake_data = generator.generate(z)
            real_data = tf.random.normal([batch_size, 1]) * 0.5 + 2.0  # N(2, 0.5)
            
            fake_scores = discriminator.discriminate(fake_data)
            real_scores = discriminator.discriminate(real_data)
            
            d_loss = tf.reduce_mean(tf.square(real_scores - 1.0)) + \
                     tf.reduce_mean(tf.square(fake_scores - 0.0))
        
        d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
        
        if d_grads is None or all(g is None for g in d_grads):
            print("  ✗ Discriminator adversarial gradients are None!")
            return False
        
        print(f"  ✓ Discriminator adversarial gradients computed")
        print(f"    Loss: {d_loss.numpy():.6f}")
        print(f"    Quantum gradient norm: {tf.norm(d_grads[0]).numpy():.6f}")
        
        # Test generator adversarial update
        print("\nTesting Generator adversarial gradients...")
        with tf.GradientTape() as tape:
            fake_data = generator.generate(z)
            fake_scores = discriminator.discriminate(fake_data)
            
            # Generator wants high scores (fool discriminator)
            g_loss = tf.reduce_mean(tf.square(fake_scores - 1.0))
        
        g_grads = tape.gradient(g_loss, generator.trainable_variables)
        
        if g_grads is None or all(g is None for g in g_grads):
            print("  ✗ Generator adversarial gradients are None!")
            return False
        
        print(f"  ✓ Generator adversarial gradients computed")
        print(f"    Loss: {g_loss.numpy():.6f}")
        print(f"    Quantum gradient norm: {tf.norm(g_grads[0]).numpy():.6f}")
        
        print("\n✓ Adversarial gradient flow successful")
        return True
        
    except Exception as e:
        print(f"\n✗ Adversarial gradient test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_parameter_update(generator, discriminator):
    """Test that parameters actually update during training."""
    print("\n" + "="*70)
    print("TEST 5: Parameter Update Verification")
    print("="*70)
    
    try:
        # Create optimizers
        g_opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        d_opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        
        # Store initial parameters
        initial_gen_weights = generator.weights.numpy().copy()
        initial_disc_weights = discriminator.weights.numpy().copy()
        
        print(f"\nInitial Generator quantum weights (first 3): {initial_gen_weights.flatten()[:3]}")
        print(f"Initial Discriminator quantum weights (first 3): {initial_disc_weights.flatten()[:3]}")
        
        # Run one training step
        batch_size = 4
        z = tf.random.normal([batch_size, 2])
        real_data = tf.random.normal([batch_size, 1]) * 0.5 + 2.0
        
        # Update discriminator
        print("\nUpdating Discriminator...")
        with tf.GradientTape() as tape:
            fake_data = generator.generate(z)
            fake_scores = discriminator.discriminate(fake_data)
            real_scores = discriminator.discriminate(real_data)
            d_loss = tf.reduce_mean(tf.square(real_scores - 1.0)) + \
                     tf.reduce_mean(tf.square(fake_scores - 0.0))
        
        d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))
        
        # Update generator
        print("Updating Generator...")
        with tf.GradientTape() as tape:
            fake_data = generator.generate(z)
            fake_scores = discriminator.discriminate(fake_data)
            g_loss = tf.reduce_mean(tf.square(fake_scores - 1.0))
        
        g_grads = tape.gradient(g_loss, generator.trainable_variables)
        g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
        
        # Check if parameters changed
        final_gen_weights = generator.weights.numpy()
        final_disc_weights = discriminator.weights.numpy()
        
        print(f"\nFinal Generator quantum weights (first 3): {final_gen_weights.flatten()[:3]}")
        print(f"Final Discriminator quantum weights (first 3): {final_disc_weights.flatten()[:3]}")
        
        gen_changed = not np.allclose(initial_gen_weights, final_gen_weights)
        disc_changed = not np.allclose(initial_disc_weights, final_disc_weights)
        
        print(f"\nGenerator weights changed: {gen_changed}")
        print(f"Discriminator weights changed: {disc_changed}")
        
        if gen_changed and disc_changed:
            print("\n✓ Parameters updated successfully")
            return True
        else:
            print("\n✗ Parameters did not update!")
            return False
        
    except Exception as e:
        print(f"\n✗ Parameter update test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("QUANTUM SF GAN - REPO CLASSES VERIFICATION")
    print("="*70)
    print("\nThis test verifies that the actual repo classes work correctly")
    print("with the fixed gradient flow patterns.\n")
    
    results = []
    
    # Test 1: Initialization
    success, generator, discriminator = test_1_initialization()
    results.append(("Initialization", success))
    
    if not success:
        print("\n✗ Cannot proceed without successful initialization")
        return
    
    # Test 2: Forward pass
    success = test_2_forward_pass(generator, discriminator)
    results.append(("Forward Pass", success))
    
    if not success:
        print("\n✗ Cannot proceed without successful forward pass")
        return
    
    # Test 3: Gradient flow
    success = test_3_gradient_flow(generator, discriminator)
    results.append(("Gradient Flow", success))
    
    # Test 4: Adversarial gradients
    success = test_4_adversarial_gradients(generator, discriminator)
    results.append(("Adversarial Gradients", success))
    
    # Test 5: Parameter updates
    success = test_5_parameter_update(generator, discriminator)
    results.append(("Parameter Updates", success))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
        print("\nThe repo classes are working correctly with fixed gradient flow.")
    else:
        print("SOME TESTS FAILED! ✗")
        print("\nFix the failing tests before proceeding.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
