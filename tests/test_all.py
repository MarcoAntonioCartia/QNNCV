#!/usr/bin/env python
"""
Test Suite for CV Quantum GAN
=============================

Verifies all components work correctly:
1. Generator with input encoding
2. Discriminator 
3. Gradient flow
4. Monitoring utilities
5. Full training step

Run from repo root:
    python -m pytest tests/
    # or
    python tests/test_all.py
"""

import sys
import os

# Add src to path (works whether run from root or tests/)
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(repo_root, 'src'))

# =============================================================================
# SCIPY COMPATIBILITY PATCH - MUST BE APPLIED BEFORE IMPORTING STRAWBERRYFIELDS
# =============================================================================
import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson
# =============================================================================

import numpy as np
import tensorflow as tf


def test_1_generator_initialization():
    """Test generator can be created with different encoding types."""
    print("=" * 60)
    print("TEST 1: Generator Initialization")
    print("=" * 60)
    
    from models.generators import QuantumSFGenerator
    
    encoding_types = ['displacement_simple', 'displacement_full', 'displacement_squeezing']
    
    for enc_type in encoding_types:
        print(f"\n  Testing {enc_type}...")
        try:
            gen = QuantumSFGenerator(
                n_modes=1,
                n_layers=1,
                cutoff_dim=6,
                output_dim=1,
                encoding_type=enc_type
            )
            config = gen.get_config()
            print(f"    ✓ Created: latent_dim={config['latent_dim']}, params={config['num_params']}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            return False
    
    print("\n✓ TEST 1 PASSED: All encoding types work")
    return True


def test_2_generator_forward_pass():
    """Test generator forward pass produces correct shapes."""
    print("\n" + "=" * 60)
    print("TEST 2: Generator Forward Pass")
    print("=" * 60)
    
    from models.generators import QuantumSFGenerator
    
    gen = QuantumSFGenerator(
        n_modes=1,
        n_layers=1,
        cutoff_dim=6,
        output_dim=1,
        encoding_type='displacement_full'
    )
    
    batch_size = 4
    z = tf.random.normal([batch_size, gen.latent_dim])
    
    print(f"\n  Input shape: {z.shape}")
    print(f"  Expected output: [{batch_size}, 1]")
    
    try:
        output = gen.generate(z)
        print(f"  Actual output: {output.shape}")
        
        if output.shape == (batch_size, 1):
            print("\n✓ TEST 2 PASSED: Forward pass shapes correct")
            return True
        else:
            print("\n✗ TEST 2 FAILED: Unexpected output shape")
            return False
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_generator_gradient_flow():
    """Test gradients flow through the quantum generator."""
    print("\n" + "=" * 60)
    print("TEST 3: Generator Gradient Flow")
    print("=" * 60)
    
    from models.generators import QuantumSFGenerator
    
    gen = QuantumSFGenerator(
        n_modes=1,
        n_layers=1,
        cutoff_dim=6,
        output_dim=1,
        encoding_type='displacement_full'
    )
    
    z = tf.random.normal([4, gen.latent_dim])
    
    print("\n  Computing gradients...")
    with tf.GradientTape() as tape:
        output = gen.generate(z)
        loss = tf.reduce_mean(tf.square(output - 2.0))
    
    grads = tape.gradient(loss, gen.trainable_variables)
    
    print(f"\n  Loss: {loss.numpy():.4f}")
    print(f"  Trainable variables: {len(gen.trainable_variables)}")
    
    all_grads_ok = True
    for i, (g, v) in enumerate(zip(grads, gen.trainable_variables)):
        if g is not None:
            norm = tf.norm(g).numpy()
            print(f"    {v.name}: grad_norm = {norm:.6f}")
            if norm == 0:
                print(f"      ⚠ Warning: Zero gradient!")
        else:
            print(f"    {v.name}: grad = None ✗")
            all_grads_ok = False
    
    if all_grads_ok:
        print("\n✓ TEST 3 PASSED: All gradients computed")
        return True
    else:
        print("\n✗ TEST 3 FAILED: Some gradients are None")
        return False


def test_4_discriminator():
    """Test discriminator forward pass and gradients."""
    print("\n" + "=" * 60)
    print("TEST 4: Discriminator")
    print("=" * 60)
    
    from models.discriminators import ClassicalDiscriminator
    
    disc = ClassicalDiscriminator(
        input_dim=1,
        hidden_dims=[32, 32],
        output_dim=1
    )
    
    print(f"\n  Config: {disc.get_config()}")
    
    x = tf.random.normal([4, 1])
    
    # Forward pass
    scores = disc.discriminate(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {scores.shape}")
    
    # Gradients
    with tf.GradientTape() as tape:
        scores = disc.discriminate(x)
        loss = tf.reduce_mean(tf.square(scores))
    
    grads = tape.gradient(loss, disc.trainable_variables)
    
    if all(g is not None for g in grads):
        print("\n✓ TEST 4 PASSED: Discriminator works")
        return True
    else:
        print("\n✗ TEST 4 FAILED: Discriminator gradients missing")
        return False


def test_5_adversarial_gradients():
    """Test full GAN adversarial gradient flow."""
    print("\n" + "=" * 60)
    print("TEST 5: Adversarial Gradient Flow")
    print("=" * 60)
    
    from models.generators import QuantumSFGenerator
    from models.discriminators import ClassicalDiscriminator
    
    gen = QuantumSFGenerator(
        n_modes=1,
        n_layers=1,
        cutoff_dim=6,
        output_dim=1,
        encoding_type='displacement_full'
    )
    
    disc = ClassicalDiscriminator(
        input_dim=1,
        hidden_dims=[32, 32],
        output_dim=1
    )
    
    batch_size = 4
    z = tf.random.normal([batch_size, gen.latent_dim])
    real_data = tf.random.normal([batch_size, 1]) * 0.5 + 2.0
    
    # Discriminator step
    print("\n  Testing discriminator update...")
    with tf.GradientTape() as tape:
        fake_data = gen.generate(z)
        fake_scores = disc.discriminate(fake_data)
        real_scores = disc.discriminate(real_data)
        d_loss = tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)
    
    d_grads = tape.gradient(d_loss, disc.trainable_variables)
    d_grads_ok = all(g is not None for g in d_grads)
    print(f"    D_loss: {d_loss.numpy():.4f}")
    print(f"    Gradients OK: {d_grads_ok}")
    
    # Generator step
    print("\n  Testing generator update...")
    with tf.GradientTape() as tape:
        fake_data = gen.generate(z)
        fake_scores = disc.discriminate(fake_data)
        g_loss = -tf.reduce_mean(fake_scores)
    
    g_grads = tape.gradient(g_loss, gen.trainable_variables)
    g_grads_ok = all(g is not None for g in g_grads)
    print(f"    G_loss: {g_loss.numpy():.4f}")
    print(f"    Gradients OK: {g_grads_ok}")
    
    if d_grads_ok and g_grads_ok:
        print("\n✓ TEST 5 PASSED: Adversarial gradients flow correctly")
        return True
    else:
        print("\n✗ TEST 5 FAILED: Adversarial gradient flow broken")
        return False


def test_6_parameter_update():
    """Test that parameters actually change during training."""
    print("\n" + "=" * 60)
    print("TEST 6: Parameter Update")
    print("=" * 60)
    
    from models.generators import QuantumSFGenerator
    from models.discriminators import ClassicalDiscriminator
    
    gen = QuantumSFGenerator(
        n_modes=1,
        n_layers=1,
        cutoff_dim=6,
        output_dim=1,
        encoding_type='displacement_full'
    )
    
    disc = ClassicalDiscriminator(
        input_dim=1,
        hidden_dims=[32, 32],
        output_dim=1
    )
    
    g_opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    d_opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    # Store initial weights
    initial_g_weights = gen.weights.numpy().copy()
    initial_d_weights = [v.numpy().copy() for v in disc.trainable_variables]
    
    batch_size = 4
    z = tf.random.normal([batch_size, gen.latent_dim])
    real_data = tf.random.normal([batch_size, 1]) * 0.5 + 2.0
    
    # One training step
    print("\n  Running one training step...")
    
    # D step
    with tf.GradientTape() as tape:
        fake_data = gen.generate(z)
        fake_scores = disc.discriminate(fake_data)
        real_scores = disc.discriminate(real_data)
        d_loss = tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)
    
    d_grads = tape.gradient(d_loss, disc.trainable_variables)
    d_opt.apply_gradients(zip(d_grads, disc.trainable_variables))
    
    # G step
    with tf.GradientTape() as tape:
        fake_data = gen.generate(z)
        fake_scores = disc.discriminate(fake_data)
        g_loss = -tf.reduce_mean(fake_scores)
    
    g_grads = tape.gradient(g_loss, gen.trainable_variables)
    g_opt.apply_gradients(zip(g_grads, gen.trainable_variables))
    
    # Check if weights changed
    g_changed = not np.allclose(initial_g_weights, gen.weights.numpy())
    d_changed = not all(
        np.allclose(init, curr.numpy()) 
        for init, curr in zip(initial_d_weights, disc.trainable_variables)
    )
    
    print(f"\n  Generator weights changed: {g_changed}")
    print(f"  Discriminator weights changed: {d_changed}")
    
    if g_changed and d_changed:
        print("\n✓ TEST 6 PASSED: Parameters update correctly")
        return True
    else:
        print("\n✗ TEST 6 FAILED: Parameters not updating")
        return False


def test_7_monitoring():
    """Test monitoring utilities."""
    print("\n" + "=" * 60)
    print("TEST 7: Monitoring Utilities")
    print("=" * 60)
    
    from utils.monitoring import TrainingMonitor
    
    monitor = TrainingMonitor(target_mean=2.0, target_std=0.5)
    
    # Simulate some training updates
    print("\n  Simulating training updates...")
    for i in range(5):
        g_loss = tf.constant(1.0 - i * 0.1)
        d_loss = tf.constant(0.5 + i * 0.05)
        g_grads = [tf.random.normal([10])]
        d_grads = [tf.random.normal([10])]
        gen_samples = np.random.normal(2.0, 0.5, size=(32,))
        real_samples = np.random.normal(2.0, 0.5, size=(32,))
        
        metrics = monitor.update(g_loss, d_loss, g_grads, d_grads, gen_samples, real_samples)
    
    summary = monitor.get_summary()
    print(f"  Total epochs tracked: {summary['total_epochs']}")
    print(f"  Final Wasserstein: {summary['final_wasserstein']:.4f}")
    print(f"  Best Wasserstein: {summary['best_wasserstein']:.4f}")
    
    print("\n✓ TEST 7 PASSED: Monitoring works")
    return True


def test_8_visualization_imports():
    """Test visualization functions can be imported."""
    print("\n" + "=" * 60)
    print("TEST 8: Visualization Imports")
    print("=" * 60)
    
    try:
        from utils.visualization import (
            plot_wigner_3d,
            plot_wigner_2d,
            plot_distribution_comparison,
            plot_training_dashboard
        )
        print("\n  ✓ All visualization functions imported")
        print("\n✓ TEST 8 PASSED: Visualization ready")
        return True
    except ImportError as e:
        print(f"\n  ✗ Import error: {e}")
        print("\n✗ TEST 8 FAILED")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CV QUANTUM GAN - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print("\nThis suite verifies all components work correctly.")
    print("Run this before training to catch issues early.\n")
    
    results = []
    
    # Run tests
    results.append(("Generator Initialization", test_1_generator_initialization()))
    results.append(("Generator Forward Pass", test_2_generator_forward_pass()))
    results.append(("Generator Gradient Flow", test_3_generator_gradient_flow()))
    results.append(("Discriminator", test_4_discriminator()))
    results.append(("Adversarial Gradients", test_5_adversarial_gradients()))
    results.append(("Parameter Update", test_6_parameter_update()))
    results.append(("Monitoring", test_7_monitoring()))
    results.append(("Visualization Imports", test_8_visualization_imports()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
        print("\nYou can now run training with:")
        print("  python train_killoran_qgan.py --n-peaks 3")
    else:
        print("SOME TESTS FAILED! ✗")
        print("\nFix the failing tests before training.")
    print("=" * 60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
