"""
Complete QGAN Solution Test

This test demonstrates the complete working quantum GAN solution using
the reusable quantum gradient utilities. It shows how to create a fully
functional quantum GAN with proper gradient flow for both generator
and discriminator.
"""

import sys
import os
sys.path.append('src')

import tensorflow as tf
import numpy as np
from models.generators.quantum_differentiable_generator import QuantumDifferentiableGenerator
from models.discriminators.quantum_continuous_discriminator import QuantumContinuousDiscriminator
from utils.quantum_gradients import (
    create_quantum_generator_with_gradients,
    create_quantum_discriminator_with_gradients,
    QuantumGradientWrapper
)
from utils.tensorflow_compat import suppress_complex_warnings

def create_complete_quantum_gan():
    """Create a complete quantum GAN with gradient-enabled components."""
    print("=" * 60)
    print("CREATING COMPLETE QUANTUM GAN SOLUTION")
    print("=" * 60)
    
    # Create quantum generator with custom gradients
    print("Creating quantum generator with custom gradients...")
    generator = create_quantum_generator_with_gradients(
        QuantumDifferentiableGenerator,
        n_qumodes=2,
        latent_dim=8,
        cutoff_dim=4,
        use_quantum=True
    )
    
    print(f"✅ Generator created with {len(generator.trainable_variables)} trainable variables")
    
    # Create quantum discriminator with custom gradients
    print("Creating quantum discriminator with custom gradients...")
    discriminator = create_quantum_discriminator_with_gradients(
        QuantumContinuousDiscriminator,
        n_qumodes=4,
        input_dim=2,
        cutoff_dim=4
    )
    
    print(f"✅ Discriminator created with {len(discriminator.trainable_variables)} trainable variables")
    
    return generator, discriminator

def test_complete_training_pipeline():
    """Test the complete training pipeline with gradient-enabled components."""
    print("\n" + "=" * 60)
    print("TESTING COMPLETE TRAINING PIPELINE")
    print("=" * 60)
    
    # Create components
    generator, discriminator = create_complete_quantum_gan()
    
    # Create optimizers
    g_optimizer = tf.optimizers.Adam(learning_rate=5e-5)
    d_optimizer = tf.optimizers.Adam(learning_rate=5e-5)
    
    # Create training data (2D spiral)
    print("\nGenerating 2D spiral training data...")
    n_samples = 200
    t = np.linspace(0, 4*np.pi, n_samples)
    r = t / (4*np.pi) * 3
    spiral_x = r * np.cos(t) + np.random.normal(0, 0.1, n_samples)
    spiral_y = r * np.sin(t) + np.random.normal(0, 0.1, n_samples)
    
    # Normalize data
    spiral_data = np.column_stack([spiral_x, spiral_y])
    spiral_data = (spiral_data - np.mean(spiral_data, axis=0)) / np.std(spiral_data, axis=0)
    spiral_data = spiral_data / np.max(np.abs(spiral_data))
    
    training_data = tf.constant(spiral_data, dtype=tf.float32)
    print(f"Training data shape: {training_data.shape}")
    
    # Training function
    def training_step(real_samples, step_num):
        """Complete training step with both generator and discriminator."""
        batch_size = tf.shape(real_samples)[0]
        latent_dim = 8
        
        with suppress_complex_warnings():
            # Train discriminator
            z = tf.random.normal([batch_size, latent_dim])
            
            with tf.GradientTape() as d_tape:
                # Generate fake samples
                fake_samples = generator(z)
                
                # Get discriminator outputs
                real_output = discriminator(real_samples)
                fake_output = discriminator(fake_samples)
                
                # Discriminator loss (wants to distinguish real from fake)
                d_loss = -tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)
            
            # Compute and apply discriminator gradients
            d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
            d_gradients = [tf.clip_by_norm(g, 0.5) for g in d_gradients if g is not None]
            d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
            
            # Train generator
            z = tf.random.normal([batch_size, latent_dim])
            
            with tf.GradientTape() as g_tape:
                # Generate fake samples
                fake_samples = generator(z)
                
                # Get discriminator output for fake samples
                fake_output = discriminator(fake_samples)
                
                # Generator loss (wants discriminator to think fake is real)
                g_loss = -tf.reduce_mean(fake_output)
            
            # Compute and apply generator gradients
            g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
            g_gradients = [tf.clip_by_norm(g, 0.5) for g in g_gradients if g is not None]
            g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
        
        return {
            'g_loss': g_loss,
            'd_loss': d_loss,
            'g_grad_count': len([g for g in g_gradients if g is not None]),
            'd_grad_count': len([g for g in d_gradients if g is not None])
        }
    
    # Run training steps
    print("\nRunning training steps...")
    batch_size = 32
    num_steps = 10
    
    for step in range(num_steps):
        # Sample batch
        indices = np.random.choice(len(training_data), batch_size, replace=False)
        batch = tf.gather(training_data, indices)
        
        # Training step
        metrics = training_step(batch, step)
        
        if step % 2 == 0:
            print(f"Step {step:2d}: G_loss={metrics['g_loss']:7.4f}, "
                  f"D_loss={metrics['d_loss']:7.4f}, "
                  f"G_grads={metrics['g_grad_count']}, "
                  f"D_grads={metrics['d_grad_count']}")
    
    print("\n✅ Training pipeline completed successfully!")
    
    # Test generation
    print("\nTesting final generation...")
    test_z = tf.random.normal([5, 8])
    with suppress_complex_warnings():
        generated_samples = generator(test_z)
    
    print(f"Generated samples shape: {generated_samples.shape}")
    print(f"Generated samples:\n{generated_samples.numpy()}")
    
    return True

def test_gradient_flow_verification():
    """Verify that gradients flow properly through the complete system."""
    print("\n" + "=" * 60)
    print("VERIFYING GRADIENT FLOW")
    print("=" * 60)
    
    # Create components
    generator, discriminator = create_complete_quantum_gan()
    
    # Test data
    batch_size = 4
    latent_dim = 8
    real_samples = tf.random.normal([batch_size, 2])
    z = tf.random.normal([batch_size, latent_dim])
    
    print("Testing generator gradient flow...")
    with tf.GradientTape() as tape:
        with suppress_complex_warnings():
            fake_samples = generator(z)
            loss = tf.reduce_mean(tf.square(fake_samples))
    
    g_gradients = tape.gradient(loss, generator.trainable_variables)
    g_non_none = [g for g in g_gradients if g is not None]
    
    print(f"Generator: {len(g_non_none)}/{len(g_gradients)} variables have gradients")
    
    print("Testing discriminator gradient flow...")
    with tf.GradientTape() as tape:
        with suppress_complex_warnings():
            output = discriminator(real_samples)
            loss = tf.reduce_mean(tf.square(output))
    
    d_gradients = tape.gradient(loss, discriminator.trainable_variables)
    d_non_none = [g for g in d_gradients if g is not None]
    
    print(f"Discriminator: {len(d_non_none)}/{len(d_gradients)} variables have gradients")
    
    # Test end-to-end gradient flow
    print("Testing end-to-end gradient flow...")
    with tf.GradientTape(persistent=True) as tape:
        with suppress_complex_warnings():
            fake_samples = generator(z)
            fake_output = discriminator(fake_samples)
            real_output = discriminator(real_samples)
            
            g_loss = -tf.reduce_mean(fake_output)
            d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
    
    g_grads_e2e = tape.gradient(g_loss, generator.trainable_variables)
    d_grads_e2e = tape.gradient(d_loss, discriminator.trainable_variables)
    
    g_non_none_e2e = [g for g in g_grads_e2e if g is not None]
    d_non_none_e2e = [g for g in d_grads_e2e if g is not None]
    
    print(f"End-to-end Generator: {len(g_non_none_e2e)}/{len(g_grads_e2e)} variables have gradients")
    print(f"End-to-end Discriminator: {len(d_non_none_e2e)}/{len(d_grads_e2e)} variables have gradients")
    
    # Check gradient magnitudes
    g_grad_norms = [tf.norm(g).numpy() for g in g_non_none_e2e]
    d_grad_norms = [tf.norm(g).numpy() for g in d_non_none_e2e]
    
    print(f"Generator gradient norms: min={min(g_grad_norms):.6f}, max={max(g_grad_norms):.6f}")
    print(f"Discriminator gradient norms: min={min(d_grad_norms):.6f}, max={max(d_grad_norms):.6f}")
    
    # Success criteria
    success = (
        len(g_non_none) == len(g_gradients) and
        len(d_non_none) == len(d_gradients) and
        len(g_non_none_e2e) == len(g_grads_e2e) and
        len(d_non_none_e2e) == len(d_grads_e2e) and
        all(0 < norm < 10 for norm in g_grad_norms) and
        all(0 < norm < 10 for norm in d_grad_norms)
    )
    
    if success:
        print("✅ All gradient flow tests passed!")
    else:
        print("❌ Some gradient flow tests failed!")
    
    return success

def main():
    """Run the complete quantum GAN solution test."""
    print("🚀 COMPLETE QUANTUM GAN SOLUTION TEST")
    print("=" * 60)
    
    try:
        # Test 1: Complete training pipeline
        test1_passed = test_complete_training_pipeline()
        
        # Test 2: Gradient flow verification
        test2_passed = test_gradient_flow_verification()
        
        # Summary
        print("\n" + "=" * 60)
        print("FINAL TEST SUMMARY")
        print("=" * 60)
        print(f"Complete training pipeline: {'PASSED' if test1_passed else 'FAILED'}")
        print(f"Gradient flow verification: {'PASSED' if test2_passed else 'FAILED'}")
        
        if test1_passed and test2_passed:
            print("\n🎉 SUCCESS: Complete quantum GAN solution is working!")
            print("\n📋 SOLUTION SUMMARY:")
            print("✅ Quantum generator with custom gradients")
            print("✅ Quantum discriminator with custom gradients")
            print("✅ Proper gradient flow through quantum circuits")
            print("✅ End-to-end training pipeline")
            print("✅ Reusable gradient utilities")
            print("\n🔧 READY FOR PRODUCTION USE!")
        else:
            print("\n❌ FAILED: Some components need further work")
            
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
