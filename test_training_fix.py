"""
Test the fixed training pipeline to verify gradients work.
"""

import sys
import os
sys.path.append('src')

import tensorflow as tf
import numpy as np
from models.generators.quantum_differentiable_generator import QuantumDifferentiableGenerator
from models.discriminators.quantum_continuous_discriminator import QuantumContinuousDiscriminator
from training.qgan_trainer import QGAN

def test_fixed_training():
    """Test the fixed QGAN training pipeline."""
    print("=" * 60)
    print("TESTING FIXED QGAN TRAINING PIPELINE")
    print("=" * 60)
    
    # Create components
    print("Creating quantum GAN components...")
    
    generator = QuantumDifferentiableGenerator(
        n_qumodes=2, 
        latent_dim=8, 
        cutoff_dim=4, 
        use_quantum=True
    )
    
    discriminator = QuantumContinuousDiscriminator(
        n_qumodes=4,
        input_dim=2,
        cutoff_dim=4
    )
    
    qgan = QGAN(
        generator=generator,
        discriminator=discriminator,
        latent_dim=8,
        generator_lr=5e-5,
        discriminator_lr=5e-5,
        gradient_clip_norm=0.5
    )
    
    print("✅ QGAN created successfully")
    
    # Create test data
    print("\nCreating test data...")
    batch_size = 4
    real_samples = tf.random.normal([batch_size, 2])
    
    # Test single training step
    print("\nTesting single training step...")
    try:
        metrics = qgan.train_step(real_samples, use_wasserstein=False)
        
        print(f"✅ Training step successful!")
        print(f"  Generator loss: {metrics['g_loss']:.6f}")
        print(f"  Discriminator loss: {metrics['d_loss']:.6f}")
        print(f"  Generator grad norm: {metrics['g_grad_norm']:.6f}")
        print(f"  Discriminator grad norm: {metrics['d_grad_norm']:.6f}")
        print(f"  Stability metric: {metrics['stability_metric']:.6f}")
        
        # Check if gradients are non-zero
        if metrics['g_grad_norm'] > 0 and metrics['d_grad_norm'] > 0:
            print("✅ Both generator and discriminator have non-zero gradients!")
            return True
        else:
            print("❌ Some gradients are zero")
            return False
            
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return False

def test_mini_training():
    """Test a mini training loop."""
    print("\n" + "=" * 60)
    print("TESTING MINI TRAINING LOOP")
    print("=" * 60)
    
    # Create components
    generator = QuantumDifferentiableGenerator(
        n_qumodes=2, 
        latent_dim=8, 
        cutoff_dim=4, 
        use_quantum=True
    )
    
    discriminator = QuantumContinuousDiscriminator(
        n_qumodes=4,
        input_dim=2,
        cutoff_dim=4
    )
    
    qgan = QGAN(
        generator=generator,
        discriminator=discriminator,
        latent_dim=8,
        generator_lr=5e-5,
        discriminator_lr=5e-5,
        gradient_clip_norm=0.5
    )
    
    # Create spiral test data
    print("Creating spiral test data...")
    n_samples = 100
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
    
    # Run mini training
    print("\nRunning mini training (2 epochs)...")
    try:
        history = qgan.train(
            data=training_data,
            epochs=2,
            batch_size=16,
            use_wasserstein=False,
            verbose=True,
            save_interval=1
        )
        
        print("✅ Mini training completed successfully!")
        print(f"Final generator loss: {history['g_loss'][-1]:.6f}")
        print(f"Final discriminator loss: {history['d_loss'][-1]:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mini training failed: {e}")
        return False

if __name__ == "__main__":
    # Run tests
    test1_passed = test_fixed_training()
    test2_passed = test_mini_training()
    
    print("\n" + "=" * 60)
    print("TRAINING FIX TEST SUMMARY")
    print("=" * 60)
    print(f"Single training step: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Mini training loop: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 SUCCESS: Training pipeline is now working!")
        print("The gradient issue has been resolved.")
    else:
        print("\n❌ FAILED: Training pipeline still has issues.")
