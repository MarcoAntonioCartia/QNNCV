"""
Example demonstrating clean quantum training with warning suppression.

This script shows how to use the warning suppression utility to get clean,
readable output during quantum machine learning training.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
import numpy as np

# Import our clean training utilities
from src.utils.warning_suppression import enable_clean_training, clean_training_output
from src.models.generators.quantum_sf_generator import QuantumSFGenerator
from src.models.discriminators.quantum_sf_discriminator import QuantumSFDiscriminator
from src.training.qgan_sf_trainer import QGANSFTrainer

def main():
    """Demonstrate clean quantum training."""
    
    print("="*60)
    print("CLEAN QUANTUM TRAINING EXAMPLE")
    print("="*60)
    
    # Enable clean training environment
    enable_clean_training()
    
    print("\n1. Creating quantum components...")
    
    # Create quantum components
    generator = QuantumSFGenerator(n_modes=2, latent_dim=4, layers=2, cutoff_dim=6)
    discriminator = QuantumSFDiscriminator(n_modes=2, input_dim=2, layers=2, cutoff_dim=6)
    
    print("✓ Generator and discriminator created")
    
    # Create trainer
    trainer = QGANSFTrainer(
        generator=generator,
        discriminator=discriminator,
        latent_dim=4,
        generator_lr=1e-3,
        discriminator_lr=1e-3
    )
    
    print("✓ Trainer initialized")
    
    print("\n2. Generating synthetic training data...")
    
    # Create some synthetic 2D data (e.g., a simple Gaussian mixture)
    n_samples = 200
    
    # Create two Gaussian clusters
    cluster1 = np.random.normal([1.0, 1.0], 0.3, (n_samples//2, 2))
    cluster2 = np.random.normal([-1.0, -1.0], 0.3, (n_samples//2, 2))
    
    real_data = tf.constant(np.vstack([cluster1, cluster2]), dtype=tf.float32)
    
    print(f"✓ Created {n_samples} training samples")
    print(f"  Data shape: {real_data.shape}")
    print(f"  Data range: [{tf.reduce_min(real_data):.3f}, {tf.reduce_max(real_data):.3f}]")
    
    print("\n3. Training with clean output...")
    
    # Train with clean output (warnings suppressed)
    history = trainer.train(
        data=real_data,
        epochs=5,
        batch_size=8,
        verbose=True,
        save_interval=1,
        suppress_warnings=True  # This enables clean output
    )
    
    print("\n4. Evaluating results...")
    
    # Evaluate generation quality
    quality_metrics = trainer.evaluate_generation_quality(real_data, n_samples=100)
    
    print(f"\nGeneration Quality Metrics:")
    print(f"  Mean difference: {quality_metrics['mean_difference']:.4f}")
    print(f"  Std difference: {quality_metrics['std_difference']:.4f}")
    
    print("\n5. Demonstrating warning suppression context...")
    
    print("\nWithout suppression (you'll see warnings):")
    # This will show warnings
    x = tf.constant([1.0 + 2.0j], dtype=tf.complex64)
    y = tf.cast(x, tf.float32)
    
    print("\nWith suppression (clean output):")
    # This will be clean
    with clean_training_output():
        x = tf.constant([1.0 + 2.0j], dtype=tf.complex64)
        y = tf.cast(x, tf.float32)
        print("✓ Complex casting completed without warnings")
    
    print("\n" + "="*60)
    print("CLEAN TRAINING EXAMPLE COMPLETED")
    print("="*60)
    
    print("\nKey benefits of warning suppression:")
    print("• Clean, readable training output")
    print("• Focus on important metrics, not TensorFlow warnings")
    print("• Easy to spot actual issues vs. harmless warnings")
    print("• Professional-looking training logs")
    
    return history, quality_metrics

if __name__ == "__main__":
    try:
        history, metrics = main()
        print(f"\n✓ Training completed successfully!")
        print(f"  Final losses: G={history['g_loss'][-1]:.4f}, D={history['d_loss'][-1]:.4f}")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
