"""
Simplified Clusterized Quantum GAN Training

This script focuses on the training loop with minimal complexity and extensive debugging.
"""

import numpy as np
import tensorflow as tf
import os
import sys
import time
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.generators.clusterized_quantum_generator import ClusterizedQuantumGenerator
from src.models.discriminators.pure_sf_discriminator import PureSFDiscriminator
from src.training.data_generators import BimodalDataGenerator


def debug_print(message: str):
    """Print debug message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()


def simple_training_loop():
    """Simplified training loop with extensive debugging."""
    debug_print("Starting simplified training...")
    
    # Suppress TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    # Create target data
    debug_print("Creating target data...")
    np.random.seed(42)
    cluster1 = np.random.normal([-1.5, -1.5], 0.3, (100, 2))
    cluster2 = np.random.normal([1.5, 1.5], 0.3, (100, 2))
    target_data = np.vstack([cluster1, cluster2])
    debug_print(f"Target data created: shape {target_data.shape}")
    
    # Create models with minimal complexity
    debug_print("Creating generator...")
    generator = ClusterizedQuantumGenerator(
        latent_dim=4,
        output_dim=2,
        n_modes=2,
        layers=1,
        cutoff_dim=4,
        clustering_method='kmeans',
        coordinate_names=['X', 'Y']
    )
    debug_print("Generator created")
    
    debug_print("Creating discriminator...")
    discriminator = PureSFDiscriminator(
        input_dim=2,
        n_modes=2,
        layers=1,
        cutoff_dim=4
    )
    debug_print("Discriminator created")
    
    # Analyze target data
    debug_print("Analyzing target data...")
    generator.analyze_target_data(target_data)
    debug_print("Target data analysis completed")
    
    # Create data generator
    debug_print("Creating data generator...")
    if generator.cluster_centers is not None and len(generator.cluster_centers) >= 2:
        mode1_center = tuple(generator.cluster_centers[0])
        mode2_center = tuple(generator.cluster_centers[1])
    else:
        mode1_center = (-1.5, -1.5)
        mode2_center = (1.5, 1.5)
    
    data_generator = BimodalDataGenerator(
        batch_size=8,  # Small batch size
        n_features=2,
        mode1_center=mode1_center,
        mode2_center=mode2_center,
        mode_std=0.3
    )
    debug_print("Data generator created")
    
    # Create optimizers
    debug_print("Creating optimizers...")
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.5)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.5)
    debug_print("Optimizers created")
    
    # Training loop
    epochs = 5  # Very short training
    debug_print(f"Starting training loop: {epochs} epochs")
    
    for epoch in range(epochs):
        debug_print(f"=== EPOCH {epoch + 1}/{epochs} ===")
        epoch_start = time.time()
        
        # Single training step per epoch for debugging
        debug_print("Getting real batch...")
        real_batch = data_generator.generate_batch()
        debug_print(f"Real batch shape: {real_batch.shape}")
        
        batch_size = tf.shape(real_batch)[0]
        debug_print(f"Batch size: {batch_size}")
        
        # Train discriminator
        debug_print("Training discriminator...")
        z = tf.random.normal([batch_size, generator.latent_dim])
        debug_print(f"Generated latent vector: {z.shape}")
        
        with tf.GradientTape() as tape:
            debug_print("Generating fake samples...")
            fake_batch = generator.generate(z)
            debug_print(f"Fake batch generated: {fake_batch.shape}")
            
            debug_print("Computing discriminator outputs...")
            real_output = discriminator.discriminate(real_batch)
            fake_output = discriminator.discriminate(fake_batch)
            debug_print(f"Discriminator outputs: real {real_output.shape}, fake {fake_output.shape}")
            
            debug_print("Computing discriminator loss...")
            w_distance = tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)
            d_loss = -w_distance
            debug_print(f"Discriminator loss: {d_loss.numpy():.4f}")
        
        debug_print("Computing discriminator gradients...")
        d_gradients = tape.gradient(d_loss, discriminator.trainable_variables)
        d_grad_count = sum(1 for g in d_gradients if g is not None)
        debug_print(f"Discriminator gradients: {d_grad_count}/{len(discriminator.trainable_variables)}")
        
        debug_print("Applying discriminator gradients...")
        d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
        debug_print("Discriminator step completed")
        
        # Train generator
        debug_print("Training generator...")
        z = tf.random.normal([batch_size, generator.latent_dim])
        
        with tf.GradientTape() as tape:
            debug_print("Generating fake samples for generator...")
            fake_batch = generator.generate(z)
            fake_output = discriminator.discriminate(fake_batch)
            
            debug_print("Computing generator loss...")
            g_loss = -tf.reduce_mean(fake_output)
            debug_print(f"Generator loss: {g_loss.numpy():.4f}")
        
        debug_print("Computing generator gradients...")
        g_gradients = tape.gradient(g_loss, generator.trainable_variables)
        g_grad_count = sum(1 for g in g_gradients if g is not None)
        debug_print(f"Generator gradients: {g_grad_count}/{len(generator.trainable_variables)}")
        
        debug_print("Applying generator gradients...")
        g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
        debug_print("Generator step completed")
        
        epoch_time = time.time() - epoch_start
        debug_print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        debug_print(f"Losses: G={g_loss.numpy():.4f}, D={d_loss.numpy():.4f}")
        
        # Test generation
        debug_print("Testing generation...")
        z_test = tf.random.normal([4, generator.latent_dim])
        generated_samples = generator.generate(z_test)
        debug_print(f"Generated samples: {generated_samples.shape}")
        debug_print(f"Sample range: [{generated_samples.numpy().min():.3f}, {generated_samples.numpy().max():.3f}]")
        
        debug_print(f"=== END EPOCH {epoch + 1} ===\n")
    
    debug_print("Training completed successfully!")
    
    # Final test
    debug_print("Final generation test...")
    z_final = tf.random.normal([10, generator.latent_dim])
    final_samples = generator.generate(z_final)
    debug_print(f"Final samples shape: {final_samples.shape}")
    debug_print(f"Final sample statistics:")
    debug_print(f"  Mean: [{final_samples.numpy().mean(axis=0)[0]:.3f}, {final_samples.numpy().mean(axis=0)[1]:.3f}]")
    debug_print(f"  Std:  [{final_samples.numpy().std(axis=0)[0]:.3f}, {final_samples.numpy().std(axis=0)[1]:.3f}]")
    
    return generator, discriminator


def main():
    """Main function."""
    debug_print("=" * 80)
    debug_print("SIMPLIFIED CLUSTERIZED QUANTUM GAN TRAINING")
    debug_print("=" * 80)
    
    try:
        generator, discriminator = simple_training_loop()
        debug_print("SUCCESS: Training completed without hanging!")
    except Exception as e:
        debug_print(f"ERROR: Training failed with exception: {e}")
        import traceback
        debug_print(f"Traceback: {traceback.format_exc()}")
    
    debug_print("=" * 80)


if __name__ == "__main__":
    main()
