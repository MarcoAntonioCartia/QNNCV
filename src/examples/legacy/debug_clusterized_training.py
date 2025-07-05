"""
Debug version of Clusterized Quantum GAN Training

This script adds extensive logging to identify where the training hangs.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.generators.clusterized_quantum_generator import ClusterizedQuantumGenerator
from src.models.discriminators.pure_sf_discriminator import PureSFDiscriminator
from src.losses.improved_quantum_gan_loss import ImprovedQuantumWassersteinLoss
from src.training.data_generators import BimodalDataGenerator


def debug_print(message: str):
    """Print debug message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] DEBUG: {message}")
    sys.stdout.flush()


def test_basic_operations():
    """Test basic TensorFlow and quantum operations."""
    debug_print("Testing basic TensorFlow operations...")
    
    # Test TensorFlow
    try:
        x = tf.constant([1.0, 2.0, 3.0])
        y = tf.reduce_mean(x)
        debug_print(f"TensorFlow test passed: {y.numpy()}")
    except Exception as e:
        debug_print(f"TensorFlow test failed: {e}")
        return False
    
    # Test random generation
    try:
        z = tf.random.normal([4, 6])
        debug_print(f"Random generation test passed: shape {z.shape}")
    except Exception as e:
        debug_print(f"Random generation test failed: {e}")
        return False
    
    return True


def test_data_generator():
    """Test the data generator."""
    debug_print("Testing BimodalDataGenerator...")
    
    try:
        data_gen = BimodalDataGenerator(
            batch_size=8,
            n_features=2,
            mode1_center=(-1.5, -1.5),
            mode2_center=(1.5, 1.5),
            mode_std=0.3
        )
        debug_print("Data generator created successfully")
        
        batch = data_gen.generate_batch()
        debug_print(f"Data batch generated: shape {batch.shape}")
        
        return data_gen
    except Exception as e:
        debug_print(f"Data generator test failed: {e}")
        return None


def test_generator_creation():
    """Test creating the clusterized generator."""
    debug_print("Testing ClusterizedQuantumGenerator creation...")
    
    try:
        # Use minimal complexity
        generator = ClusterizedQuantumGenerator(
            latent_dim=4,  # Reduced from 6
            output_dim=2,
            n_modes=2,     # Reduced from 4
            layers=1,      # Reduced from 2
            cutoff_dim=4,  # Reduced from 6
            clustering_method='kmeans',
            coordinate_names=['X', 'Y']
        )
        debug_print("Generator created successfully")
        debug_print(f"Generator has {len(generator.trainable_variables)} trainable variables")
        
        return generator
    except Exception as e:
        debug_print(f"Generator creation failed: {e}")
        return None


def test_discriminator_creation():
    """Test creating the discriminator."""
    debug_print("Testing PureSFDiscriminator creation...")
    
    try:
        discriminator = PureSFDiscriminator(
            input_dim=2,
            n_modes=2,
            layers=1,
            cutoff_dim=4
        )
        debug_print("Discriminator created successfully")
        debug_print(f"Discriminator has {len(discriminator.trainable_variables)} trainable variables")
        
        return discriminator
    except Exception as e:
        debug_print(f"Discriminator creation failed: {e}")
        return None


def test_generator_execution(generator):
    """Test generator execution step by step."""
    debug_print("Testing generator execution...")
    
    try:
        # Create test data for analysis
        debug_print("Creating test data...")
        np.random.seed(42)
        cluster1 = np.random.normal([-1.5, -1.5], 0.3, (50, 2))
        cluster2 = np.random.normal([1.5, 1.5], 0.3, (50, 2))
        target_data = np.vstack([cluster1, cluster2])
        debug_print(f"Test data created: shape {target_data.shape}")
        
        # Analyze target data
        debug_print("Analyzing target data...")
        generator.analyze_target_data(target_data)
        debug_print("Target data analysis completed")
        
        # Test generation with small batch
        debug_print("Testing generation with small batch...")
        z_test = tf.random.normal([2, generator.latent_dim])  # Very small batch
        debug_print(f"Created latent vector: shape {z_test.shape}")
        
        debug_print("Calling generator.generate()...")
        start_time = time.time()
        generated = generator.generate(z_test)
        end_time = time.time()
        
        debug_print(f"Generation completed in {end_time - start_time:.3f}s")
        debug_print(f"Generated samples shape: {generated.shape}")
        debug_print(f"Generated samples range: [{generated.numpy().min():.3f}, {generated.numpy().max():.3f}]")
        
        return True
    except Exception as e:
        debug_print(f"Generator execution failed: {e}")
        import traceback
        debug_print(f"Traceback: {traceback.format_exc()}")
        return False


def test_discriminator_execution(discriminator):
    """Test discriminator execution."""
    debug_print("Testing discriminator execution...")
    
    try:
        # Create test input
        test_input = tf.random.normal([2, 2])
        debug_print(f"Created test input: shape {test_input.shape}")
        
        debug_print("Calling discriminator.discriminate()...")
        start_time = time.time()
        output = discriminator.discriminate(test_input)
        end_time = time.time()
        
        debug_print(f"Discrimination completed in {end_time - start_time:.3f}s")
        debug_print(f"Discriminator output shape: {output.shape}")
        debug_print(f"Discriminator output: {output.numpy().flatten()}")
        
        return True
    except Exception as e:
        debug_print(f"Discriminator execution failed: {e}")
        import traceback
        debug_print(f"Traceback: {traceback.format_exc()}")
        return False


def test_training_step(generator, discriminator, data_gen):
    """Test a single training step."""
    debug_print("Testing single training step...")
    
    try:
        # Create loss function
        debug_print("Creating loss function...")
        loss_fn = ImprovedQuantumWassersteinLoss(
            lambda_gp=1.0,
            lambda_entropy=0.5,
            gp_center=0.0,
            noise_std=0.01
        )
        debug_print("Loss function created")
        
        # Create optimizers
        debug_print("Creating optimizers...")
        g_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.5)
        d_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.5)
        debug_print("Optimizers created")
        
        # Get real batch
        debug_print("Generating real batch...")
        real_batch = data_gen.generate_batch()
        debug_print(f"Real batch shape: {real_batch.shape}")
        
        # Generate latent vector
        debug_print("Generating latent vector...")
        batch_size = tf.shape(real_batch)[0]
        z = tf.random.normal([batch_size, generator.latent_dim])
        debug_print(f"Latent vector shape: {z.shape}")
        
        # Test discriminator training step
        debug_print("Testing discriminator training step...")
        with tf.GradientTape() as tape:
            debug_print("Generating fake samples...")
            fake_batch = generator.generate(z)
            debug_print(f"Fake batch generated: shape {fake_batch.shape}")
            
            debug_print("Computing discriminator outputs...")
            real_output = discriminator.discriminate(real_batch)
            fake_output = discriminator.discriminate(fake_batch)
            debug_print(f"Discriminator outputs computed: real {real_output.shape}, fake {fake_output.shape}")
            
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
        
        # Test generator training step
        debug_print("Testing generator training step...")
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
        
        debug_print("Training step completed successfully!")
        return True
        
    except Exception as e:
        debug_print(f"Training step failed: {e}")
        import traceback
        debug_print(f"Traceback: {traceback.format_exc()}")
        return False


def main():
    """Main debug function."""
    debug_print("=" * 80)
    debug_print("CLUSTERIZED QUANTUM GAN DEBUG SESSION")
    debug_print("=" * 80)
    
    # Suppress TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    # Test 1: Basic operations
    debug_print("\n1. Testing basic operations...")
    if not test_basic_operations():
        debug_print("FAILED: Basic operations test failed")
        return
    
    # Test 2: Data generator
    debug_print("\n2. Testing data generator...")
    data_gen = test_data_generator()
    if data_gen is None:
        debug_print("FAILED: Data generator test failed")
        return
    
    # Test 3: Generator creation
    debug_print("\n3. Testing generator creation...")
    generator = test_generator_creation()
    if generator is None:
        debug_print("FAILED: Generator creation failed")
        return
    
    # Test 4: Discriminator creation
    debug_print("\n4. Testing discriminator creation...")
    discriminator = test_discriminator_creation()
    if discriminator is None:
        debug_print("FAILED: Discriminator creation failed")
        return
    
    # Test 5: Generator execution
    debug_print("\n5. Testing generator execution...")
    if not test_generator_execution(generator):
        debug_print("FAILED: Generator execution failed")
        return
    
    # Test 6: Discriminator execution
    debug_print("\n6. Testing discriminator execution...")
    if not test_discriminator_execution(discriminator):
        debug_print("FAILED: Discriminator execution failed")
        return
    
    # Test 7: Training step
    debug_print("\n7. Testing training step...")
    if not test_training_step(generator, discriminator, data_gen):
        debug_print("FAILED: Training step failed")
        return
    
    debug_print("\n" + "=" * 80)
    debug_print("ALL TESTS PASSED! The issue is likely in the training loop.")
    debug_print("=" * 80)


if __name__ == "__main__":
    main()
