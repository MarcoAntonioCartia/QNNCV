"""
Basic test script to verify QGAN implementation works.
This tests the current classical GAN with synthetic data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from utils import load_synthetic_data, plot_results
from NN.classical_generator import ClassicalGenerator
from NN.classical_discriminator import ClassicalDiscriminator
from main_qgan import QGAN

def test_classical_gan():
    """Test classical GAN with 2D synthetic data."""
    
    print("Testing Classical GAN Implementation")
    print("=====================================")
    
    # Load synthetic 2D data for visualization
    data = load_synthetic_data(dataset_type="moons", num_samples=500)
    print(f"Loaded data shape: {data.shape}")
    
    # Initialize classical GAN components
    generator = ClassicalGenerator(latent_dim=2, output_dim=2)
    discriminator = ClassicalDiscriminator(input_dim=2)
    
    print("Created generator and discriminator")
    
    # Initialize QGAN
    qgan = QGAN(generator=generator, discriminator=discriminator, latent_dim=2)
    print("Initialized QGAN")
    
    # Generate some initial samples to test
    test_noise = tf.random.normal([100, 2])
    generated_samples = generator.generate(test_noise)
    print(f"Generated samples shape: {generated_samples.shape}")
    
    # Test discriminator
    real_output = discriminator.discriminate(data[:100])
    fake_output = discriminator.discriminate(generated_samples)
    print(f"Discriminator outputs - Real: {real_output.shape}, Fake: {fake_output.shape}")
    
    print("Basic functionality test passed!")
    
    # Quick training test (just a few epochs)
    print("\nRunning quick training test (5 epochs)...")
    try:
        qgan.train(data, epochs=5, batch_size=32)
        print("Training test completed successfully!")
        
        # Generate final samples for comparison
        final_samples = generator.generate(test_noise)
        
        # Plot results
        plot_results(data[:100], final_samples, epoch=5)
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_classical_gan()
    if success:
        print("\nAll tests passed! Classical GAN is working correctly.")
    else:
        print("\nTests failed. Check the implementation.") 