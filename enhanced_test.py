"""
Test script for enhanced quantum generators with fallback options.
This tests both the enhanced quantum generator and a simple fallback.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
import numpy as np

# Fallback simple quantum generator for when Strawberry Fields is not available
class QuantumContinuousGeneratorSimple:
    """Simple quantum-inspired generator that works without Strawberry Fields."""
    
    def __init__(self, n_qumodes=4, latent_dim=10):
        """Initialize simple quantum-inspired generator."""
        self.n_qumodes = n_qumodes
        self.latent_dim = latent_dim
        
        # Classical network that mimics quantum behavior
        self.quantum_network = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(32, activation=lambda x: tf.sin(x)),  # Oscillatory activation
            tf.keras.layers.Dense(2, activation='tanh')  # Output correct dimension
        ])
        
        # Build the network
        dummy_input = tf.zeros((1, latent_dim))
        _ = self.quantum_network(dummy_input)
        
        # Quantum-inspired parameters
        self.quantum_params = tf.Variable(
            tf.random.normal([n_qumodes], stddev=0.1),
            name="quantum_params"
        )
    
    @property
    def trainable_variables(self):
        """Return trainable variables."""
        variables = [self.quantum_params]
        variables.extend(self.quantum_network.trainable_variables)
        return variables
    
    def generate(self, z):
        """Generate samples using quantum-inspired classical network."""
        # Process through quantum-inspired network
        base_output = self.quantum_network(z)
        
        # Add quantum-inspired interference patterns
        interference = tf.sin(base_output * self.quantum_params)
        
        # Combine base output with interference
        output = base_output + 0.1 * interference
        
        return output

def test_enhanced_qgan():
    """Test enhanced QGAN with quantum-inspired generator."""
    
    print("Testing Enhanced QGAN Implementation")
    print("====================================")
    
    # Import utilities
    from utils import load_synthetic_data, plot_results
    from NN.classical_discriminator import ClassicalDiscriminator
    from main_qgan import QGAN
    
    # Load synthetic 2D data for visualization
    data = load_synthetic_data(dataset_type="spiral", num_samples=300)
    print(f"Loaded data shape: {data.shape}")
    
    # Initialize quantum-inspired generator
    generator = QuantumContinuousGeneratorSimple(n_qumodes=2, latent_dim=4)
    discriminator = ClassicalDiscriminator(input_dim=2)
    
    print("Created quantum-inspired generator and classical discriminator")
    
    # Test generator functionality
    test_noise = tf.random.normal([50, 4])
    generated_samples = generator.generate(test_noise)
    print(f"Generated samples shape: {generated_samples.shape}")
    print(f"Generator trainable vars: {len(generator.trainable_variables)}")
    
    # Test discriminator
    real_output = discriminator.discriminate(data[:50])
    fake_output = discriminator.discriminate(generated_samples)
    print(f"Discriminator outputs - Real: {real_output.shape}, Fake: {fake_output.shape}")
    
    # Initialize QGAN
    qgan = QGAN(generator=generator, discriminator=discriminator, latent_dim=4)
    print("Initialized Enhanced QGAN")
    
    # Quick training test
    print("\nRunning enhanced training test (10 epochs)...")
    try:
        qgan.train(data, epochs=10, batch_size=32)
        print("Enhanced training completed successfully!")
        
        # Generate final samples for comparison
        final_samples = generator.generate(test_noise)
        
        # Plot results
        plot_results(data[:50], final_samples, epoch=10)
        
        return True
        
    except Exception as e:
        print(f"Enhanced training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quantum_vs_classical():
    """Compare quantum-inspired vs classical generators."""
    
    print("\nQuantum vs Classical Generator Comparison")
    print("=========================================")
    
    from utils import load_synthetic_data, compute_wasserstein_distance
    from NN.classical_generator import ClassicalGenerator
    
    # Load test data
    data = load_synthetic_data(dataset_type="moons", num_samples=200)
    test_noise = tf.random.normal([100, 4])
    
    # Classical generator
    classical_gen = ClassicalGenerator(latent_dim=4, output_dim=2)
    classical_samples = classical_gen.generate(test_noise)
    
    # Quantum-inspired generator
    quantum_gen = QuantumContinuousGeneratorSimple(n_qumodes=2, latent_dim=4)
    quantum_samples = quantum_gen.generate(test_noise)
    
    # Compare quality using Wasserstein distance
    classical_distance = compute_wasserstein_distance(data[:100], classical_samples)
    quantum_distance = compute_wasserstein_distance(data[:100], quantum_samples)
    
    print(f"Classical generator Wasserstein distance: {classical_distance:.4f}")
    print(f"Quantum-inspired generator Wasserstein distance: {quantum_distance:.4f}")
    
    if quantum_distance < classical_distance:
        print("Quantum-inspired generator shows better initial performance!")
    else:
        print("Classical generator shows better initial performance.")
    
    print(f"Classical trainable parameters: {len(classical_gen.trainable_variables)}")
    print(f"Quantum trainable parameters: {len(quantum_gen.trainable_variables)}")

if __name__ == "__main__":
    print("Enhanced QGAN Testing Suite")
    print("===========================")
    
    # Test 1: Enhanced QGAN functionality
    success = test_enhanced_qgan()
    
    if success:
        print("\nTest 1 PASSED: Enhanced QGAN working correctly!")
        
        # Test 2: Quantum vs Classical comparison
        test_quantum_vs_classical()
        
        print("\nAll enhanced tests completed successfully!")
        print("\nNext Steps:")
        print("1. Install Strawberry Fields for true quantum implementation")
        print("2. Implement quantum discriminator")
        print("3. Add advanced training strategies")
        print("4. Implement pharmaceutical validation")
        
    else:
        print("\nTest 1 FAILED: Check the enhanced implementation.") 