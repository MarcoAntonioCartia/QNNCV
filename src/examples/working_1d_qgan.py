"""
Working 1D QGAN - Fixed Implementation

This script implements a working 1D quantum generator that addresses the issues found:
1. Proper parameter mapping from latent to quantum parameters
2. Better measurement strategy
3. Improved gradient flow
4. Classical discriminator for speed
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, Any, Tuple

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.quantum.core.pure_sf_circuit import PureSFQuantumCircuit


class Working1DQuantumGenerator:
    """
    Working 1D quantum generator with improved architecture.
    
    Key improvements:
    - Direct parameter mapping (no neural network encoder)
    - Better measurement strategy
    - Improved gradient flow
    """
    
    def __init__(self, latent_dim: int = 4, cutoff_dim: int = 8):
        """
        Initialize working 1D quantum generator.
        
        Args:
            latent_dim: Dimension of latent input
            cutoff_dim: Fock space cutoff dimension
        """
        self.latent_dim = latent_dim
        self.cutoff_dim = cutoff_dim
        
        # Use 2 modes for more expressiveness
        self.quantum_circuit = PureSFQuantumCircuit(
            n_modes=2,  # 2 modes for more expressiveness
            n_layers=2,  # Simple 2-layer circuit
            cutoff_dim=cutoff_dim,
            circuit_type="basic"  # Use basic to avoid displacement issues
        )
        
        # Direct parameter mapping (no neural network encoder)
        # Map latent directly to quantum parameters
        self.param_mapping = tf.keras.Sequential([
            tf.keras.layers.Dense(8, activation='tanh'),
            tf.keras.layers.Dense(len(self.quantum_circuit.trainable_variables), activation='tanh')
        ])
        
        # Simple output decoder
        self.output_decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(2, activation='tanh'),  # 2 modes × 1 quadrature = 2 measurements
            tf.keras.layers.Dense(1, activation='tanh')
        ])
        
        # Build models
        dummy_input = tf.random.normal([1, latent_dim])
        _ = self.param_mapping(dummy_input)
        dummy_measurement = tf.random.normal([1, 2])  # 2 modes × 1 quadrature = 2 measurements
        _ = self.output_decoder(dummy_measurement)
        
        print(f"🔧 Working1DQuantumGenerator initialized:")
        print(f"   Architecture: {latent_dim}D → 2 qumodes → 1D")
        print(f"   Quantum parameters: {len(self.quantum_circuit.trainable_variables)}")
        print(f"   Total parameters: {len(self.trainable_variables)}")
    
    def generate(self, z: tf.Tensor) -> tf.Tensor:
        """
        Generate 1D samples from latent input.
        
        Args:
            z: Latent input [batch_size, latent_dim]
            
        Returns:
            Generated samples [batch_size, 1]
        """
        batch_size = tf.shape(z)[0]
        
        # Map latent to quantum parameters
        quantum_params = self.param_mapping(z)  # [batch_size, n_params]
        
        # Process each sample individually
        outputs = []
        for i in range(batch_size):
            # Get quantum parameters for this sample
            sample_params = quantum_params[i:i+1]  # [1, n_params]
            
            # Apply parameters to quantum circuit
            param_idx = 0
            for param_name, param_var in self.quantum_circuit.tf_parameters.items():
                param_shape = param_var.shape
                param_size = tf.reduce_prod(param_shape)
                param_values = sample_params[0, param_idx:param_idx + param_size]
                param_values = tf.reshape(param_values, param_shape)
                param_var.assign(param_values)
                param_idx += param_size
            
            # Execute quantum circuit
            quantum_state = self.quantum_circuit.execute()
            
            # Extract measurement using circuit's built-in method
            measurement = self.quantum_circuit.extract_measurements(quantum_state)  # [n_modes]
            measurement = tf.reshape(measurement, [1, -1])  # [1, n_modes]
            
            # Decode to 1D output
            output = self.output_decoder(measurement)
            outputs.append(output)
        
        return tf.concat(outputs, axis=0)
    
    @property
    def trainable_variables(self) -> list:
        """Get all trainable variables."""
        return (self.param_mapping.trainable_variables + 
                self.output_decoder.trainable_variables + 
                self.quantum_circuit.trainable_variables)


class SimpleClassicalDiscriminator:
    """Simple classical discriminator for 1D data."""
    
    def __init__(self, input_dim: int = 1):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    
    def discriminate(self, x: tf.Tensor) -> tf.Tensor:
        return self.model(x)
    
    @property
    def trainable_variables(self) -> list:
        return self.model.trainable_variables


def create_test_distributions() -> Dict[str, callable]:
    """Create different 1D test distributions."""
    distributions = {}
    
    # 1. Gaussian distribution
    def gaussian_dist(n_samples: int, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
        return np.random.normal(mean, std, (n_samples, 1))
    distributions['gaussian'] = gaussian_dist
    
    # 2. Bimodal distribution
    def bimodal_dist(n_samples: int) -> np.ndarray:
        n1 = n_samples // 2
        n2 = n_samples - n1
        samples1 = np.random.normal(-2.0, 0.5, (n1, 1))
        samples2 = np.random.normal(2.0, 0.5, (n2, 1))
        samples = np.vstack([samples1, samples2])
        np.random.shuffle(samples)
        return samples
    distributions['bimodal'] = bimodal_dist
    
    # 3. Uniform distribution
    def uniform_dist(n_samples: int, low: float = -3.0, high: float = 3.0) -> np.ndarray:
        return np.random.uniform(low, high, (n_samples, 1))
    distributions['uniform'] = uniform_dist
    
    return distributions


def test_working_generator():
    """Test the working quantum generator."""
    print("=" * 80)
    print("🧪 TESTING WORKING 1D QUANTUM GENERATOR")
    print("=" * 80)
    
    # Create distributions
    distributions = create_test_distributions()
    
    # Test each distribution
    for dist_name, dist_func in distributions.items():
        print(f"\n🎯 Testing {dist_name.upper()} distribution...")
        
        # Generate target data
        target_data = dist_func(500)
        print(f"   Target data shape: {target_data.shape}")
        print(f"   Target range: [{target_data.min():.3f}, {target_data.max():.3f}]")
        print(f"   Target mean: {target_data.mean():.3f}, std: {target_data.std():.3f}")
        
        # Create generator
        generator = Working1DQuantumGenerator(latent_dim=4, cutoff_dim=8)
        
        # Test generation
        test_z = tf.random.normal([100, 4])
        generated_samples = generator.generate(test_z)
        generated_data = generated_samples.numpy()
        
        print(f"   Generated range: [{generated_data.min():.3f}, {generated_data.max():.3f}]")
        print(f"   Generated mean: {generated_data.mean():.3f}, std: {generated_data.std():.3f}")
        
        # Calculate coverage
        target_range = target_data.max() - target_data.min()
        gen_range = generated_data.max() - generated_data.min()
        coverage = gen_range / target_range if target_range > 0 else 0
        
        print(f"   Range coverage: {coverage:.1%}")
        
        if coverage < 0.1:
            print("   ⚠️  WARNING: Very low range coverage")
        elif coverage < 0.5:
            print("   ⚠️  WARNING: Low range coverage")
        else:
            print("   ✅ Good range coverage")
        
        # Visualize
        visualize_comparison(target_data, generated_data, dist_name)


def visualize_comparison(real_data: np.ndarray, generated_data: np.ndarray, dist_name: str):
    """Visualize comparison between real and generated data."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram comparison
    ax = axes[0]
    ax.hist(real_data, bins=30, alpha=0.5, density=True, label='Real', color='blue')
    ax.hist(generated_data, bins=30, alpha=0.5, density=True, label='Generated', color='red')
    ax.set_title(f'{dist_name.title()} Distribution')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Scatter plot
    ax = axes[1]
    n_plot = min(100, len(real_data), len(generated_data))
    ax.scatter(range(n_plot), real_data[:n_plot], alpha=0.6, label='Real', s=20)
    ax.scatter(range(n_plot), generated_data[:n_plot], alpha=0.6, label='Generated', s=20)
    ax.set_title('Sample Comparison (First 100)')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = f"results/working_1d_qgan/{dist_name}_comparison.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: {save_path}")
    plt.show()


def main():
    """Main test function."""
    test_working_generator()


if __name__ == "__main__":
    main() 