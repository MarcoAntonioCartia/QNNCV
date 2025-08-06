"""
Simple State Learner QGAN - Based on Tutorial Approach

This script implements a simple 1D quantum generator using the universal
CV quantum neural network architecture from the state learner tutorial.
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

# Import warning suppression first
from src.utils.warning_suppression import enable_clean_training, clean_training_output

# Import existing modular components
from src.quantum.core.pure_sf_circuit import PureSFQuantumCircuit

# Enable clean training environment
enable_clean_training()


class SimpleStateLearnerGenerator:
    """
    Simple quantum generator based on state learner tutorial.
    
    Uses universal CV quantum neural network architecture:
    - Displacement gates for input encoding
    - Squeezing gates for nonlinearity
    - Kerr interactions for universality
    - Direct parameter optimization
    """
    
    def __init__(self, latent_dim: int = 4, depth: int = 2, cutoff: int = 6):
        """
        Initialize simple state learner generator.
        
        Args:
            latent_dim: Dimension of latent input
            depth: Number of quantum layers
            cutoff: Fock space cutoff dimension
        """
        self.latent_dim = latent_dim
        self.depth = depth
        self.cutoff = cutoff
        
        # Create quantum circuit with universal architecture
        self.quantum_circuit = PureSFQuantumCircuit(
            n_modes=1,  # Single mode for simplicity
            n_layers=depth,
            cutoff_dim=cutoff,
            circuit_type="variational"  # Includes displacement, squeezing, Kerr
        )
        
        # Simple latent encoder (linear mapping to quantum parameters)
        # Get the actual number of quantum parameters
        test_state = self.quantum_circuit.execute()
        num_quantum_params = len(self.quantum_circuit.trainable_variables)
        
        self.latent_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(8, activation='tanh'),
            tf.keras.layers.Dense(num_quantum_params, activation='tanh')
        ])
        
        # Simple output decoder
        self.output_decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(4, activation='tanh'),
            tf.keras.layers.Dense(1, activation='tanh')
        ])
        
        # Build models
        dummy_input = tf.random.normal([1, latent_dim])
        _ = self.latent_encoder(dummy_input)
        dummy_measurement = tf.random.normal([1, 1])
        _ = self.output_decoder(dummy_measurement)
        
        print(f"🔧 SimpleStateLearnerGenerator initialized:")
        print(f"   Architecture: {latent_dim}D → 1 qumode → 1D")
        print(f"   Depth: {depth} layers")
        print(f"   Cutoff: {cutoff}")
        print(f"   Quantum parameters: {len(self.quantum_circuit.trainable_variables)}")
        print(f"   Total parameters: {len(self.trainable_variables)}")
    
    def generate(self, z: tf.Tensor) -> tf.Tensor:
        """
        Generate 1D samples using state learner approach.
        
        Args:
            z: Latent input [batch_size, latent_dim]
            
        Returns:
            Generated samples [batch_size, 1]
        """
        batch_size = tf.shape(z)[0]
        outputs = []
        
        for i in range(batch_size):
            # Encode latent to quantum parameters
            quantum_params = self.latent_encoder(z[i:i+1])  # [1, num_quantum_params]
            
            # Apply parameters to quantum circuit
            param_idx = 0
            for param_name, param_var in self.quantum_circuit.tf_parameters.items():
                param_shape = param_var.shape
                param_size = tf.reduce_prod(param_shape)
                param_values = quantum_params[0, param_idx:param_idx + param_size]
                param_values = tf.reshape(param_values, param_shape)
                param_var.assign(param_values)
                param_idx += param_size
            
            # Execute quantum circuit
            state = self.quantum_circuit.execute()
            
            # Extract measurement
            measurement = self.quantum_circuit.extract_measurements(state)  # [1]
            measurement = tf.reshape(measurement, [1, -1])  # [1, 1]
            
            # Decode to output
            output = self.output_decoder(measurement)
            outputs.append(output)
        
        return tf.concat(outputs, axis=0)
    
    @property
    def trainable_variables(self) -> list:
        """Get all trainable variables."""
        return (self.latent_encoder.trainable_variables + 
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


class SimpleStateLearnerTrainer:
    """Simple trainer for state learner QGAN."""
    
    def __init__(self, generator: SimpleStateLearnerGenerator, discriminator: SimpleClassicalDiscriminator):
        self.generator = generator
        self.discriminator = discriminator
        
        # Optimizers with higher learning rates for better learning
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)
        
        # Loss function
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        
    def train_step(self, real_data: tf.Tensor, batch_size: int = 32):
        """Single training step."""
        # Generate noise
        noise = tf.random.normal([batch_size, self.generator.latent_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake data
            fake_data = self.generator.generate(noise)
            
            # Discriminator predictions
            real_output = self.discriminator.discriminate(real_data)
            fake_output = self.discriminator.discriminate(fake_data)
            
            # Discriminator loss
            d_loss_real = self.cross_entropy(tf.ones_like(real_output), real_output)
            d_loss_fake = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
            d_loss = d_loss_real + d_loss_fake
            
            # Generator loss
            g_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
        
        # Calculate gradients
        g_gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        d_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        
        # Apply gradients
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        return {
            'g_loss': g_loss,
            'd_loss': d_loss,
            'real_output': tf.reduce_mean(real_output),
            'fake_output': tf.reduce_mean(fake_output)
        }
    
    def train(self, real_data: np.ndarray, epochs: int = 50, batch_size: int = 32, 
              save_interval: int = 10):
        """Train the QGAN."""
        print(f"🚀 Starting State Learner QGAN training for {epochs} epochs...")
        print(f"   Real data shape: {real_data.shape}")
        print(f"   Batch size: {batch_size}")
        
        # Convert to tensor
        real_data_tensor = tf.convert_to_tensor(real_data, dtype=tf.float32)
        
        # Training history
        history = {
            'g_loss': [], 'd_loss': [], 
            'real_output': [], 'fake_output': [],
            'coverage': [], 'std_ratio': []
        }
        
        for epoch in range(epochs):
            # Train step
            metrics = self.train_step(real_data_tensor, batch_size)
            
            # Generate samples for evaluation
            test_noise = tf.random.normal([100, self.generator.latent_dim])
            fake_samples = self.generator.generate(test_noise).numpy()
            
            # Calculate metrics
            real_std = np.std(real_data)
            fake_std = np.std(fake_samples)
            std_ratio = fake_std / real_std if real_std > 0 else 0
            
            real_range = np.max(real_data) - np.min(real_data)
            fake_range = np.max(fake_samples) - np.min(fake_samples)
            coverage = fake_range / real_range if real_range > 0 else 0
            
            # Store history
            history['g_loss'].append(float(metrics['g_loss']))
            history['d_loss'].append(float(metrics['d_loss']))
            history['real_output'].append(float(metrics['real_output']))
            history['fake_output'].append(float(metrics['fake_output']))
            history['coverage'].append(coverage)
            history['std_ratio'].append(std_ratio)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"   Epoch {epoch+1:3d}/{epochs}: "
                      f"G={metrics['g_loss']:.4f}, D={metrics['d_loss']:.4f}, "
                      f"Cov={coverage:.1%}, Std={std_ratio:.2f}")
            
            # Save intermediate results
            if (epoch + 1) % save_interval == 0:
                self.save_results(real_data, fake_samples, epoch + 1, history)
        
        return history
    
    def save_results(self, real_data: np.ndarray, fake_data: np.ndarray, 
                    epoch: int, history: Dict[str, list]):
        """Save training results."""
        # Create results directory
        os.makedirs("results/state_learner_qgan", exist_ok=True)
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogram comparison
        ax = axes[0, 0]
        ax.hist(real_data, bins=30, alpha=0.5, density=True, label='Real', color='blue')
        ax.hist(fake_data, bins=30, alpha=0.5, density=True, label='Generated', color='red')
        ax.set_title(f'Distribution Comparison (Epoch {epoch})')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Loss curves
        ax = axes[0, 1]
        epochs = range(1, len(history['g_loss']) + 1)
        ax.plot(epochs, history['g_loss'], label='Generator Loss', color='red')
        ax.plot(epochs, history['d_loss'], label='Discriminator Loss', color='blue')
        ax.set_title('Training Losses')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Coverage and std ratio
        ax = axes[1, 0]
        ax.plot(epochs, history['coverage'], label='Range Coverage', color='green')
        ax.plot(epochs, history['std_ratio'], label='Std Ratio', color='orange')
        ax.set_title('Quality Metrics')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Discriminator outputs
        ax = axes[1, 1]
        ax.plot(epochs, history['real_output'], label='Real Output', color='blue')
        ax.plot(epochs, history['fake_output'], label='Fake Output', color='red')
        ax.set_title('Discriminator Outputs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Output')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = f"results/state_learner_qgan/epoch_{epoch:03d}_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Results saved to: {save_path}")


def create_training_data(distribution_name: str, n_samples: int = 1000) -> np.ndarray:
    """Create training data for different distributions."""
    if distribution_name == "gaussian":
        return np.random.normal(0, 1, (n_samples, 1))
    elif distribution_name == "bimodal":
        n1 = n_samples // 2
        n2 = n_samples - n1
        samples1 = np.random.normal(-2, 0.5, (n1, 1))
        samples2 = np.random.normal(2, 0.5, (n2, 1))
        samples = np.vstack([samples1, samples2])
        np.random.shuffle(samples)
        return samples
    elif distribution_name == "uniform":
        return np.random.uniform(-3, 3, (n_samples, 1))
    else:
        raise ValueError(f"Unknown distribution: {distribution_name}")


def main():
    """Main training function using state learner approach."""
    with clean_training_output():
        print("=" * 80)
        print("🚀 SIMPLE STATE LEARNER QUANTUM GAN TRAINING")
        print("   Using universal CV quantum neural network architecture")
        print("=" * 80)
        
        # Test distributions
        distributions = ["gaussian", "bimodal", "uniform"]
        
        for dist_name in distributions:
            print(f"\n🎯 Training on {dist_name.upper()} distribution...")
            
            # Create training data
            real_data = create_training_data(dist_name, n_samples=1000)
            print(f"   Training data shape: {real_data.shape}")
            print(f"   Data range: [{real_data.min():.3f}, {real_data.max():.3f}]")
            print(f"   Data mean: {real_data.mean():.3f}, std: {real_data.std():.3f}")
            
            # Create models using state learner approach
            generator = SimpleStateLearnerGenerator(latent_dim=4, depth=2, cutoff=6)
            discriminator = SimpleClassicalDiscriminator(input_dim=1)
            
            # Create trainer
            trainer = SimpleStateLearnerTrainer(generator, discriminator)
            
            # Train
            history = trainer.train(
                real_data=real_data,
                epochs=50,  # More epochs for better learning
                batch_size=32,
                save_interval=10
            )
            
            print(f"✅ Training completed for {dist_name}")
            
            # Final evaluation
            test_noise = tf.random.normal([500, generator.latent_dim])
            final_samples = generator.generate(test_noise).numpy()
            
            print(f"   Final generated range: [{final_samples.min():.3f}, {final_samples.max():.3f}]")
            print(f"   Final generated mean: {final_samples.mean():.3f}, std: {final_samples.std():.3f}")
            
            # Calculate final metrics
            real_range = real_data.max() - real_data.min()
            fake_range = final_samples.max() - final_samples.min()
            coverage = fake_range / real_range if real_range > 0 else 0
            
            real_std = real_data.std()
            fake_std = final_samples.std()
            std_ratio = fake_std / real_std if real_std > 0 else 0
            
            print(f"   Final coverage: {coverage:.1%}")
            print(f"   Final std ratio: {std_ratio:.2f}")
            
            if coverage > 0.5 and std_ratio > 0.5:
                print("   🎉 SUCCESS: Good learning achieved!")
            elif coverage > 0.2 or std_ratio > 0.2:
                print("   ⚠️  PARTIAL: Some learning achieved")
            else:
                print("   ❌ FAILURE: Poor learning")


if __name__ == "__main__":
    main() 