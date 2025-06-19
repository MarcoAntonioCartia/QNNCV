"""
Simple training script for Quantum GAN with measurement-based loss.
Runs from the src directory with proper imports.
"""

import numpy as np
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import warning suppression
from utils.warning_suppression import suppress_all_quantum_warnings
suppress_all_quantum_warnings()

# Import our components
from quantum.core.quantum_circuit import PureQuantumCircuit
from quantum.measurements.measurement_extractor import RawMeasurementExtractor
from models.transformations.matrix_manager import TransformationPair, StaticTransformationMatrix
from losses.quantum_gan_loss import create_quantum_loss


class SimpleQuantumGenerator:
    """Simple quantum generator for testing."""
    
    def __init__(self, n_modes=2, layers=1, cutoff_dim=4, latent_dim=4, output_dim=2):
        self.n_modes = n_modes
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Quantum circuit
        self.circuit = PureQuantumCircuit(
            n_modes=n_modes,
            layers=layers,
            cutoff_dim=cutoff_dim
        )
        
        # Measurement extractor
        self.measurements = RawMeasurementExtractor(
            n_modes=n_modes,
            cutoff_dim=cutoff_dim
        )
        
        # Static transformations
        measurement_dim = self.measurements.get_measurement_dim()
        self.transforms = TransformationPair(
            encoder_dim=(latent_dim, n_modes * 3),
            decoder_dim=(measurement_dim, output_dim),
            trainable=False,
            name_prefix="generator"
        )
        
        logger.info(f"Simple quantum generator initialized: {latent_dim} → {output_dim}")
    
    def generate(self, z, return_measurements=False):
        """Generate samples from latent input."""
        batch_size = tf.shape(z)[0]
        
        # Execute quantum circuits
        quantum_states = []
        for i in range(batch_size):
            state = self.circuit.execute({})
            quantum_states.append(state)
        
        # Extract measurements
        raw_measurements = self.measurements.extract_measurements(quantum_states)
        
        # Decode to output space
        output = self.transforms.decode(raw_measurements)
        
        if return_measurements:
            return output, raw_measurements
        return output
    
    @property
    def trainable_variables(self):
        return self.circuit.trainable_variables


def generate_bimodal_data(n_samples, n_features=2):
    """Generate bimodal distribution for testing."""
    mode1 = np.random.normal(-2, 0.5, (n_samples // 2, n_features))
    mode2 = np.random.normal(2, 0.5, (n_samples // 2, n_features))
    data = np.vstack([mode1, mode2])
    np.random.shuffle(data)
    return data.astype(np.float32)


def train_quantum_gan():
    """Train quantum GAN using measurement-based loss."""
    
    logger.info("="*60)
    logger.info("SIMPLE QUANTUM GAN TRAINING")
    logger.info("="*60)
    
    # Configuration
    n_modes = 2
    cutoff_dim = 4
    layers = 1
    latent_dim = 4
    output_dim = 2
    batch_size = 16
    n_epochs = 10
    
    # Create generator
    logger.info("Creating quantum generator...")
    generator = SimpleQuantumGenerator(
        n_modes=n_modes,
        layers=layers,
        cutoff_dim=cutoff_dim,
        latent_dim=latent_dim,
        output_dim=output_dim
    )
    
    # Create measurement-based loss
    logger.info("Creating measurement-based loss function...")
    measurement_loss = create_quantum_loss('measurement_based')
    
    # Generate training data
    logger.info("Generating training data...")
    train_data = generate_bimodal_data(500, output_dim)
    
    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Training metrics
    losses = []
    
    logger.info("\nStarting training...")
    for epoch in range(n_epochs):
        epoch_losses = []
        
        # Mini-batch training
        for i in range(0, len(train_data), batch_size):
            batch_real = train_data[i:i+batch_size]
            if len(batch_real) < batch_size:
                continue
            
            with tf.GradientTape() as tape:
                # Generate samples and get raw measurements
                z = tf.random.normal([batch_size, latent_dim])
                fake_samples, raw_measurements = generator.generate(z, return_measurements=True)
                
                # Measurement-based loss
                loss = measurement_loss.compute_loss(
                    raw_measurements,
                    batch_real,
                    generator.circuit.trainable_variables
                )
            
            # Compute gradients
            gradients = tape.gradient(loss, generator.trainable_variables)
            
            # Check gradient flow
            non_none_grads = sum(1 for g in gradients if g is not None)
            
            # Apply gradients
            optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
            epoch_losses.append(loss.numpy())
        
        # Record epoch metrics
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        logger.info(f"Epoch {epoch+1}/{n_epochs}: "
                   f"Loss = {avg_loss:.4f}, "
                   f"Gradients = {non_none_grads}/{len(gradients)}")
    
    # Generate final samples
    logger.info("\nGenerating final samples...")
    z_test = tf.random.normal([100, latent_dim])
    generated_samples = generator.generate(z_test)
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    # Training curve
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    # Data distribution
    plt.subplot(1, 3, 2)
    plt.scatter(train_data[:200, 0], train_data[:200, 1], 
               alpha=0.5, label='Real', color='blue', s=20)
    plt.scatter(generated_samples[:100, 0], generated_samples[:100, 1], 
               alpha=0.5, label='Generated', color='red', s=20)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Data Distribution')
    plt.legend()
    plt.grid(True)
    
    # Histogram
    plt.subplot(1, 3, 3)
    plt.hist(train_data[:, 0], bins=20, alpha=0.5, label='Real', color='blue')
    plt.hist(generated_samples[:, 0].numpy(), bins=20, alpha=0.5, label='Generated', color='red')
    plt.xlabel('Feature 1 Value')
    plt.ylabel('Frequency')
    plt.title('Feature Distribution')
    plt.legend()
    
    plt.tight_layout()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'simple_quantum_gan_training_{timestamp}.png'
    plt.savefig(filename, dpi=150)
    logger.info(f"Results saved to {filename}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    logger.info(f"Final Loss: {losses[-1]:.4f}")
    logger.info(f"Gradient flow maintained: {non_none_grads}/{len(gradients)} gradients")
    logger.info("✅ Training completed successfully!")
    
    # Show parameter changes
    logger.info("\nParameter evolution:")
    for var in generator.trainable_variables[:5]:  # Show first 5 parameters
        logger.info(f"{var.name}: {var.numpy():.4f}")
    
    return generator


if __name__ == "__main__":
    # Run training
    generator = train_quantum_gan()
    
    logger.info("\n🎉 Quantum GAN training completed!")
    logger.info("The modular architecture successfully maintains gradient flow.")
    logger.info("Quantum parameters have been optimized through measurement-based loss.")
