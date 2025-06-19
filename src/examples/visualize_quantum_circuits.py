"""
Visualize Quantum Circuits for Generator, Discriminator, and QGAN

This script demonstrates the circuit visualization capabilities for our
modular quantum GAN architecture.
"""

import numpy as np
import tensorflow as tf
import logging
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import warning suppression
from utils.warning_suppression import suppress_all_quantum_warnings
suppress_all_quantum_warnings()

# Import our components
from models.generators.quantum_generator import PureQuantumGenerator
from models.discriminators.quantum_discriminator import PureQuantumDiscriminator
from models.quantum_gan import QuantumGAN
from utils.quantum_circuit_visualizer import visualize_circuit, QuantumCircuitVisualizer


def visualize_generator():
    """Visualize the quantum generator circuit."""
    logger.info("="*80)
    logger.info("🎨 QUANTUM GENERATOR VISUALIZATION")
    logger.info("="*80)
    
    # Create generator
    generator = PureQuantumGenerator(
        latent_dim=6,
        output_dim=2,
        n_modes=4,
        layers=2,
        cutoff_dim=6,
        measurement_type='raw'
    )
    
    logger.info(f"\nGenerator Configuration:")
    logger.info(f"  Latent dimension: {generator.latent_dim}")
    logger.info(f"  Output dimension: {generator.output_dim}")
    logger.info(f"  Quantum modes: {generator.n_modes}")
    logger.info(f"  Circuit layers: {generator.layers}")
    logger.info(f"  Cutoff dimension: {generator.cutoff_dim}")
    
    # Visualize the quantum circuit
    print("\n" + "="*80)
    print("GENERATOR QUANTUM CIRCUIT")
    print("="*80)
    
    # Full circuit diagram
    visualize_circuit(generator.circuit, style='full', show_values=True)
    
    # Compact view
    visualize_circuit(generator.circuit, style='compact')
    
    return generator


def visualize_discriminator():
    """Visualize the quantum discriminator circuit."""
    logger.info("\n" + "="*80)
    logger.info("🔍 QUANTUM DISCRIMINATOR VISUALIZATION")
    logger.info("="*80)
    
    # Create discriminator
    discriminator = PureQuantumDiscriminator(
        input_dim=2,
        n_modes=2,
        layers=2,
        cutoff_dim=6,
        measurement_type='raw'
    )
    
    logger.info(f"\nDiscriminator Configuration:")
    logger.info(f"  Input dimension: {discriminator.input_dim}")
    logger.info(f"  Quantum modes: {discriminator.n_modes}")
    logger.info(f"  Circuit layers: {discriminator.layers}")
    logger.info(f"  Cutoff dimension: {discriminator.cutoff_dim}")
    
    # Visualize the quantum circuit
    print("\n" + "="*80)
    print("DISCRIMINATOR QUANTUM CIRCUIT")
    print("="*80)
    
    # Full circuit diagram
    visualize_circuit(discriminator.circuit, style='full', show_values=True)
    
    # Parameter list
    visualize_circuit(discriminator.circuit, style='list', show_values=True)
    
    return discriminator


def visualize_qgan():
    """Visualize both circuits in a QGAN setup."""
    logger.info("\n" + "="*80)
    logger.info("🌟 QUANTUM GAN COMPLETE VISUALIZATION")
    logger.info("="*80)
    
    # Create QGAN
    generator_config = {
        'latent_dim': 6,
        'output_dim': 2,
        'n_modes': 3,
        'layers': 1,
        'cutoff_dim': 4,
        'measurement_type': 'raw'
    }
    
    discriminator_config = {
        'input_dim': 2,
        'n_modes': 2,
        'layers': 1,
        'cutoff_dim': 4,
        'measurement_type': 'raw'
    }
    
    qgan = QuantumGAN(
        generator_config=generator_config,
        discriminator_config=discriminator_config,
        loss_type='wasserstein'
    )
    
    logger.info(f"\nQGAN Configuration:")
    logger.info(f"  Loss type: {qgan.loss_type}")
    logger.info(f"  Generator parameters: {len(qgan.generator.trainable_variables)}")
    logger.info(f"  Discriminator parameters: {len(qgan.discriminator.trainable_variables)}")
    
    # Visualize both circuits
    print("\n" + "="*60)
    print("QGAN GENERATOR CIRCUIT (Compact View)")
    print("="*60)
    visualize_circuit(qgan.generator.circuit, style='compact')
    
    print("\n" + "="*60)
    print("QGAN DISCRIMINATOR CIRCUIT (Compact View)")
    print("="*60)
    visualize_circuit(qgan.discriminator.circuit, style='compact')
    
    # Show parameter statistics
    print("\n" + "="*80)
    print("📊 QGAN PARAMETER STATISTICS")
    print("="*80)
    
    gen_visualizer = QuantumCircuitVisualizer(qgan.generator.circuit)
    disc_visualizer = QuantumCircuitVisualizer(qgan.discriminator.circuit)
    
    gen_info = gen_visualizer.export_parameter_info()
    disc_info = disc_visualizer.export_parameter_info()
    
    print(f"\nGenerator Circuit:")
    print(f"  Total parameters: {gen_info['circuit_info']['total_parameters']}")
    print(f"  Modes: {gen_info['circuit_info']['n_modes']}")
    print(f"  Layers: {gen_info['circuit_info']['layers']}")
    
    print(f"\nDiscriminator Circuit:")
    print(f"  Total parameters: {disc_info['circuit_info']['total_parameters']}")
    print(f"  Modes: {disc_info['circuit_info']['n_modes']}")
    print(f"  Layers: {disc_info['circuit_info']['layers']}")
    
    print(f"\nTotal QGAN Parameters: {gen_info['circuit_info']['total_parameters'] + disc_info['circuit_info']['total_parameters']}")
    
    return qgan


def demonstrate_parameter_evolution():
    """Show how parameters change during a training step."""
    logger.info("\n" + "="*80)
    logger.info("📈 PARAMETER EVOLUTION DEMONSTRATION")
    logger.info("="*80)
    
    # Create simple generator
    generator = PureQuantumGenerator(
        latent_dim=4,
        output_dim=2,
        n_modes=2,
        layers=1,
        cutoff_dim=4,
        measurement_type='raw'
    )
    
    # Get initial parameters
    initial_params = {var.name: var.numpy().copy() for var in generator.trainable_variables}
    
    print("\n" + "="*60)
    print("INITIAL PARAMETER VALUES")
    print("="*60)
    visualize_circuit(generator.circuit, style='list', show_values=True)
    
    # Simulate a training step
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    
    with tf.GradientTape() as tape:
        # Generate some samples
        z = tf.random.normal([4, 4])  # batch_size=4, latent_dim=4
        output = generator.generate(z)
        
        # Simple loss
        loss = tf.reduce_mean(tf.square(output))
    
    # Apply gradients
    gradients = tape.gradient(loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    
    print("\n" + "="*60)
    print("PARAMETER VALUES AFTER ONE TRAINING STEP")
    print("="*60)
    visualize_circuit(generator.circuit, style='list', show_values=True)
    
    # Show changes
    print("\n" + "="*60)
    print("PARAMETER CHANGES")
    print("="*60)
    
    for var in generator.trainable_variables:
        initial_val = initial_params[var.name]
        current_val = var.numpy()
        change = current_val - initial_val
        print(f"{var.name}: Δ = {change:.6f}")


def main():
    """Run all visualization demonstrations."""
    logger.info("🚀 QUANTUM CIRCUIT VISUALIZATION DEMONSTRATION")
    logger.info("="*50)
    
    # Visualize individual components
    generator = visualize_generator()
    discriminator = visualize_discriminator()
    
    # Visualize complete QGAN
    qgan = visualize_qgan()
    
    # Demonstrate parameter evolution
    demonstrate_parameter_evolution()
    
    logger.info("\n" + "="*80)
    logger.info("✅ VISUALIZATION COMPLETE")
    logger.info("="*80)
    logger.info("\nThe visualizations show:")
    logger.info("  - Circuit topology with quantum gates")
    logger.info("  - Individual parameter names and values")
    logger.info("  - Layer structure and mode connections")
    logger.info("  - Parameter statistics and distributions")
    logger.info("\nThis helps understand the quantum circuit architecture and debug training issues.")


if __name__ == "__main__":
    main()
