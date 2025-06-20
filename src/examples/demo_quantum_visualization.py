"""
Demonstration of Comprehensive Quantum Visualization Manager

This example shows how to use the QuantumVisualizationManager with Pure SF 
Quantum GANs for complete circuit, state, and training visualization.
"""

import sys
import os
import numpy as np
import tensorflow as tf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Pure SF components
from src.models.generators.pure_sf_generator import PureSFGenerator
from src.models.discriminators.pure_sf_discriminator import PureSFDiscriminator
from src.training.quantum_gan_trainer import QuantumGANTrainer
from src.training.data_generators import BimodalDataGenerator
from src.utils.quantum_visualization_manager import CorrectedQuantumVisualizationManager
from src.utils.warning_suppression import suppress_all_quantum_warnings


def demo_circuit_visualization():
    """Demonstrate circuit structure visualization."""
    print("CIRCUIT VISUALIZATION DEMO")
    print("=" * 60)
    
    # Suppress warnings for cleaner output
    suppress_all_quantum_warnings()
    
    # Create visualization manager
    viz = CorrectedQuantumVisualizationManager(save_directory="demo_visualizations")
    
    # Create generator and discriminator
    generator = PureSFGenerator(
        latent_dim=4,
        output_dim=2,
        n_modes=4,
        layers=2,
        cutoff_dim=6
    )
    
    discriminator = PureSFDiscriminator(
        input_dim=2,
        n_modes=3,
        layers=2,
        cutoff_dim=6
    )
    
    # Visualize generator circuit
    print("\n1. Generator Circuit Analysis:")
    viz.integrate_with_pure_quantum_generator(generator, "Pure SF Generator")
    
    # Visualize discriminator circuit
    print("\n2. Discriminator Circuit Analysis:")
    viz.integrate_with_pure_quantum_discriminator(discriminator, "Pure SF Discriminator")
    
    return generator, discriminator, viz


def demo_qgan_comparison():
    """Demonstrate QGAN comparison dashboard."""
    print("\nQGAN COMPARISON DEMO")
    print("=" * 60)
    
    # Create models
    generator = PureSFGenerator(latent_dim=4, output_dim=2, n_modes=4, layers=2)
    discriminator = PureSFDiscriminator(input_dim=2, n_modes=3, layers=2)
    viz = CorrectedQuantumVisualizationManager()
    
    # Create sample data
    data_generator = BimodalDataGenerator(
        batch_size=100,
        n_features=2,
        mode1_center=(-2.0, -2.0),
        mode2_center=(2.0, 2.0),
        mode_std=0.5
    )
    real_data = data_generator.generate_dataset(n_batches=1)
    
    # Create comprehensive comparison
    viz.create_qgan_comparison_dashboard(
        generator=generator,
        discriminator=discriminator,
        real_data=real_data,
        n_samples=100,
        title="Pure SF QGAN Analysis"
    )
    
    return viz


def demo_training_visualization():
    """Demonstrate training progress visualization."""
    print("\nTRAINING VISUALIZATION DEMO")
    print("=" * 60)
    
    # Create visualization manager
    viz = CorrectedQuantumVisualizationManager()
    
    # Generate synthetic training history (simulating real training)
    epochs = 3
    
    def generate_realistic_curve(start, end, noise_level, epochs):
        """Generate realistic training curve."""
        base_trend = np.linspace(start, end, epochs)
        noise = np.random.normal(0, noise_level, epochs)
        oscillation = 0.1 * np.sin(np.linspace(0, 4*np.pi, epochs))
        return base_trend + noise + oscillation
    
    # Create realistic training history
    training_history = {
        'g_loss': generate_realistic_curve(2.5, 0.8, 0.2, epochs),
        'd_loss': generate_realistic_curve(1.8, 0.6, 0.15, epochs),
        'w_distance': generate_realistic_curve(1.0, 0.1, 0.05, epochs),
        'gradient_penalty': np.random.exponential(0.1, epochs),
        'g_gradients': 30 * np.ones(epochs) + np.random.normal(0, 2, epochs),
        'd_gradients': 22 * np.ones(epochs) + np.random.normal(0, 1.5, epochs),
        'entropy_bonus': -np.random.exponential(0.01, epochs),
        'physics_penalty': np.random.exponential(0.05, epochs)
    }
    
    # Create comprehensive training dashboard
    viz.create_training_dashboard(
        training_history=training_history,
        title="Pure SF QGAN Training Progress"
    )
    
    # Simulate parameter tracking
    print("\nSimulating parameter evolution tracking...")
    n_params = 52  # 30 generator + 22 discriminator
    for epoch in range(0, epochs, 5):  # Track every 5th epoch
        # Simulate parameter drift during training
        params = np.random.normal(0, 0.1 + 0.02*epoch/epochs, n_params)
        viz.parameter_history.append(params.tolist())
    
    # Recreate dashboard with parameter evolution
    viz.create_training_dashboard(
        training_history=training_history,
        title="Pure SF QGAN with Parameter Evolution"
    )
    
    return viz


def demo_mini_training_with_visualization():
    """Demonstrate actual mini-training with real-time visualization."""
    print("\n🚀 MINI-TRAINING WITH VISUALIZATION DEMO")
    print("=" * 60)
    
    # Create models
    generator = PureSFGenerator(latent_dim=4, output_dim=2, n_modes=4, layers=2)
    discriminator = PureSFDiscriminator(input_dim=2, n_modes=3, layers=2)
    
    # Create data generator
    data_generator = BimodalDataGenerator(
        batch_size=8,
        n_features=2,
        mode1_center=(-1.5, -1.5),
        mode2_center=(1.5, 1.5),
        mode_std=0.3
    )
    
    # Create trainer
    trainer = QuantumGANTrainer(
        generator=generator,
        discriminator=discriminator,
        entropy_weight=0.1,
        verbose=True
    )
    
    # Create visualization manager with parameter tracking
    viz = CorrectedQuantumVisualizationManager(save_directory="mini_training_viz")
    
    print("\nStarting mini-training (3 epochs for demo)...")
    
    # Mini training loop with visualization
    epochs = 3
    steps_per_epoch = 5
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        # Track parameters before epoch
        viz.track_parameters(generator, discriminator)
        
        # Train one epoch
        epoch_metrics = trainer.train_epoch(
            data_generator=data_generator,
            steps_per_epoch=steps_per_epoch,
            latent_dim=4,
            epoch_num=epoch + 1
        )
        
        print(f"Epoch {epoch + 1} Summary:")
        for key, value in epoch_metrics.items():
            print(f"  {key}: {value:.4f}")
    
    # Create final visualization with actual training data
    viz.create_training_dashboard(
        training_history=trainer.metrics_history,
        title="Mini-Training Results"
    )
    
    # Final QGAN comparison
    real_data = data_generator.generate_dataset(n_batches=10)
    viz.create_qgan_comparison_dashboard(
        generator=generator,
        discriminator=discriminator,
        real_data=real_data,
        n_samples=50,
        title="Post-Training Analysis"
    )
    
    return trainer, viz


def demo_visualization_summary():
    """Show summary of visualization capabilities."""
    print("\n📋 VISUALIZATION SYSTEM SUMMARY")
    print("=" * 60)
    
    viz = CorrectedQuantumVisualizationManager()
    summary = viz.get_visualization_summary()
    
    print("Available visualization features:")
    for i, feature in enumerate(summary['features'], 1):
        print(f"  {i:2d}. {feature}")
    
    print(f"\nVisualization Configuration:")
    print(f"  Save directory: {summary['save_directory']}")
    print(f"  Parameter tracking: {'Enabled' if summary['tracked_parameters'] > 0 else 'Available'}")
    print(f"  State tracking: {'Enabled' if summary['tracked_states'] > 0 else 'Available'}")
    
    print(f"\nIntegration Methods:")
    print(f"  - integrate_with_generator()")
    print(f"  - integrate_with_discriminator()")
    print(f"  - create_qgan_comparison_dashboard()")
    print(f"  - create_training_dashboard()")
    print(f"  - track_parameters()")
    
    print(f"\nQuick Access Functions:")
    print(f"  - quick_circuit_visualization()")
    print(f"  - quick_state_visualization()")
    print(f"  - create_visualization_manager()")


def main():
    """Main demonstration function."""
    print("COMPREHENSIVE QUANTUM VISUALIZATION DEMONSTRATION")
    print("=" * 80)
    print("This demo showcases the complete visualization system for Pure SF Quantum GANs")
    print("=" * 80)
    
    try:
        # 1. Circuit visualization
        generator, discriminator, viz1 = demo_circuit_visualization()
        
        # 2. QGAN comparison
        viz2 = demo_qgan_comparison()
        
        # 3. Training visualization
        viz3 = demo_training_visualization()
        
        # 4. Mini-training with real-time visualization
        trainer, viz4 = demo_mini_training_with_visualization()
        
        # 5. System summary
        demo_visualization_summary()
        
        print("\n" + "=" * 80)
        print("VISUALIZATION DEMONSTRATION COMPLETE!")
        print("=" * 80)
        print("Key Achievements:")
        print("  ✓ Circuit structure analysis and parameter visualization")
        print("  ✓ Quantum state visualization with Wigner functions")
        print("  ✓ Training progress dashboards with parameter evolution")
        print("  ✓ QGAN performance comparison and analysis")
        print("  ✓ Real-time visualization during training")
        print("  ✓ Comprehensive integration with Pure SF architecture")
        print("\nVisualization files saved to respective directories.")
        print("Ready to enhance your quantum machine learning research! 🚀")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
