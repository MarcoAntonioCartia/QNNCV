"""
Simple Quantum Circuit Visualizer

This script demonstrates the circuit visualization capabilities for our
modular quantum architecture without requiring the full QGAN implementation.
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
from quantum.core.quantum_circuit import PureQuantumCircuit
from utils.quantum_circuit_visualizer import visualize_circuit, QuantumCircuitVisualizer


def visualize_simple_circuit():
    """Visualize a simple quantum circuit."""
    logger.info("="*80)
    logger.info("🔬 SIMPLE QUANTUM CIRCUIT VISUALIZATION")
    logger.info("="*80)
    
    # Create circuit
    circuit = PureQuantumCircuit(
        n_modes=3,
        layers=2,
        cutoff_dim=6
    )
    
    logger.info(f"\nCircuit Configuration:")
    logger.info(f"  Quantum modes: {circuit.n_modes}")
    logger.info(f"  Circuit layers: {circuit.layers}")
    logger.info(f"  Cutoff dimension: {circuit.cutoff_dim}")
    logger.info(f"  Total parameters: {len(circuit.trainable_variables)}")
    
    # Visualize the quantum circuit
    print("\n" + "="*80)
    print("QUANTUM CIRCUIT DIAGRAM")
    print("="*80)
    
    # Full circuit diagram
    visualize_circuit(circuit, style='full', show_values=True)
    
    # Compact view
    visualize_circuit(circuit, style='compact')
    
    # Parameter list
    visualize_circuit(circuit, style='list', show_values=True)
    
    return circuit


def demonstrate_parameter_evolution():
    """Show how parameters change during a training step."""
    logger.info("\n" + "="*80)
    logger.info("📈 PARAMETER EVOLUTION DEMONSTRATION")
    logger.info("="*80)
    
    # Create simple circuit
    circuit = PureQuantumCircuit(
        n_modes=2,
        layers=1,
        cutoff_dim=4
    )
    
    # Get initial parameters
    initial_params = {var.name: var.numpy().copy() for var in circuit.trainable_variables}
    
    print("\n" + "="*60)
    print("INITIAL PARAMETER VALUES")
    print("="*60)
    visualize_circuit(circuit, style='list', show_values=True)
    
    # Simulate a training step
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    
    with tf.GradientTape() as tape:
        # Simple dummy loss - sum of all parameters
        loss = tf.reduce_sum([tf.reduce_sum(var) for var in circuit.trainable_variables])
    
    # Apply gradients
    gradients = tape.gradient(loss, circuit.trainable_variables)
    optimizer.apply_gradients(zip(gradients, circuit.trainable_variables))
    
    print("\n" + "="*60)
    print("PARAMETER VALUES AFTER ONE TRAINING STEP")
    print("="*60)
    visualize_circuit(circuit, style='list', show_values=True)
    
    # Show changes
    print("\n" + "="*60)
    print("PARAMETER CHANGES")
    print("="*60)
    
    for var in circuit.trainable_variables:
        initial_val = initial_params[var.name]
        current_val = var.numpy()
        change = current_val - initial_val
        print(f"{var.name}: Δ = {change:.6f}")


def main():
    """Run all visualization demonstrations."""
    logger.info("🚀 QUANTUM CIRCUIT VISUALIZATION DEMONSTRATION")
    logger.info("="*50)
    
    # Visualize simple circuit
    circuit = visualize_simple_circuit()
    
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
