"""
Debug Quantum Learning - Understanding Why Quantum Circuits Don't Learn

This script systematically investigates why the quantum generator isn't learning:
1. Test quantum circuit expressiveness
2. Check gradient flow
3. Analyze parameter sensitivity
4. Test different measurement strategies
5. Compare with classical baseline
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, Any, List, Tuple

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.quantum.core.pure_sf_circuit import PureSFQuantumCircuit


def test_quantum_circuit_expressiveness():
    """Test if quantum circuit can produce diverse outputs."""
    print("=" * 80)
    print("🔬 TESTING QUANTUM CIRCUIT EXPRESSIVENESS")
    print("=" * 80)
    
    # Test different parameter configurations
    circuit = PureSFQuantumCircuit(n_modes=1, n_layers=2, cutoff_dim=8)
    
    print(f"Circuit parameters: {len(circuit.trainable_variables)}")
    print(f"Parameter names: {circuit.sf_param_names}")
    
    # Test 1: Random parameters
    print("\n🎲 Test 1: Random Parameters")
    outputs_random = []
    for i in range(10):
        # Set random parameters
        for param_name, param_var in circuit.tf_parameters.items():
            param_shape = param_var.shape
            param_var.assign(tf.random.uniform(param_shape, -2.0, 2.0))
        
        state = circuit.execute()
        measurement = circuit.extract_measurements(state)
        outputs_random.append(measurement.numpy()[0])
    
    print(f"   Random outputs: {outputs_random}")
    print(f"   Range: [{min(outputs_random):.3f}, {max(outputs_random):.3f}]")
    print(f"   Std: {np.std(outputs_random):.3f}")
    
    # Test 2: Systematic parameter variation
    print("\n🎯 Test 2: Systematic Parameter Variation")
    outputs_systematic = []
    param_names = list(circuit.tf_parameters.keys())
    
    for i in range(10):
        # Vary one parameter at a time
        param_idx = i % len(param_names)
        param_name = param_names[param_idx]
        
        # Set all parameters to 0, then vary one
        for name, param_var in circuit.tf_parameters.items():
            param_shape = param_var.shape
            param_var.assign(tf.zeros(param_shape))
        
        # Vary the selected parameter
        param_shape = circuit.tf_parameters[param_name].shape
        circuit.tf_parameters[param_name].assign(tf.constant(float(i - 5), shape=param_shape))
        
        state = circuit.execute()
        measurement = circuit.extract_measurements(state)
        outputs_systematic.append(measurement.numpy()[0])
    
    print(f"   Systematic outputs: {outputs_systematic}")
    print(f"   Range: [{min(outputs_systematic):.3f}, {max(outputs_systematic):.3f}]")
    print(f"   Std: {np.std(outputs_systematic):.3f}")
    
    return outputs_random, outputs_systematic


def test_gradient_flow():
    """Test if gradients flow through quantum parameters."""
    print("\n" + "=" * 80)
    print("🔍 TESTING GRADIENT FLOW")
    print("=" * 80)
    
    circuit = PureSFQuantumCircuit(n_modes=1, n_layers=2, cutoff_dim=8)
    
    # Test gradient flow for each parameter
    print("Testing gradient flow for each parameter:")
    
    for param_name, param_var in circuit.tf_parameters.items():
        with tf.GradientTape() as tape:
            state = circuit.execute()
            measurement = circuit.extract_measurements(state)
            loss = tf.reduce_mean(measurement)
        
        gradients = tape.gradient(loss, [param_var])
        grad_value = gradients[0].numpy() if gradients[0] is not None else 0.0
        
        print(f"   {param_name}: gradient = {grad_value:.6f}")
    
    # Test gradient flow with input encoding
    print("\nTesting gradient flow with input encoding:")
    
    # Create simple input encoder
    input_encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation='tanh')
    ])
    
    test_input = tf.random.normal([1, 4])
    _ = input_encoder(test_input)  # Build the model
    
    with tf.GradientTape() as tape:
        # Encode input to quantum parameters
        encoded_params = input_encoder(test_input)
        
        # Apply encoded parameters to circuit
        for i, (param_name, param_var) in enumerate(circuit.tf_parameters.items()):
            if i < encoded_params.shape[1]:
                param_var.assign(encoded_params[0, i])
        
        state = circuit.execute()
        measurement = circuit.extract_measurements(state)
        loss = tf.reduce_mean(measurement)
    
    # Check gradients for input encoder
    encoder_gradients = tape.gradient(loss, input_encoder.trainable_variables)
    print(f"   Input encoder gradients: {[g.numpy() if g is not None else 0.0 for g in encoder_gradients]}")


def test_measurement_strategies():
    """Test different measurement strategies."""
    print("\n" + "=" * 80)
    print("📊 TESTING MEASUREMENT STRATEGIES")
    print("=" * 80)
    
    circuit = PureSFQuantumCircuit(n_modes=1, n_layers=2, cutoff_dim=8)
    
    # Set some parameters to get non-zero state
    for param_name, param_var in circuit.tf_parameters.items():
        param_var.assign(tf.constant(1.0))
    
    state = circuit.execute()
    
    # Test different measurement strategies
    print("Testing different measurement strategies:")
    
    # 1. X quadrature (current)
    try:
        x_quad = state.quad_expectation(0, 0)
        print(f"   X quadrature: {x_quad.numpy()}")
    except Exception as e:
        print(f"   X quadrature failed: {e}")
    
    # 2. P quadrature
    try:
        p_quad = state.quad_expectation(0, 1)
        print(f"   P quadrature: {p_quad.numpy()}")
    except Exception as e:
        print(f"   P quadrature failed: {e}")
    
    # 3. Number operator
    try:
        n_op = state.number_expectation(0)
        print(f"   Number operator: {n_op.numpy()}")
    except Exception as e:
        print(f"   Number operator failed: {e}")
    
    # 4. Variance
    try:
        x_var = state.quad_variance(0, 0)
        print(f"   X variance: {x_var.numpy()}")
    except Exception as e:
        print(f"   X variance failed: {e}")


def test_classical_baseline():
    """Test classical neural network baseline for comparison."""
    print("\n" + "=" * 80)
    print("🤖 TESTING CLASSICAL BASELINE")
    print("=" * 80)
    
    # Create classical generator with same architecture
    classical_generator = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(4, activation='tanh'),
        tf.keras.layers.Dense(1, activation='tanh')
    ])
    
    # Test generation
    test_z = tf.random.normal([100, 4])
    classical_outputs = classical_generator(test_z)
    classical_data = classical_outputs.numpy()
    
    print(f"   Classical outputs range: [{classical_data.min():.3f}, {classical_data.max():.3f}]")
    print(f"   Classical outputs mean: {classical_data.mean():.3f}, std: {classical_data.std():.3f}")
    
    # Test gradient flow
    with tf.GradientTape() as tape:
        outputs = classical_generator(test_z)
        loss = tf.reduce_mean(outputs)
    
    gradients = tape.gradient(loss, classical_generator.trainable_variables)
    grad_norms = [tf.norm(g).numpy() if g is not None else 0.0 for g in gradients]
    
    print(f"   Classical gradient norms: {grad_norms}")
    
    return classical_data


def create_improved_quantum_generator():
    """Create an improved quantum generator with better architecture."""
    print("\n" + "=" * 80)
    print("🚀 CREATING IMPROVED QUANTUM GENERATOR")
    print("=" * 80)
    
    # Use 2 modes instead of 1 for more expressiveness
    circuit = PureSFQuantumCircuit(n_modes=2, n_layers=3, cutoff_dim=8)
    
    print(f"   Improved circuit: {circuit.n_modes} modes, {circuit.n_layers} layers")
    print(f"   Parameters: {len(circuit.trainable_variables)}")
    
    # Test expressiveness
    outputs_improved = []
    for i in range(10):
        # Set random parameters
        for param_name, param_var in circuit.tf_parameters.items():
            param_shape = param_var.shape
            param_var.assign(tf.random.uniform(param_shape, -2.0, 2.0))
        
        state = circuit.execute()
        measurement = circuit.extract_measurements(state)
        outputs_improved.append(measurement.numpy())
    
    print(f"   Improved outputs shape: {np.array(outputs_improved).shape}")
    print(f"   Improved outputs range: [{np.min(outputs_improved):.3f}, {np.max(outputs_improved):.3f}]")
    print(f"   Improved outputs std: {np.std(outputs_improved):.3f}")
    
    return circuit


def visualize_debug_results(random_outputs, systematic_outputs, classical_outputs):
    """Visualize debug results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Random vs Systematic
    ax = axes[0, 0]
    ax.plot(random_outputs, 'o-', label='Random Parameters', alpha=0.7)
    ax.plot(systematic_outputs, 's-', label='Systematic Parameters', alpha=0.7)
    ax.set_title('Quantum Circuit Outputs')
    ax.set_xlabel('Test Index')
    ax.set_ylabel('Output Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Classical baseline
    ax = axes[0, 1]
    ax.hist(classical_outputs, bins=20, alpha=0.7, density=True)
    ax.set_title('Classical Generator Output Distribution')
    ax.set_xlabel('Output Value')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    
    # 3. Quantum vs Classical comparison
    ax = axes[1, 0]
    ax.boxplot([random_outputs, systematic_outputs, classical_outputs.flatten()], 
               labels=['Quantum Random', 'Quantum Systematic', 'Classical'])
    ax.set_title('Output Distribution Comparison')
    ax.set_ylabel('Output Value')
    ax.grid(True, alpha=0.3)
    
    # 4. Parameter sensitivity
    ax = axes[1, 1]
    ax.scatter(range(len(systematic_outputs)), systematic_outputs, alpha=0.7)
    ax.set_title('Parameter Sensitivity (Systematic Variation)')
    ax.set_xlabel('Parameter Index')
    ax.set_ylabel('Output Value')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = "results/debug_quantum_learning.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Debug visualization saved to: {save_path}")
    plt.show()


def main():
    """Main debug function."""
    print("🔬 QUANTUM LEARNING DEBUG SESSION")
    print("=" * 80)
    
    # Run all tests
    random_outputs, systematic_outputs = test_quantum_circuit_expressiveness()
    test_gradient_flow()
    test_measurement_strategies()
    classical_outputs = test_classical_baseline()
    improved_circuit = create_improved_quantum_generator()
    
    # Visualize results
    visualize_debug_results(random_outputs, systematic_outputs, classical_outputs)
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 DEBUG SUMMARY")
    print("=" * 80)
    
    print("Key Findings:")
    print("1. Quantum circuit expressiveness: Check output ranges above")
    print("2. Gradient flow: Check gradient values above")
    print("3. Measurement strategies: Check measurement outputs above")
    print("4. Classical baseline: Compare with quantum outputs")
    print("5. Improved architecture: 2-mode circuit with more parameters")
    
    print("\nNext Steps:")
    print("1. If quantum outputs are too narrow → Increase circuit complexity")
    print("2. If gradients are zero → Fix parameter mapping")
    print("3. If measurements are constant → Try different measurement strategies")
    print("4. If classical works better → Use hybrid approach")


if __name__ == "__main__":
    main() 