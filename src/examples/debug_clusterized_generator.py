"""
Debug script for Clusterized Quantum Generator

This script helps debug why the quantum measurements are all zeros.
"""

import numpy as np
import tensorflow as tf
import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.generators.clusterized_quantum_generator import ClusterizedQuantumGenerator
from src.quantum.core.pure_sf_circuit import PureSFQuantumCircuit


def debug_quantum_circuit():
    """Debug the quantum circuit directly."""
    print("\n🔍 DEBUGGING QUANTUM CIRCUIT")
    print("="*60)
    
    # Create a simple quantum circuit
    circuit = PureSFQuantumCircuit(
        n_modes=2,
        n_layers=1,
        cutoff_dim=4,
        circuit_type="variational"
    )
    
    print(f"Circuit created with {circuit.get_parameter_count()} parameters")
    
    # Test basic execution
    print("\n1. Testing basic execution (no input)...")
    try:
        state = circuit.execute()
        print(f"   ✅ State created: {state}")
        
        measurements = circuit.extract_measurements(state)
        print(f"   Measurements: {measurements.numpy()}")
        print(f"   Measurements shape: {measurements.shape}")
        
        # Check if measurements are non-zero
        if tf.reduce_sum(tf.abs(measurements)) > 0:
            print("   ✅ Non-zero measurements!")
        else:
            print("   ❌ All measurements are zero!")
            
            # Check parameters
            print("\n   Checking circuit parameters:")
            for name, var in circuit.tf_parameters.items():
                print(f"     {name}: {var.numpy()}")
                
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with input encoding
    print("\n2. Testing with input encoding...")
    try:
        input_encoding = tf.constant([0.5, -0.3])
        state = circuit.execute(input_encoding=input_encoding)
        measurements = circuit.extract_measurements(state)
        print(f"   Measurements: {measurements.numpy()}")
        
        if tf.reduce_sum(tf.abs(measurements)) > 0:
            print("   ✅ Non-zero measurements with input!")
        else:
            print("   ❌ Still zero measurements!")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")


def debug_generator():
    """Debug the clusterized generator."""
    print("\n🔍 DEBUGGING CLUSTERIZED GENERATOR")
    print("="*60)
    
    # Create simple test data
    np.random.seed(42)
    target_data = np.array([
        [-1.5, -1.5], [-1.4, -1.6], [-1.6, -1.4],  # Cluster 1
        [1.5, 1.5], [1.4, 1.6], [1.6, 1.4]          # Cluster 2
    ])
    
    print(f"Test data shape: {target_data.shape}")
    
    # Create generator with minimal configuration
    generator = ClusterizedQuantumGenerator(
        latent_dim=2,
        output_dim=2,
        n_modes=4,
        layers=1,
        cutoff_dim=4,
        clustering_method='kmeans',
        coordinate_names=['X', 'Y']
    )
    
    print(f"Generator created with {len(generator.trainable_variables)} trainable variables")
    
    # Analyze target data
    print("\nAnalyzing target data...")
    generator.analyze_target_data(target_data)
    
    # Test generation with single sample
    print("\nTesting single sample generation...")
    z = tf.constant([[0.5, -0.3]])  # Single sample
    
    # Step through generation process
    print("\n3. Step-by-step generation:")
    
    # Step 1: Input encoding
    quantum_params = generator.input_encoder(z)
    print(f"   Input encoder output: {quantum_params.numpy()}")
    print(f"   Shape: {quantum_params.shape}")
    
    # Step 2: Execute quantum circuit
    print("\n   Executing quantum circuit...")
    sample_params = quantum_params[0]
    
    try:
        quantum_state = generator.quantum_circuit.execute(input_encoding=sample_params)
        print(f"   ✅ Quantum state created")
        
        # Step 3: Extract measurements
        measurements = generator.quantum_circuit.extract_measurements(quantum_state)
        print(f"   Raw measurements: {measurements.numpy()}")
        
        # Check circuit parameters
        print("\n   Quantum circuit parameters:")
        for i, var in enumerate(generator.quantum_circuit.trainable_variables[:5]):
            print(f"     Param {i}: {var.numpy()}")
            
    except Exception as e:
        print(f"   ❌ Quantum execution error: {e}")
        import traceback
        traceback.print_exc()
    
    # Try full generation
    print("\n4. Full generation test:")
    try:
        generated = generator.generate(z)
        print(f"   Generated sample: {generated.numpy()}")
        
        # Check last measurements
        if generator.last_measurements is not None:
            print(f"   Last measurements: {generator.last_measurements.numpy()}")
    except Exception as e:
        print(f"   ❌ Generation error: {e}")
        import traceback
        traceback.print_exc()


def test_simple_sf_circuit():
    """Test a very simple SF circuit to isolate the issue."""
    print("\n🔍 TESTING SIMPLE SF CIRCUIT")
    print("="*60)
    
    import strawberryfields as sf
    from strawberryfields import ops
    
    # Create simple 1-mode circuit
    prog = sf.Program(1)
    eng = sf.Engine("tf", backend_options={"cutoff_dim": 4})
    
    with prog.context as q:
        # Simple displacement
        ops.Dgate(0.5) | q[0]
    
    print("Simple circuit created")
    
    try:
        # Run circuit
        state = eng.run(prog).state
        
        # Get X quadrature
        x_quad = state.quad_expectation(0, 0)
        print(f"X quadrature: {x_quad}")
        
        # Get mean photon number
        n_mean = state.mean_photon(0)
        print(f"Mean photon number: {n_mean}")
        
    except Exception as e:
        print(f"❌ Simple circuit failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all debug tests."""
    print("\n" + "🐛"*30)
    print("CLUSTERIZED QUANTUM GENERATOR - DEBUG")
    print("🐛"*30)
    
    # Suppress TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    # Run debug tests
    test_simple_sf_circuit()
    debug_quantum_circuit()
    debug_generator()
    
    print("\n" + "🐛"*30)
    print("DEBUG COMPLETE")
    print("🐛"*30)


if __name__ == "__main__":
    main()
