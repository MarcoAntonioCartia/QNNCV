#!/usr/bin/env python3
"""
Quantum Diversity Debugging Script
=================================

This script performs a thorough analysis of where diversity collapse occurs
in the quantum circuit pipeline.

Tests:
1. Input encoding diversity
2. Quantum parameter diversity  
3. Quantum state diversity
4. Measurement diversity
5. Direct circuit testing

Goal: Identify exactly where the 1e-9 variance issue originates.
"""

import sys
import os
import numpy as np
import tensorflow as tf

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Enable clean output
from src.utils.warning_suppression import enable_clean_training
enable_clean_training()

# Import quantum components
from src.models.generators.quantum_sf_generator import QuantumSFGenerator
from src.quantum.core.sf_tutorial_quantum_circuit import SFTutorialQuantumCircuit

def analyze_input_encoding():
    """Test 1: Check if different latent inputs create different encodings."""
    print("🔍 Test 1: Input Encoding Diversity Analysis")
    print("=" * 60)
    
    generator = QuantumSFGenerator(latent_dim=4, output_dim=2, n_modes=4, layers=2)
    
    # Create diverse latent inputs
    z1 = tf.constant([[1.0, 0.0, 0.0, 0.0]], dtype=tf.float32)
    z2 = tf.constant([[0.0, 1.0, 0.0, 0.0]], dtype=tf.float32) 
    z3 = tf.constant([[-1.0, 0.0, 0.0, 0.0]], dtype=tf.float32)
    z4 = tf.constant([[0.0, -1.0, 0.0, 0.0]], dtype=tf.float32)
    
    # Test encoding diversity
    encoding1 = tf.matmul(z1, generator.input_encoder)
    encoding2 = tf.matmul(z2, generator.input_encoder)
    encoding3 = tf.matmul(z3, generator.input_encoder)
    encoding4 = tf.matmul(z4, generator.input_encoder)
    
    # Analyze diversity
    all_encodings = tf.stack([encoding1[0], encoding2[0], encoding3[0], encoding4[0]])
    encoding_variance = tf.math.reduce_variance(all_encodings, axis=0)
    max_encoding_variance = tf.reduce_max(encoding_variance)
    mean_encoding_variance = tf.reduce_mean(encoding_variance)
    
    print(f"Input encoding variance (max): {max_encoding_variance:.6e}")
    print(f"Input encoding variance (mean): {mean_encoding_variance:.6e}")
    print(f"Encoding 1 norm: {tf.norm(encoding1):.6f}")
    print(f"Encoding 2 norm: {tf.norm(encoding2):.6f}")
    print(f"Encoding difference norm: {tf.norm(encoding1 - encoding2):.6f}")
    
    encoding_diverse = max_encoding_variance > 1e-6
    print(f"✅ PASS: Input encodings are diverse" if encoding_diverse else "❌ FAIL: Input encodings collapse")
    
    return encoding_diverse, float(max_encoding_variance)

def analyze_quantum_parameter_diversity():
    """Test 2: Check if different inputs create different quantum parameters."""
    print("\n🔍 Test 2: Quantum Parameter Diversity Analysis")
    print("=" * 60)
    
    circuit = SFTutorialQuantumCircuit(n_modes=4, layers=2, cutoff_dim=6)
    
    # Test different input encodings
    encoding1 = tf.random.normal([1, 12], stddev=1.0, seed=42)
    encoding2 = tf.random.normal([1, 12], stddev=1.0, seed=123)
    
    # Execute with each encoding and check weight modulation
    original_weights = circuit.weights.numpy().copy()
    
    # Check weight modulation for encoding1
    state1 = circuit.execute(encoding1)
    
    # Reset weights manually and check encoding2
    circuit.weights.assign(original_weights)
    state2 = circuit.execute(encoding2)
    
    # Analyze quantum parameter diversity
    base_weights_variance = tf.math.reduce_variance(circuit.weights)
    weights_range = tf.reduce_max(circuit.weights) - tf.reduce_min(circuit.weights)
    
    print(f"Base quantum weights variance: {base_weights_variance:.6e}")
    print(f"Quantum weights range: {weights_range:.6e}")
    print(f"Quantum weights mean: {tf.reduce_mean(circuit.weights):.6e}")
    print(f"Quantum weights std: {tf.math.reduce_std(circuit.weights):.6e}")
    
    weights_diverse = base_weights_variance > 1e-8
    print(f"✅ PASS: Quantum parameters are diverse" if weights_diverse else "❌ FAIL: Quantum parameters too uniform")
    
    return weights_diverse, float(base_weights_variance)

def analyze_quantum_state_diversity():
    """Test 3: Check if different inputs create different quantum states."""
    print("\n🔍 Test 3: Quantum State Diversity Analysis")
    print("=" * 60)
    
    circuit = SFTutorialQuantumCircuit(n_modes=4, layers=2, cutoff_dim=6)
    
    # Create diverse input encodings
    encodings = []
    states = []
    
    for i in range(4):
        encoding = tf.random.normal([1, 12], stddev=1.0, seed=i*42)
        encodings.append(encoding)
        
        state = circuit.execute(encoding)
        ket = state.ket()
        states.append(ket)
    
    # Analyze state diversity
    state_norms = [tf.norm(state) for state in states]
    state_differences = []
    
    for i in range(len(states)):
        for j in range(i+1, len(states)):
            diff = tf.norm(states[i] - states[j])
            state_differences.append(float(diff))
    
    mean_state_difference = np.mean(state_differences)
    max_state_difference = np.max(state_differences)
    
    print(f"State norms: {[float(norm) for norm in state_norms]}")
    print(f"Mean state difference: {mean_state_difference:.6e}")
    print(f"Max state difference: {max_state_difference:.6e}")
    
    states_diverse = mean_state_difference > 1e-6
    print(f"✅ PASS: Quantum states are diverse" if states_diverse else "❌ FAIL: Quantum states too similar")
    
    return states_diverse, mean_state_difference

def analyze_measurement_diversity():
    """Test 4: Check if different states create different measurements."""
    print("\n🔍 Test 4: Measurement Diversity Analysis")
    print("=" * 60)
    
    circuit = SFTutorialQuantumCircuit(n_modes=4, layers=2, cutoff_dim=6)
    
    # Create diverse states and extract measurements
    measurements_list = []
    
    for i in range(4):
        # Use more extreme encoding differences
        encoding = tf.constant([[float(i), float(i*2), float(i*3), float(i*4), 
                               float(i*5), float(i*6), float(i*7), float(i*8),
                               float(i*9), float(i*10), float(i*11), float(i*12)]], dtype=tf.float32)
        
        state = circuit.execute(encoding)
        measurements = circuit.extract_measurements(state)
        measurements_list.append(measurements.numpy())
        
        print(f"Measurement {i}: {measurements.numpy()}")
    
    # Analyze measurement diversity
    all_measurements = np.array(measurements_list)
    measurement_variance = np.var(all_measurements, axis=0)
    max_measurement_variance = np.max(measurement_variance)
    mean_measurement_variance = np.mean(measurement_variance)
    
    print(f"\nMeasurement variance per component: {measurement_variance}")
    print(f"Max measurement variance: {max_measurement_variance:.6e}")
    print(f"Mean measurement variance: {mean_measurement_variance:.6e}")
    
    measurements_diverse = max_measurement_variance > 1e-6
    print(f"✅ PASS: Measurements are diverse" if measurements_diverse else "❌ FAIL: Measurements collapse")
    
    return measurements_diverse, max_measurement_variance

def analyze_parameter_scales():
    """Test 5: Analyze SF tutorial parameter initialization scales."""
    print("\n🔍 Test 5: Parameter Scale Analysis")
    print("=" * 60)
    
    circuit = SFTutorialQuantumCircuit(n_modes=4, layers=2, cutoff_dim=6)
    
    weights = circuit.weights.numpy()
    
    print(f"Weights shape: {weights.shape}")
    print(f"Weights mean: {np.mean(weights):.6e}")
    print(f"Weights std: {np.std(weights):.6e}")
    print(f"Weights min: {np.min(weights):.6e}")
    print(f"Weights max: {np.max(weights):.6e}")
    print(f"Weights range: {np.max(weights) - np.min(weights):.6e}")
    
    # Check if parameter scales are reasonable for creating diversity
    scale_adequate = np.std(weights) > 1e-6
    print(f"✅ PASS: Parameter scales adequate" if scale_adequate else "❌ FAIL: Parameter scales too small")
    
    return scale_adequate, float(np.std(weights))

def test_modulation_strength():
    """Test 6: Check if input modulation actually affects quantum parameters."""
    print("\n🔍 Test 6: Input Modulation Strength Analysis") 
    print("=" * 60)
    
    circuit = SFTutorialQuantumCircuit(n_modes=4, layers=2, cutoff_dim=6)
    
    # Test without input encoding
    state_base = circuit.execute(None)
    measurements_base = circuit.extract_measurements(state_base)
    
    # Test with strong input encoding
    strong_encoding = tf.constant([[10.0] * 12], dtype=tf.float32)  # Strong input
    state_modulated = circuit.execute(strong_encoding)
    measurements_modulated = circuit.extract_measurements(state_modulated)
    
    # Compare results
    measurement_difference = tf.norm(measurements_base - measurements_modulated)
    
    print(f"Base measurements: {measurements_base.numpy()}")
    print(f"Modulated measurements: {measurements_modulated.numpy()}")
    print(f"Measurement difference: {measurement_difference:.6e}")
    
    modulation_effective = measurement_difference > 1e-6
    print(f"✅ PASS: Input modulation is effective" if modulation_effective else "❌ FAIL: Input modulation too weak")
    
    return modulation_effective, float(measurement_difference)

def main():
    """Run comprehensive quantum diversity analysis."""
    print("🚀 QUANTUM DIVERSITY DEBUGGING ANALYSIS")
    print("=" * 70)
    print("Investigating the source of diversity collapse (variance ~1e-9)")
    print("=" * 70)
    
    results = {}
    
    # Run all tests
    results['input_encoding'] = analyze_input_encoding()
    results['quantum_parameters'] = analyze_quantum_parameter_diversity()
    results['quantum_states'] = analyze_quantum_state_diversity()
    results['measurements'] = analyze_measurement_diversity()
    results['parameter_scales'] = analyze_parameter_scales()
    results['modulation_strength'] = test_modulation_strength()
    
    # Summary analysis
    print("\n🎯 DIVERSITY COLLAPSE DIAGNOSIS SUMMARY")
    print("=" * 70)
    
    all_passed = True
    problem_areas = []
    
    for test_name, (passed, value) in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status} (value: {value:.6e})")
        
        if not passed:
            all_passed = False
            problem_areas.append(test_name)
    
    print("\n🔍 ROOT CAUSE ANALYSIS:")
    if all_passed:
        print("🤔 All tests passed - the issue might be in the batch processing or test setup")
    else:
        print(f"🎯 PROBLEM IDENTIFIED in: {', '.join(problem_areas)}")
        
        # Specific diagnostics
        if 'input_encoding' in problem_areas:
            print("   → Static input encoder not creating sufficient diversity")
        if 'quantum_parameters' in problem_areas:
            print("   → SF tutorial parameter scales too small")
        if 'quantum_states' in problem_areas:
            print("   → Quantum circuit not responding to different inputs")
        if 'measurements' in problem_areas:
            print("   → Measurement extraction not capturing quantum diversity")
        if 'parameter_scales' in problem_areas:
            print("   → Parameter initialization too conservative")
        if 'modulation_strength' in problem_areas:
            print("   → Input modulation factor too weak")
    
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    main()
