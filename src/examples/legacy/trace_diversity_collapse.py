#!/usr/bin/env python3
"""
Diversity Collapse Pipeline Tracer
=================================

This script traces the exact step where diversity collapses in the 
generator.generate() pipeline by examining each intermediate step.
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

# Import components
from src.models.generators.quantum_sf_generator import QuantumSFGenerator

def trace_generate_pipeline():
    """Trace each step of generator.generate() to find diversity collapse."""
    print("🔍 TRACING DIVERSITY COLLAPSE IN GENERATE() PIPELINE")
    print("=" * 65)
    
    # Create generator
    generator = QuantumSFGenerator(latent_dim=4, output_dim=2, n_modes=4, layers=2)
    
    # Create diverse latent inputs (same as fixed test)
    z = tf.constant([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0], 
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [-0.5, -0.5, 0.0, 0.0]
    ], dtype=tf.float32)
    
    print(f"📥 Step 1: Input Latent Vectors")
    print(f"Shape: {z.shape}")
    z_variance = tf.math.reduce_variance(z, axis=0)
    print(f"Variance per dimension: {z_variance.numpy()}")
    print(f"Max variance: {tf.reduce_max(z_variance):.6e}")
    print()
    
    # Step 2: Input encoding
    print(f"🔧 Step 2: Input Encoding")
    input_encoding = tf.matmul(z, generator.input_encoder)
    print(f"Shape: {input_encoding.shape}")
    encoding_variance = tf.math.reduce_variance(input_encoding, axis=0)
    print(f"Encoding variance (first 5): {encoding_variance[:5].numpy()}")
    print(f"Max encoding variance: {tf.reduce_max(encoding_variance):.6e}")
    print()
    
    # Step 3: Individual sample processing
    print(f"⚛️  Step 3: Individual Quantum Processing")
    batch_size = tf.shape(z)[0]
    outputs = []
    individual_variances = []
    
    for i in range(batch_size):
        # Extract individual sample
        sample_encoding = input_encoding[i:i+1]
        print(f"  Sample {i}: encoding = {sample_encoding[0][:4].numpy()} ...")
        
        # Execute quantum circuit
        state = generator.quantum_circuit.execute(sample_encoding)
        measurements = generator.quantum_circuit.extract_measurements(state)
        measurements_flat = tf.reshape(measurements, [-1])
        outputs.append(measurements_flat)
        
        # Check measurement diversity within this sample
        # The quantum circuit produces more measurements than expected
        measurement_shape = tf.shape(measurements)
        print(f"    Measurement shape: {measurement_shape.numpy()}")
        sample_variance = tf.math.reduce_variance(tf.reshape(measurements, [-1]))
        individual_variances.append(float(sample_variance.numpy()))
        
        print(f"    Measurements: {measurements_flat[:4].numpy()} ...")
        print(f"    Sample variance: {sample_variance.numpy()}")
    
    print(f"Individual sample variances summary:")
    for i, var in enumerate(individual_variances):
        print(f"  Sample {i}: {var}")
    print()
    
    # Step 4: Stack results
    print(f"📚 Step 4: Stack Individual Results")
    batch_measurements = tf.stack(outputs, axis=0)
    print(f"Batch measurements shape: {batch_measurements.shape}")
    batch_variance = tf.math.reduce_variance(batch_measurements, axis=0)
    print(f"Batch variance (first 5): {batch_variance[:5].numpy()}")
    print(f"Max batch variance: {tf.reduce_max(batch_variance):.6e}")
    print()
    
    # Step 5: Direct mapping to output
    print(f"🎯 Step 5: Direct Mapping to Output")
    output = batch_measurements[:, :generator.output_dim]
    print(f"Output shape: {output.shape}")
    output_variance = tf.math.reduce_variance(output, axis=0)
    print(f"Output variance: {output_variance.numpy()}")
    print(f"Max output variance: {tf.reduce_max(output_variance):.6e}")
    print()
    
    # Analysis
    print(f"🔍 DIVERSITY COLLAPSE ANALYSIS")
    print("=" * 50)
    
    steps = [
        ("Input latents", tf.reduce_max(z_variance)),
        ("Input encodings", tf.reduce_max(encoding_variance)),
        ("Batch measurements", tf.reduce_max(batch_variance)),
        ("Final output", tf.reduce_max(output_variance))
    ]
    
    threshold = 1e-6
    for step_name, variance in steps:
        status = "✅" if variance > threshold else "❌"
        print(f"{step_name:20s}: {variance:.6e} {status}")
    
    # Find the collapse point
    print(f"\n🎯 COLLAPSE POINT IDENTIFICATION:")
    for i in range(len(steps)-1):
        current_step, current_var = steps[i]
        next_step, next_var = steps[i+1]
        
        collapse_ratio = float(next_var / current_var) if current_var > 0 else 0
        if collapse_ratio < 0.1:  # Major drop
            print(f"💥 MAJOR COLLAPSE: {current_step} → {next_step}")
            print(f"   Variance drops by {collapse_ratio:.1%} ({current_var:.2e} → {next_var:.2e})")
    
    return output

def analyze_input_encoder_conditioning():
    """Check if input encoder has poor conditioning."""
    print(f"\n🔧 INPUT ENCODER CONDITIONING ANALYSIS")
    print("=" * 50)
    
    generator = QuantumSFGenerator(latent_dim=4, output_dim=2, n_modes=4, layers=2)
    encoder_matrix = generator.input_encoder.numpy()
    
    print(f"Encoder shape: {encoder_matrix.shape}")
    print(f"Encoder mean: {np.mean(encoder_matrix):.6e}")
    print(f"Encoder std: {np.std(encoder_matrix):.6e}")
    print(f"Encoder range: {np.max(encoder_matrix) - np.min(encoder_matrix):.6e}")
    
    # Check condition number
    condition_number = np.linalg.cond(encoder_matrix)
    print(f"Condition number: {condition_number:.2e}")
    
    if condition_number > 1e12:
        print("❌ PROBLEM: Encoder is poorly conditioned!")
    else:
        print("✅ OK: Encoder conditioning is reasonable")
    
    # Check column diversity
    col_variances = np.var(encoder_matrix, axis=0)
    print(f"Column variances (first 5): {col_variances[:5]}")
    print(f"Min column variance: {np.min(col_variances):.6e}")
    print(f"Max column variance: {np.max(col_variances):.6e}")
    
    return condition_number

def analyze_measurement_selection():
    """Check which measurements are most diverse."""
    print(f"\n📊 MEASUREMENT DIVERSITY ANALYSIS")
    print("=" * 50)
    
    generator = QuantumSFGenerator(latent_dim=4, output_dim=2, n_modes=4, layers=2)
    
    # Generate with diverse inputs
    z = tf.constant([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0], 
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0]
    ], dtype=tf.float32)
    
    input_encoding = tf.matmul(z, generator.input_encoder)
    
    # Collect all measurements
    all_measurements = []
    for i in range(tf.shape(z)[0]):
        sample_encoding = input_encoding[i:i+1]
        state = generator.quantum_circuit.execute(sample_encoding)
        measurements = generator.quantum_circuit.extract_measurements(state)
        measurements_flat = tf.reshape(measurements, [-1])
        all_measurements.append(measurements_flat.numpy())
    
    all_measurements = np.array(all_measurements)
    measurement_variances = np.var(all_measurements, axis=0)
    
    print(f"Measurement variances for each component:")
    for i, var in enumerate(measurement_variances):
        print(f"  Component {i}: {var:.6e}")
    
    # Find most diverse measurements
    sorted_indices = np.argsort(measurement_variances)[::-1]
    print(f"\nMost diverse measurements (indices): {sorted_indices[:4]}")
    print(f"Most diverse variances: {measurement_variances[sorted_indices[:4]]}")
    
    print(f"\nCurrent selection (first {generator.output_dim}): indices {list(range(generator.output_dim))}")
    print(f"Current variances: {measurement_variances[:generator.output_dim]}")
    
    return sorted_indices, measurement_variances

def main():
    """Run complete diversity collapse analysis."""
    
    # Step 1: Trace the full pipeline
    output = trace_generate_pipeline()
    
    # Step 2: Analyze encoder conditioning  
    condition_number = analyze_input_encoder_conditioning()
    
    # Step 3: Analyze measurement selection
    best_indices, variances = analyze_measurement_selection()
    
    # Summary and recommendations
    print(f"\n🎯 SUMMARY & RECOMMENDATIONS")
    print("=" * 50)
    
    if condition_number > 1e12:
        print("🔧 FIX 1: Improve input encoder conditioning")
        print("   → Make encoder trainable or use better initialization")
    
    current_variance = variances[:2].max()
    best_variance = variances[best_indices[:2]].max()
    
    if best_variance > current_variance * 10:
        print("🔧 FIX 2: Use most diverse measurements instead of first 2")
        print(f"   → Current: {current_variance:.6e}, Best possible: {best_variance:.6e}")
        print(f"   → Use indices {best_indices[:2]} instead of [0, 1]")
    
    print("\n🚀 Ready to implement fixes!")

if __name__ == "__main__":
    main()
