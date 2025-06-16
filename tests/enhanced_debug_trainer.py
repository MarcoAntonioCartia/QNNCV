# enhanced_debug_trainer.py
"""
Comprehensive quantum circuit debugging for parameter and state analysis.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
import time

class QuantumCircuitDebugger:
    """Comprehensive debugger for quantum circuit analysis."""
    
    def __init__(self, generator, discriminator, latent_dim=4):
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        
        # Debugging storage
        self.debug_history = {
            'parameters': defaultdict(list),
            'states': [],
            'measurements': [],
            'gradients': defaultdict(list),
            'step_info': []
        }
        
    def debug_single_generation(self, z_sample, step_num=0):
        """Debug a single generation step with full monitoring."""
        print(f"\n🔬 DEBUGGING GENERATION STEP {step_num}")
        print("=" * 50)
        
        # Monitor all trainable variables BEFORE generation
        param_snapshot = self._capture_parameter_snapshot("before_generation")
        print(f"📊 Captured {len(param_snapshot)} parameter groups")
        
        # Generate with state monitoring
        with tf.GradientTape() as tape:
            # Override the generator's _generate_single method for state capture
            generated_sample, debug_info = self._generate_with_monitoring(z_sample)
        
        # Monitor parameters AFTER generation
        param_snapshot_after = self._capture_parameter_snapshot("after_generation")
        
        # Store debug information
        self.debug_history['step_info'].append({
            'step': step_num,
            'input_z': z_sample.numpy(),
            'output_sample': generated_sample.numpy(),
            'timing': debug_info.get('timing', {}),
            'quantum_info': debug_info.get('quantum_info', {})
        })
        
        return generated_sample, debug_info
    
    def _generate_with_monitoring(self, z_sample):
        """Generate sample with comprehensive state monitoring."""
        debug_info = {'timing': {}, 'quantum_info': {}}
        
        print(f"🎯 Input latent vector: {z_sample.numpy()}")
        
        # STEP 1: Classical encoding (if used)
        start_time = time.time()
        if hasattr(self.generator, 'encoder') and self.generator.encoder is not None:
            encoded_params = self.generator.encoder(tf.expand_dims(z_sample, 0))
            encoded_params = tf.squeeze(encoded_params, 0)
            print(f"🔧 Classical encoder output: {encoded_params.numpy()}")
        else:
            # Direct parameter mapping or quantum encoding
            encoded_params = z_sample
            print(f"🔧 Direct parameter mapping: {encoded_params.numpy()}")
        
        debug_info['timing']['encoding'] = time.time() - start_time
        
        # STEP 2: Monitor quantum circuit execution
        start_time = time.time()
        
        # Access the quantum circuit components
        if hasattr(self.generator, '_generate_single'):
            # Monkey patch to capture intermediate states
            original_extract = self.generator._extract_samples_from_state
            
            def debug_extract_samples(state):
                """Extract samples while capturing full state information."""
                print("\n🌌 QUANTUM STATE ANALYSIS")
                print("-" * 30)
                
                # Get the full quantum state
                ket = state.ket()
                ket_array = ket.numpy() if hasattr(ket, 'numpy') else ket
                
                print(f"📐 State vector shape: {ket_array.shape}")
                print(f"📊 State norm: {np.linalg.norm(ket_array):.6f}")
                
                # Analyze per-mode information
                self._analyze_quantum_state_per_mode(state, ket_array)
                
                # Call original extraction
                samples = original_extract(state)
                print(f"🎯 Extracted samples: {samples.numpy()}")
                
                # Store state information
                self.debug_history['states'].append({
                    'ket_vector': ket_array,
                    'extracted_samples': samples.numpy(),
                    'norm': float(np.linalg.norm(ket_array))
                })
                
                return samples
            
            # Temporarily replace extraction method
            self.generator._extract_samples_from_state = debug_extract_samples
            
            # Generate the sample
            try:
                generated_sample = self.generator._generate_single(encoded_params)
                print(f"✅ Generation completed: {generated_sample.numpy()}")
            finally:
                # Restore original method
                self.generator._extract_samples_from_state = original_extract
        else:
            # Fallback method
            generated_sample = self.generator.generate(tf.expand_dims(z_sample, 0))
            generated_sample = tf.squeeze(generated_sample, 0)
        
        debug_info['timing']['quantum_generation'] = time.time() - start_time
        
        return generated_sample, debug_info
    
    def _analyze_quantum_state_per_mode(self, state, ket_array):
        """Analyze quantum state information for each mode."""
        print("\n🔍 PER-MODE ANALYSIS")
        print("-" * 20)
        
        n_modes = self.generator.n_modes
        cutoff = self.generator.cutoff_dim
        
        # Calculate expectation values for each mode
        mode_info = {}
        
        for mode in range(n_modes):
            print(f"\n📡 MODE {mode}:")
            
            # Position expectation (X quadrature)
            try:
                import strawberryfields as sf
                from strawberryfields.ops import Xgate
                
                # This is complex - for now, extract basic info
                mode_info[f'mode_{mode}'] = {
                    'state_contribution': f"Complex analysis needed",
                    'note': "Full per-mode analysis requires SF state methods"
                }
                
                print(f"   State vector contribution: [Complex - requires SF analysis]")
                
            except Exception as e:
                print(f"   ⚠️ Mode analysis failed: {e}")
        
        self.debug_history['measurements'].append(mode_info)
    
    def _capture_parameter_snapshot(self, stage_name):
        """Capture all trainable parameters at current state."""
        snapshot = {}
        
        print(f"\n📸 PARAMETER SNAPSHOT - {stage_name.upper()}")
        print("-" * 40)
        
        # Generator parameters
        gen_vars = self.generator.trainable_variables
        for i, var in enumerate(gen_vars):
            var_name = f"gen_param_{i}_{var.name}" if hasattr(var, 'name') else f"gen_param_{i}"
            var_values = var.numpy()
            snapshot[var_name] = var_values.copy()
            
            print(f"🎛️  {var_name}: shape={var_values.shape}")
            print(f"     values: [{np.min(var_values):.4f}, {np.max(var_values):.4f}], std={np.std(var_values):.4f}")
            
            # Check for parameter collapse (all values similar)
            if var_values.size > 1:
                value_range = np.max(var_values) - np.min(var_values)
                if value_range < 1e-6:
                    print(f"     ⚠️  PARAMETER COLLAPSE DETECTED! Range: {value_range:.2e}")
        
        # Store in history
        self.debug_history['parameters'][stage_name].append(snapshot)
        
        return snapshot
    
    def debug_training_loop(self, real_data, epochs=5, batch_size=2):
        """Debug training loop with comprehensive monitoring."""
        print("\n" + "="*60)
        print("🚀 STARTING QUANTUM CIRCUIT DEBUGGING")
        print("="*60)
        
        # Use small subset for debugging
        debug_data = real_data[:50]  # 50 data points as requested
        
        print(f"📊 Debug data shape: {debug_data.shape}")
        print(f"📊 Debug data range: [{tf.reduce_min(debug_data):.3f}, {tf.reduce_max(debug_data):.3f}]")
        
        # Simple optimizer for debugging
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
        for epoch in range(epochs):
            print(f"\n{'='*20} EPOCH {epoch+1}/{epochs} {'='*20}")
            
            # Process small batches
            dataset = tf.data.Dataset.from_tensor_slices(debug_data).batch(batch_size)
            
            for batch_idx, real_batch in enumerate(dataset):
                print(f"\n--- Batch {batch_idx+1} ---")
                
                batch_size_actual = tf.shape(real_batch)[0]
                z_batch = tf.random.normal([batch_size_actual, self.latent_dim])
                
                # Debug each sample in the batch
                for sample_idx in range(batch_size_actual):
                    z_sample = z_batch[sample_idx]
                    step_num = epoch * len(list(dataset)) * batch_size + batch_idx * batch_size + sample_idx
                    
                    print(f"\n🎯 Sample {sample_idx+1}/{batch_size_actual} in batch")
                    
                    # Debug generation with monitoring
                    with tf.GradientTape() as tape:
                        generated_sample, debug_info = self.debug_single_generation(z_sample, step_num)
                        
                        # Simple loss for gradient monitoring
                        target_sample = real_batch[sample_idx]
                        loss = tf.reduce_mean(tf.square(generated_sample - target_sample))
                    
                    # Monitor gradients
                    gradients = tape.gradient(loss, self.generator.trainable_variables)
                    
                    print(f"📉 Loss: {loss.numpy():.6f}")
                    
                    # Analyze gradients
                    print(f"\n🌊 GRADIENT ANALYSIS")
                    print("-" * 20)
                    for i, grad in enumerate(gradients):
                        if grad is not None:
                            grad_norm = tf.norm(grad).numpy()
                            grad_mean = tf.reduce_mean(grad).numpy()
                            print(f"   Grad {i}: norm={grad_norm:.6f}, mean={grad_mean:.6f}")
                            
                            # Store gradient info
                            self.debug_history['gradients'][f'param_{i}'].append({
                                'step': step_num,
                                'norm': float(grad_norm),
                                'mean': float(grad_mean)
                            })
                        else:
                            print(f"   Grad {i}: None (no gradient)")
                    
                    # Apply gradients (optional - for debugging gradient flow)
                    # optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
                    
                    # Limit output for readability
                    if sample_idx >= 1:  # Only debug first 2 samples per batch
                        print("   ... (skipping remaining samples in batch for brevity)")
                        break
                
                # Limit batches for debugging
                if batch_idx >= 2:  # Only process first 3 batches per epoch
                    print("   ... (skipping remaining batches for brevity)")
                    break
        
        print(f"\n🎉 DEBUGGING COMPLETED!")
        self._generate_debug_report()
    
    def _generate_debug_report(self):
        """Generate comprehensive debug report."""
        print(f"\n" + "="*60)
        print("📋 QUANTUM CIRCUIT DEBUG REPORT")
        print("="*60)
        
        # Parameter evolution analysis
        print(f"\n🎛️  PARAMETER EVOLUTION ANALYSIS")
        print("-" * 40)
        
        if 'before_generation' in self.debug_history['parameters']:
            snapshots = self.debug_history['parameters']['before_generation']
            print(f"📊 Captured {len(snapshots)} parameter snapshots")
            
            # Analyze parameter stability
            if len(snapshots) > 1:
                first_snapshot = snapshots[0]
                last_snapshot = snapshots[-1]
                
                print(f"\n🔍 Parameter Change Analysis:")
                for param_name in first_snapshot:
                    if param_name in last_snapshot:
                        initial = first_snapshot[param_name]
                        final = last_snapshot[param_name]
                        change = np.linalg.norm(final - initial)
                        print(f"   {param_name}: change_norm={change:.6f}")
                        
                        if change < 1e-6:
                            print(f"      ⚠️  PARAMETER STUCK - no learning!")
        
        # State analysis
        print(f"\n🌌 QUANTUM STATE ANALYSIS")
        print("-" * 30)
        print(f"📊 Captured {len(self.debug_history['states'])} quantum states")
        
        if self.debug_history['states']:
            state_norms = [s['norm'] for s in self.debug_history['states']]
            print(f"   State norms: min={min(state_norms):.6f}, max={max(state_norms):.6f}")
            
            # Check if all extracted samples are similar
            samples = [s['extracted_samples'] for s in self.debug_history['states']]
            if len(samples) > 1:
                sample_matrix = np.array(samples)
                sample_std = np.std(sample_matrix, axis=0)
                print(f"   Sample diversity: std_per_dim={sample_std}")
                
                if np.all(sample_std < 1e-4):
                    print(f"      ❌ MODE COLLAPSE DETECTED - all samples identical!")
                else:
                    print(f"      ✅ Some sample diversity detected")
        
        # Gradient analysis
        print(f"\n🌊 GRADIENT FLOW ANALYSIS")
        print("-" * 25)
        
        for param_name, grad_history in self.debug_history['gradients'].items():
            if grad_history:
                grad_norms = [g['norm'] for g in grad_history]
                avg_norm = np.mean(grad_norms)
                print(f"   {param_name}: avg_grad_norm={avg_norm:.6f}")
                
                if avg_norm < 1e-8:
                    print(f"      ⚠️  VANISHING GRADIENTS - parameter not learning!")
                elif avg_norm > 1.0:
                    print(f"      ⚠️  EXPLODING GRADIENTS - potential instability!")

# Usage example
def run_quantum_debugging():
    """Run the quantum circuit debugging."""
    
    # Import your quantum components
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    from models.generators.quantum_sf_generator import QuantumSFGenerator
    from models.discriminators.quantum_sf_discriminator import QuantumSFDiscriminator
    
    # Create simple components for debugging
    print("🔧 Creating quantum components for debugging...")
    
    generator = QuantumSFGenerator(
        n_modes=4,        # Use your current configuration
        latent_dim=4,
        layers=2,
        cutoff_dim=8      # Keep manageable for debugging
    )
    
    discriminator = QuantumSFDiscriminator(
        n_modes=4,
        input_dim=4,
        layers=1,
        cutoff_dim=6
    )
    
    # Create debugger
    debugger = QuantumCircuitDebugger(generator, discriminator, latent_dim=4)
    
    # Create simple test data
    real_data = tf.random.normal([50, 4])  # 50 samples, 4D as requested
    
    # Run debugging
    debugger.debug_training_loop(real_data, epochs=5, batch_size=2)
    
    return debugger

if __name__ == "__main__":
    debugger = run_quantum_debugging()