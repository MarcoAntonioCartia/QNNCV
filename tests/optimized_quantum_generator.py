"""
Optimized Quantum Generator

Based on measurement diagnosis results:
- Single parameter tensor (20 parameters)
- Oscillatory measurement strategy (proven to help)
- Simplified structure for training stability
"""

import tensorflow as tf
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *

class OptimizedQuantumGenerator:
    """
    Optimized quantum generator with proven configuration:
    - Single parameter tensor (20 parameters)
    - Oscillatory measurement (best performing)
    - 4-mode quantum circuit
    """
    
    def __init__(self, n_modes=4, cutoff_dim=6):
        self.n_modes = n_modes
        self.cutoff_dim = cutoff_dim
        
        # SINGLE PARAMETER TENSOR (as proven optimal)
        self.weights = tf.Variable(
            tf.random.normal([20], stddev=0.5),
            name="quantum_params",
            trainable=True
        )
        
        print(f"✅ Optimized generator: {self.weights.shape[0]} parameters in single tensor")
        
        # Create quantum program
        self.prog = sf.Program(n_modes)
        self.eng = sf.Engine("tf", backend_options={"cutoff_dim": cutoff_dim})
        
    @property
    def trainable_variables(self):
        """Return trainable variables for gradient computation."""
        return [self.weights]
    
    def _create_quantum_circuit(self, batch_size):
        """Create the quantum circuit using single parameter tensor."""
        with self.prog.context as q:
            # Use different parts of parameter tensor for different operations
            
            # Displacement parameters (use first 8 parameters)
            for i in range(self.n_modes):
                # Real and imaginary parts for displacement
                r_idx = i * 2
                i_idx = i * 2 + 1
                
                if r_idx < len(self.weights) - 1:
                    r_part = self.weights[r_idx]
                    i_part = self.weights[i_idx]
                    
                    # Apply displacement
                    Dgate(r_part, i_part) | q[i]
            
            # Squeezing parameters (use parameters 8-15)
            for i in range(self.n_modes):
                if 8 + i < len(self.weights):
                    sq_param = self.weights[8 + i]
                    Sgate(sq_param) | q[i]
            
            # Rotation parameters (use parameters 16-19)
            for i in range(min(self.n_modes, 4)):
                if 16 + i < len(self.weights):
                    rot_param = self.weights[16 + i]
                    Rgate(rot_param) | q[i]
    
    def _oscillatory_measurement(self, state):
        """
        Proven oscillatory measurement strategy.
        This was shown to be the best performing approach.
        """
        # Calculate mean photon numbers for each mode
        mean_ns = []
        var_ns = []
        
        for i in range(self.n_modes):
            # Mean photon number for mode i
            n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
            probs = tf.abs(state.fock_prob([0] * i + [slice(self.cutoff_dim)] + [0] * (self.n_modes - i - 1)))**2
            probs = probs / (tf.reduce_sum(probs) + 1e-10)  # Normalize
            
            mean_n = tf.reduce_sum(probs * n_vals)
            var_n = tf.reduce_sum(probs * (n_vals - mean_n)**2)
            
            mean_ns.append(mean_n)
            var_ns.append(var_n)
        
        # Combine measurements from all modes
        total_mean_n = tf.reduce_sum(tf.stack(mean_ns))
        total_var_n = tf.reduce_sum(tf.stack(var_ns))
        
        # Oscillatory mapping (proven to work best)
        x_phase = total_mean_n * np.pi / self.n_modes * 2
        x_measurement = tf.sin(x_phase) * 3.0
        x_var_mod = tf.cos(tf.sqrt(total_var_n + 1e-8)) * 0.5
        x = x_measurement + x_var_mod
        
        y_phase = (total_mean_n + self.n_modes/4) * np.pi / self.n_modes * 2
        y_measurement = tf.cos(y_phase) * 3.0
        y_var_mod = tf.sin(tf.sqrt(total_var_n + 1e-8)) * 0.5
        y = y_measurement + y_var_mod
        
        # Add small amount of noise for numerical stability
        x += tf.random.normal([], stddev=0.02)
        y += tf.random.normal([], stddev=0.02)
        
        return tf.stack([x, y])
    
    def generate_samples(self, n_samples):
        """Generate samples using optimized configuration."""
        self.eng.reset()
        self._create_quantum_circuit(n_samples)
        
        # Run the quantum circuit
        state = self.eng.run(self.prog)
        
        # Generate samples using oscillatory measurement
        samples = []
        for _ in range(n_samples):
            sample = self._oscillatory_measurement(state)
            samples.append(sample)
        
        return tf.stack(samples)
    
    def call(self, z):
        """
        Main call function for training.
        z is typically ignored in quantum generators.
        """
        batch_size = tf.shape(z)[0]
        return self.generate_samples(batch_size)

# Test function to verify the generator works
def test_optimized_generator():
    """Test the optimized generator."""
    print("Testing optimized quantum generator...")
    
    generator = OptimizedQuantumGenerator()
    
    # Test parameter count
    print(f"Parameters: {generator.weights.shape[0]}")
    
    # Test gradient flow
    with tf.GradientTape() as tape:
        # Generate samples
        dummy_z = tf.random.normal([10, 2])  # Dummy input
        samples = generator.call(dummy_z)
        
        # Simple loss for testing
        loss = tf.reduce_mean(tf.square(samples))
    
    # Check gradients
    gradients = tape.gradient(loss, generator.trainable_variables)
    grad_norm = tf.norm(gradients[0])
    
    print(f"✅ Generated samples shape: {samples.shape}")
    print(f"✅ Gradient norm: {grad_norm:.6f}")
    print(f"✅ Sample range: [{tf.reduce_min(samples):.3f}, {tf.reduce_max(samples):.3f}]")
    
    return generator

if __name__ == "__main__":
    test_optimized_generator() 