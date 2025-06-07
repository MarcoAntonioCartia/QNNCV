"""
True Quantum Generator using Strawberry Fields.
This implementation uses real quantum computing principles for data generation.
"""

import numpy as np
import tensorflow as tf

# Strawberry Fields imports with fallback
try:
    import strawberryfields as sf
    from strawberryfields.ops import *
    SF_AVAILABLE = True
    print("✓ Strawberry Fields available - using true quantum implementation")
except ImportError:
    SF_AVAILABLE = False
    print("⚠ Strawberry Fields not available - falling back to quantum-inspired implementation")

class QuantumGeneratorStrawberryFields:
    """
    True quantum generator using Strawberry Fields continuous variable quantum computing.
    
    This implementation uses real quantum optics to generate data through:
    - Squeezed states for parameter encoding
    - Beam splitters for entanglement
    - Rotation gates for parameter modulation
    - Displacement operations for output generation
    """
    
    def __init__(self, n_qumodes=4, latent_dim=10, output_dim=2, cutoff_dim=10):
        """
        Initialize the quantum generator.
        
        Args:
            n_qumodes (int): Number of quantum modes (qubits)
            latent_dim (int): Dimensionality of input noise
            output_dim (int): Dimensionality of output data
            cutoff_dim (int): Fock space cutoff for simulation
        """
        self.n_qumodes = n_qumodes
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.cutoff_dim = cutoff_dim
        
        if SF_AVAILABLE:
            # Initialize Strawberry Fields engine
            self.engine = sf.Engine("tf", backend_options={"cutoff_dim": cutoff_dim})
            self.prog = sf.Program(n_qumodes)
            
            # Quantum parameters - these will be optimized during training
            self.squeeze_params = tf.Variable(
                tf.random.uniform([n_qumodes], -0.5, 0.5),
                name="squeeze_params",
                trainable=True
            )
            
            self.rotation_params = tf.Variable(
                tf.random.uniform([n_qumodes], 0, 2*np.pi),
                name="rotation_params", 
                trainable=True
            )
            
            self.displacement_params = tf.Variable(
                tf.random.uniform([n_qumodes, 2], -1, 1),
                name="displacement_params",
                trainable=True
            )
            
            # Classical post-processing network
            self.post_process = tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation='tanh'),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(output_dim, activation='tanh')
            ])
            
        else:
            # Fallback to quantum-inspired classical implementation
            self._init_classical_fallback()
    
    def _init_classical_fallback(self):
        """Initialize classical fallback when Strawberry Fields is not available."""
        # Quantum-inspired parameters
        self.quantum_params = tf.Variable(
            tf.random.normal([self.n_qumodes, 3]),  # [squeeze, rotation, displacement]
            name="quantum_params",
            trainable=True
        )
        
        # Enhanced classical network with quantum-inspired structure
        self.quantum_inspired_network = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(32, activation=lambda x: tf.sin(x)),  # Oscillatory
            tf.keras.layers.Dense(16, activation='tanh'),
            tf.keras.layers.Dense(self.output_dim, activation='tanh')
        ])
    
    @property
    def trainable_variables(self):
        """Return all trainable variables."""
        if SF_AVAILABLE:
            variables = [self.squeeze_params, self.rotation_params, self.displacement_params]
            variables.extend(self.post_process.trainable_variables)
        else:
            variables = [self.quantum_params]
            variables.extend(self.quantum_inspired_network.trainable_variables)
        return variables
    
    def _build_quantum_circuit(self, input_params):
        """
        Build the quantum circuit for data generation.
        
        Args:
            input_params: Input parameters from the latent space
            
        Returns:
            Strawberry Fields program
        """
        with self.prog.context as q:
            # Step 1: Initialize states with squeezing (entanglement preparation)
            for i in range(self.n_qumodes):
                # Squeeze each mode - this creates quantum correlations
                Sgate(self.squeeze_params[i]) | q[i]
            
            # Step 2: Create entanglement through beam splitters
            for i in range(self.n_qumodes - 1):
                # Beam splitter creates entanglement between adjacent modes
                theta = input_params[i % self.latent_dim] * np.pi
                BSgate(theta, 0) | (q[i], q[(i + 1) % self.n_qumodes])
            
            # Step 3: Apply rotations based on input parameters
            for i in range(self.n_qumodes):
                param_idx = i % self.latent_dim
                rotation_angle = self.rotation_params[i] + input_params[param_idx]
                Rgate(rotation_angle) | q[i]
            
            # Step 4: Apply displacement operations
            for i in range(self.n_qumodes):
                # Displacement amplitude modulated by input
                param_idx = i % self.latent_dim
                alpha_real = self.displacement_params[i, 0] + 0.1 * input_params[param_idx]
                alpha_imag = self.displacement_params[i, 1] + 0.1 * input_params[param_idx + 1] if param_idx + 1 < self.latent_dim else 0
                Dgate(alpha_real, alpha_imag) | q[i]
        
        return self.prog
    
    def _quantum_generate_single(self, z_sample):
        """
        Generate a single sample using quantum circuit.
        
        Args:
            z_sample: Single sample from latent space
            
        Returns:
            Generated quantum state measurements
        """
        # Build quantum circuit with input parameters
        prog = self._build_quantum_circuit(z_sample)
        
        # Run the quantum circuit
        result = self.engine.run(prog)
        
        # Extract quantum measurements
        # Use quadrature measurements (position and momentum)
        measurements = []
        for i in range(min(self.n_qumodes, self.output_dim * 2)):
            # Quadrature measurement
            x_quad = result.state.quad_expectation(i, 0)  # Position quadrature
            p_quad = result.state.quad_expectation(i, 1)  # Momentum quadrature
            measurements.extend([x_quad, p_quad])
        
        # Trim to desired output dimension
        measurements = measurements[:self.output_dim]
        
        return tf.constant(measurements, dtype=tf.float32)
    
    def _quantum_generate_batch(self, z_batch):
        """
        Generate a batch of samples using quantum circuits.
        
        Args:
            z_batch: Batch of latent samples
            
        Returns:
            Batch of generated samples
        """
        batch_size = tf.shape(z_batch)[0]
        generated_samples = []
        
        # Process each sample in the batch
        for i in range(batch_size):
            sample = self._quantum_generate_single(z_batch[i])
            generated_samples.append(sample)
        
        return tf.stack(generated_samples)
    
    def _classical_fallback_generate(self, z):
        """
        Classical fallback generation when Strawberry Fields is not available.
        
        Args:
            z: Latent space samples
            
        Returns:
            Generated samples using quantum-inspired classical computation
        """
        # Process through quantum-inspired network
        base_output = self.quantum_inspired_network(z)
        
        # Add quantum interference patterns
        interference_patterns = []
        for i in range(self.n_qumodes):
            # Simulate quantum interference using trigonometric functions
            squeeze_effect = tf.sin(base_output * self.quantum_params[i, 0])
            rotation_effect = tf.cos(base_output * self.quantum_params[i, 1])
            displacement_effect = tf.sin(base_output + self.quantum_params[i, 2])
            
            combined_effect = squeeze_effect + rotation_effect + displacement_effect
            interference_patterns.append(combined_effect)
        
        # Combine interference patterns
        if interference_patterns:
            interference = tf.reduce_mean(tf.stack(interference_patterns), axis=0)
            # Add quantum-like correlations
            output = base_output + 0.1 * interference
        else:
            output = base_output
        
        return output
    
    def generate(self, z):
        """
        Main generation method.
        
        Args:
            z: Input latent samples [batch_size, latent_dim]
            
        Returns:
            Generated samples [batch_size, output_dim]
        """
        if SF_AVAILABLE:
            try:
                # Use true quantum generation
                return self._quantum_generate_batch(z)
            except Exception as e:
                print(f"⚠ Quantum generation failed: {e}")
                print("Falling back to classical implementation")
                return self._classical_fallback_generate(z)
        else:
            # Use quantum-inspired classical generation
            return self._classical_fallback_generate(z)
    
    def get_quantum_state_info(self):
        """
        Get information about the current quantum state.
        
        Returns:
            dict: Information about quantum parameters and state
        """
        info = {
            'n_qumodes': self.n_qumodes,
            'cutoff_dim': self.cutoff_dim,
            'strawberry_fields_available': SF_AVAILABLE
        }
        
        if SF_AVAILABLE:
            info.update({
                'squeeze_params': self.squeeze_params.numpy(),
                'rotation_params': self.rotation_params.numpy(),
                'displacement_params': self.displacement_params.numpy()
            })
        else:
            info.update({
                'quantum_params': self.quantum_params.numpy(),
                'mode': 'classical_fallback'
            })
        
        return info

# Example usage and testing
if __name__ == "__main__":
    print("Testing Quantum Generator with Strawberry Fields")
    print("=" * 50)
    
    # Initialize generator
    generator = QuantumGeneratorStrawberryFields(
        n_qumodes=4,
        latent_dim=6,
        output_dim=2,
        cutoff_dim=8
    )
    
    # Test generation
    test_z = tf.random.normal([10, 6])
    generated_samples = generator.generate(test_z)
    
    print(f"Generated samples shape: {generated_samples.shape}")
    print(f"Trainable variables: {len(generator.trainable_variables)}")
    
    # Print quantum state info
    state_info = generator.get_quantum_state_info()
    print(f"Quantum state info: {state_info}")
    
    print("✅ Quantum generator test completed!") 