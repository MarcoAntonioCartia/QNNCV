import strawberryfields as sf
from strawberryfields.ops import *
import tensorflow as tf
import numpy as np
from contextlib import contextmanager

class QuantumContinuousGeneratorWithGradients:
    """Enhanced CV Quantum Generator with Custom Gradients.
    
    This version implements custom gradients to solve the gradient flow problem
    that occurs when using Strawberry Fields with TensorFlow's automatic differentiation.
    
    Key improvements:
    - Custom gradient implementation using parameter-shift rule
    - Warning suppression for complex number casting
    - Proper gradient flow through quantum operations
    - Batch processing support
    """
    
    def __init__(self, n_qumodes=4, latent_dim=10, cutoff_dim=10):
        """Initialize enhanced CV quantum generator with custom gradients.
        
        Args:
            n_qumodes (int): Number of quantum modes (output dimension)
            latent_dim (int): Dimension of classical latent input
            cutoff_dim (int): Fock space cutoff for simulation
        """
        self.n_qumodes = n_qumodes
        self.latent_dim = latent_dim
        self.cutoff_dim = cutoff_dim
        
        # Quantum engine with optimized settings
        self.eng = sf.Engine('tf', backend_options={
            'cutoff_dim': cutoff_dim,
            'pure': True,
            'batch_size': None  # Dynamic batch size
        })
        
        # Enhanced parameter structure
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize trainable quantum parameters with proper scaling."""
        
        # Classical-to-quantum encoding network
        self.encoding_network = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='tanh'),
            tf.keras.layers.Dense(self.n_qumodes * 2, activation='tanh'),  # [r, phi] pairs
        ])
        
        # Build the encoding network
        dummy_input = tf.zeros((1, self.latent_dim))
        _ = self.encoding_network(dummy_input)
        
        # Additional phase parameters for fine control
        self.phase_params = tf.Variable(
            tf.random.normal([self.n_qumodes], stddev=0.1),
            name="phase_parameters"
        )
    
    @property
    def trainable_variables(self):
        """Return all trainable parameters for optimizer."""
        variables = [self.phase_params]
        variables.extend(self.encoding_network.trainable_variables)
        return variables
    
    @contextmanager
    def _suppress_warnings(self):
        """Context manager to suppress complex casting warnings."""
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 
                                  message='.*casting.*complex.*float.*', 
                                  category=UserWarning)
            # Also suppress TensorFlow logging
            original_tf_log_level = tf.get_logger().level
            tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)
            try:
                yield
            finally:
                tf.get_logger().setLevel(original_tf_log_level)
    
    def _execute_quantum_circuit(self, displacement_params, phase_params):
        """Execute the quantum circuit for a single set of parameters.
        
        Args:
            displacement_params: [r1, phi1, r2, phi2, ...] for displacement gates
            phase_params: [phase1, phase2, ...] for rotation gates
            
        Returns:
            Measurement results as tensor
        """
        # Create quantum circuit
        prog = sf.Program(self.n_qumodes)
        
        # Create symbolic parameters
        displacement_r_symbols = [prog.params(f"displace_r_{i}") for i in range(self.n_qumodes)]
        displacement_phi_symbols = [prog.params(f"displace_phi_{i}") for i in range(self.n_qumodes)]
        phase_symbols = [prog.params(f"phase_{i}") for i in range(self.n_qumodes)]
        
        with prog.context as q:
            # Simple fixed interferometer (Fourier transform)
            n = self.n_qumodes
            angles = [2 * np.pi * i * j / n for i in range(n) for j in range(n)]
            dft_matrix = np.array([[np.exp(1j * angles[i*n + j]) / np.sqrt(n) 
                                 for j in range(n)] for i in range(n)])
            
            Interferometer(dft_matrix, mesh='rectangular') | q
            
            # Displacement gates (classical data encoding)
            for i in range(self.n_qumodes):
                Dgate(displacement_r_symbols[i], displacement_phi_symbols[i]) | q[i]
            
            # Additional phase rotations for expressivity
            for i in range(self.n_qumodes):
                Rgate(phase_symbols[i]) | q[i]
            
            # Homodyne measurements
            for i in range(self.n_qumodes):
                MeasureHomodyne(0.0) | q[i]  # X-quadrature measurement
        
        # Create parameter mapping
        param_mapping = {
            **{f"displace_r_{j}": displacement_params[j] for j in range(self.n_qumodes)},
            **{f"displace_phi_{j}": displacement_params[self.n_qumodes + j] for j in range(self.n_qumodes)},
            **{f"phase_{j}": phase_params[j] for j in range(self.n_qumodes)}
        }
        
        # Reset engine and run circuit
        if self.eng.run_progs:
            self.eng.reset()
        
        with self._suppress_warnings():
            result = self.eng.run(prog, args=param_mapping)
            samples = tf.cast(result.samples, tf.float32)
            
        # Ensure samples are 1D [n_qumodes]
        if len(samples.shape) > 1:
            samples = tf.squeeze(samples, axis=0)
            
        return samples
    
    def _compute_parameter_shift_gradients(self, z_single, dy, shift=0.1):
        """Compute gradients using parameter-shift rule.
        
        Args:
            z_single: Single latent vector
            dy: Upstream gradients
            shift: Parameter shift amount
            
        Returns:
            Gradients for all trainable parameters
        """
        # Get current parameters
        displacement_params = self.encoding_network(tf.expand_dims(z_single, 0))[0]
        current_phase_params = self.phase_params
        
        # Compute gradients for encoding network parameters
        encoding_gradients = []
        for var in self.encoding_network.trainable_variables:
            var_gradients = []
            var_flat = tf.reshape(var, [-1])
            
            for i in range(tf.size(var_flat)):
                # Create shifted variables
                var_plus = tf.identity(var_flat)
                var_minus = tf.identity(var_flat)
                
                # Apply shifts
                var_plus = tf.tensor_scatter_nd_update(var_plus, [[i]], [var_flat[i] + shift])
                var_minus = tf.tensor_scatter_nd_update(var_minus, [[i]], [var_flat[i] - shift])
                
                # Temporarily update the variable and compute outputs
                original_value = var.assign(tf.reshape(var_plus, var.shape))
                displacement_params_plus = self.encoding_network(tf.expand_dims(z_single, 0))[0]
                output_plus = self._execute_quantum_circuit(displacement_params_plus, current_phase_params)
                
                var.assign(tf.reshape(var_minus, var.shape))
                displacement_params_minus = self.encoding_network(tf.expand_dims(z_single, 0))[0]
                output_minus = self._execute_quantum_circuit(displacement_params_minus, current_phase_params)
                
                # Restore original value
                var.assign(original_value)
                
                # Compute gradient using parameter-shift rule
                grad = tf.reduce_sum((output_plus - output_minus) * dy) / (2.0 * shift)
                var_gradients.append(grad)
            
            encoding_gradients.append(tf.reshape(tf.stack(var_gradients), var.shape))
        
        # Compute gradients for phase parameters
        phase_gradients = []
        for i in range(self.n_qumodes):
            # Create shifted phase parameters
            phase_plus = tf.identity(current_phase_params)
            phase_minus = tf.identity(current_phase_params)
            
            phase_plus = tf.tensor_scatter_nd_update(phase_plus, [[i]], [current_phase_params[i] + shift])
            phase_minus = tf.tensor_scatter_nd_update(phase_minus, [[i]], [current_phase_params[i] - shift])
            
            # Compute outputs with shifted parameters
            output_plus = self._execute_quantum_circuit(displacement_params, phase_plus)
            output_minus = self._execute_quantum_circuit(displacement_params, phase_minus)
            
            # Compute gradient
            grad = tf.reduce_sum((output_plus - output_minus) * dy) / (2.0 * shift)
            phase_gradients.append(grad)
        
        phase_grad_tensor = tf.stack(phase_gradients)
        
        return [phase_grad_tensor] + encoding_gradients
    
    @tf.custom_gradient
    def _generate_with_custom_gradients(self, z):
        """Generate samples with custom gradient computation.
        
        Args:
            z: Latent noise vector [batch_size, latent_dim]
            
        Returns:
            Generated samples with custom gradients
        """
        batch_size = tf.shape(z)[0]
        
        # Forward pass: generate samples for each item in batch
        all_samples = []
        
        for i in range(batch_size):
            z_single = z[i]
            
            # Classical preprocessing
            displacement_params = self.encoding_network(tf.expand_dims(z_single, 0))[0]
            
            # Execute quantum circuit
            samples = self._execute_quantum_circuit(displacement_params, self.phase_params)
            all_samples.append(samples)
        
        # Stack samples into batch
        output = tf.stack(all_samples, axis=0)
        
        def grad_fn(dy):
            """Custom gradient function using parameter-shift rule."""
            # Compute gradients for each sample in the batch
            all_gradients = [[] for _ in range(len(self.trainable_variables))]
            
            for i in range(batch_size):
                z_single = z[i]
                dy_single = dy[i]
                
                # Compute gradients for this sample
                sample_gradients = self._compute_parameter_shift_gradients(z_single, dy_single)
                
                # Accumulate gradients
                for j, grad in enumerate(sample_gradients):
                    all_gradients[j].append(grad)
            
            # Average gradients across batch
            final_gradients = []
            for grad_list in all_gradients:
                if grad_list:
                    avg_grad = tf.reduce_mean(tf.stack(grad_list), axis=0)
                    final_gradients.append(avg_grad)
                else:
                    final_gradients.append(None)
            
            # Return gradients for z (None) and all trainable variables
            return [None] + final_gradients
        
        return output, grad_fn
    
    def generate(self, z):
        """Generate quantum samples from latent vector with proper gradients.
        
        Args:
            z (tensor): Latent noise vector [batch_size, latent_dim]
            
        Returns:
            samples (tensor): Generated samples [batch_size, n_qumodes]
        """
        return self._generate_with_custom_gradients(z)
    
    def generate_single(self, z_single):
        """Generate a single sample (useful for debugging)."""
        z_batch = tf.expand_dims(z_single, 0)
        result = self.generate(z_batch)
        return tf.squeeze(result, 0)


def test_gradient_flow():
    """Test that gradients flow properly through the quantum generator."""
    print("Testing Quantum Generator with Custom Gradients")
    print("=" * 50)
    
    # Create generator
    generator = QuantumContinuousGeneratorWithGradients(n_qumodes=3, latent_dim=4, cutoff_dim=6)
    
    # Test forward pass
    z_test = tf.random.normal([2, 4])
    
    print("Testing forward pass...")
    with tf.GradientTape() as tape:
        tape.watch(generator.trainable_variables)
        samples = generator.generate(z_test)
        loss = tf.reduce_mean(tf.square(samples))
    
    print(f"Generated samples shape: {samples.shape}")
    print(f"Loss: {loss:.4f}")
    
    # Test gradient computation
    print("\nTesting gradient computation...")
    gradients = tape.gradient(loss, generator.trainable_variables)
    
    gradient_norms = []
    for i, grad in enumerate(gradients):
        if grad is not None:
            norm = tf.norm(grad)
            gradient_norms.append(float(norm))
            print(f"Gradient {i} norm: {norm:.6f}")
        else:
            print(f"Gradient {i}: None")
            gradient_norms.append(0.0)
    
    # Check if gradients are flowing
    total_grad_norm = sum(gradient_norms)
    if total_grad_norm > 1e-8:
        print(f"\n SUCCESS :( : Gradients are flowing! Total norm: {total_grad_norm:.6f}")
        return True
    else:
        print(f"\n FAILURE :( : No gradients detected. Total norm: {total_grad_norm:.6f}")
        return False


if __name__ == "__main__":
    test_gradient_flow()
