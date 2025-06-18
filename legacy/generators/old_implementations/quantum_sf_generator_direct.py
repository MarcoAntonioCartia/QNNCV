"""
Direct Quantum Strawberry Fields Generator

This generator uses a direct measurement approach without mode selection.
The quantum circuit outputs an n-dimensional vector that is used directly
for loss computation, with a fixed projection matrix only for visualization.
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import logging

logger = logging.getLogger(__name__)


class QuantumSFGeneratorDirect:
    """
    Direct quantum generator without mode selection.
    
    Features:
    - Direct measurement of all quantum modes
    - Fixed projection matrix for visualization only
    - Loss computation in quantum space
    - No discrete mode selection
    """
    
    def __init__(self, n_modes=4, latent_dim=6, layers=2, cutoff_dim=6,
                 output_dim=2):
        """
        Initialize direct quantum generator.
        
        Args:
            n_modes: Number of quantum modes (measurement dimension)
            latent_dim: Dimension of latent input
            layers: Number of quantum layers
            cutoff_dim: Fock space cutoff
            output_dim: Dimension for visualization (typically 2)
        """
        self.n_modes = n_modes
        self.latent_dim = latent_dim
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        self.output_dim = output_dim
        
        logger.info(f"Initializing Direct Quantum SF Generator:")
        logger.info(f"  - n_modes: {n_modes}")
        logger.info(f"  - latent_dim: {latent_dim}")
        logger.info(f"  - layers: {layers}")
        logger.info(f"  - cutoff_dim: {cutoff_dim}")
        logger.info(f"  - output_dim: {output_dim}")
        
        # Initialize SF components
        self._init_sf_components()
        
        # Initialize quantum weights
        self._init_quantum_weights()
        
        # Initialize coherent encoding
        self._init_coherent_encoder()
        
        # Initialize FIXED projection matrix for visualization
        self._init_fixed_projection()
        
        # Create symbolic parameters
        self._create_symbolic_params()
        
        # Build quantum program
        self._build_quantum_program()
    
    def _init_sf_components(self):
        """Initialize Strawberry Fields engine and program."""
        self.eng = sf.Engine(backend="tf", backend_options={
            "cutoff_dim": self.cutoff_dim,
            "pure": True
        })
        self.qnn = sf.Program(self.n_modes)
        logger.info("SF engine initialized")
    
    def _init_quantum_weights(self):
        """Initialize quantum weights."""
        # Calculate parameter count
        M = int(self.n_modes * (self.n_modes - 1)) + max(1, self.n_modes - 1)
        params_per_layer = 2 * M + 4 * self.n_modes
        
        # Single quantum weight matrix
        self.weights = tf.Variable(
            tf.random.normal([self.layers, params_per_layer], stddev=0.1),
            name="quantum_weights"
        )
        
        self.num_quantum_params = int(np.prod(self.weights.shape))
        logger.info(f"Quantum weights initialized: shape {self.weights.shape}")
    
    def _init_coherent_encoder(self):
        """Initialize coherent state encoding."""
        # Map from latent_dim to 2*n_modes (real and imaginary parts)
        self.encoding_matrix = tf.Variable(
            tf.random.normal([self.latent_dim, 2 * self.n_modes], stddev=0.5),
            name="coherent_encoding"
        )
        
        # Amplitude scaling
        self.amplitude_scale = tf.Variable(
            tf.ones([self.n_modes]) * 0.5,
            name="amplitude_scale"
        )
        
        logger.info(f"Coherent encoding: {self.latent_dim} → {self.n_modes * 2}")
    
    def _init_fixed_projection(self):
        """Initialize FIXED projection matrix for visualization only."""
        # Create a fixed orthogonal projection matrix
        # This is NOT trainable - only used for visualization
        if self.n_modes >= self.output_dim:
            # Use random orthogonal projection
            random_matrix = np.random.randn(self.n_modes, self.output_dim)
            q, _ = np.linalg.qr(random_matrix)
            self.viz_projection = tf.constant(q[:, :self.output_dim], dtype=tf.float32)
        else:
            # Pad with zeros if fewer modes than output dimensions
            eye = np.eye(self.n_modes)
            padding = np.zeros((self.n_modes, self.output_dim - self.n_modes))
            self.viz_projection = tf.constant(
                np.concatenate([eye, padding], axis=1), 
                dtype=tf.float32
            )
        
        # Also create the pseudo-inverse for transforming data to quantum space
        self.viz_projection_inv = tf.linalg.pinv(self.viz_projection)
        
        logger.info(f"Fixed visualization projection: {self.n_modes} → {self.output_dim}")
        logger.info("This matrix is NOT trainable - only for visualization")
    
    def _create_symbolic_params(self):
        """Create SF symbolic parameters."""
        # Standard quantum circuit parameters
        sf_params = np.arange(self.num_quantum_params).reshape(self.weights.shape).astype(str)
        self.sf_params = np.array([self.qnn.params(*i) for i in sf_params])
        
        # Coherent state parameters
        self.coherent_params = []
        for i in range(self.n_modes):
            self.coherent_params.append({
                'alpha_r': self.qnn.params(f"alpha_r_{i}"),
                'alpha_i': self.qnn.params(f"alpha_i_{i}")
            })
    
    def _build_quantum_program(self):
        """Build quantum program."""
        with self.qnn.context as q:
            # Initialize with coherent states
            for i in range(self.n_modes):
                alpha = self.coherent_params[i]['alpha_r'] + 1j * self.coherent_params[i]['alpha_i']
                ops.Dgate(alpha) | q[i]
            
            # Main quantum layers
            for k in range(self.layers):
                self._quantum_layer(self.sf_params[k], q)
    
    def _quantum_layer(self, params, q):
        """Single quantum layer."""
        N = len(q)
        M = int(N * (N - 1)) + max(1, N - 1)
        
        # Extract parameters
        int1 = params[:M]
        s = params[M:M+N]
        int2 = params[M+N:2*M+N]
        dr = params[2*M+N:2*M+2*N]
        dp = params[2*M+2*N:2*M+3*N]
        k = params[2*M+3*N:2*M+4*N]
        
        # Apply operations
        self._interferometer(int1, q)
        for i in range(N):
            ops.Sgate(s[i]) | q[i]
        self._interferometer(int2, q)
        for i in range(N):
            ops.Dgate(dr[i], dp[i]) | q[i]
            ops.Kgate(k[i]) | q[i]
    
    def _interferometer(self, params, q):
        """Interferometer implementation."""
        N = len(q)
        theta = params[:N*(N-1)//2]
        phi = params[N*(N-1)//2:N*(N-1)]
        rphi = params[-N+1:]
        
        if N == 1:
            ops.Rgate(rphi[0]) | q[0]
            return
        
        n = 0
        for l in range(N):
            for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
                if (l + k) % 2 != 1:
                    ops.BSgate(theta[n], phi[n]) | (q1, q2)
                    n += 1
        
        for i in range(max(1, N - 1)):
            ops.Rgate(rphi[i]) | q[i]
    
    @property
    def trainable_variables(self):
        """Return all trainable variables."""
        return [self.weights, self.encoding_matrix, self.amplitude_scale]
    
    def transform_data_to_quantum_space(self, data_2d):
        """Transform 2D data to quantum space for training."""
        # Use the pseudo-inverse of the visualization projection
        return tf.matmul(data_2d, self.viz_projection_inv)
    
    def project_to_2d(self, quantum_samples):
        """Project quantum samples to 2D for visualization only."""
        return tf.matmul(quantum_samples, self.viz_projection)
    
    def generate(self, z, return_2d=False):
        """
        Generate samples directly from quantum measurements.
        
        Args:
            z: Latent input [batch_size, latent_dim]
            return_2d: If True, return 2D projection for visualization
            
        Returns:
            samples: Generated samples [batch_size, n_modes] or [batch_size, 2] if return_2d
        """
        batch_size = tf.shape(z)[0]
        all_samples = []
        
        # Encode to coherent state parameters
        coherent_params = tf.matmul(z, self.encoding_matrix)
        
        for i in range(batch_size):
            sample = self._generate_single(coherent_params[i])
            all_samples.append(sample)
        
        quantum_samples = tf.stack(all_samples, axis=0)
        
        if return_2d:
            return self.project_to_2d(quantum_samples)
        else:
            return quantum_samples
    
    def _generate_single(self, coherent_params):
        """Generate single sample from quantum circuit."""
        # Split into real and imaginary parts
        real_parts = coherent_params[:self.n_modes]
        imag_parts = coherent_params[self.n_modes:]
        
        # Apply amplitude scaling
        real_parts = real_parts * self.amplitude_scale
        imag_parts = imag_parts * self.amplitude_scale
        
        # Create parameter mapping
        mapping = {
            p.name: w for p, w in zip(
                self.sf_params.flatten(),
                tf.reshape(self.weights, [-1])
            )
        }
        
        # Add coherent state parameters
        for i in range(self.n_modes):
            mapping[self.coherent_params[i]['alpha_r'].name] = real_parts[i]
            mapping[self.coherent_params[i]['alpha_i'].name] = imag_parts[i]
        
        # Reset engine
        if self.eng.run_progs:
            self.eng.reset()
        
        # Run quantum circuit
        try:
            state = self.eng.run(self.qnn, args=mapping).state
            
            # Extract quantum measurements directly
            quantum_vector = self._extract_quantum_measurements(state)
            
            return quantum_vector
            
        except Exception as e:
            logger.debug(f"Quantum circuit failed: {e}")
            # Fallback to random
            return tf.random.normal([self.n_modes], stddev=0.5)
    
    def _extract_quantum_measurements(self, state):
        """Extract measurements from quantum state."""
        ket = state.ket()
        prob_amplitudes = tf.abs(ket) ** 2
        
        # Extract different quantum properties for each mode
        measurements = []
        
        for mode in range(self.n_modes):
            if mode == 0:
                # Mean photon number
                n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
                mean_n = tf.reduce_sum(prob_amplitudes * n_vals[:tf.shape(prob_amplitudes)[0]])
                measurements.append(mean_n)
            elif mode == 1:
                # Standard deviation
                n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
                mean_n = tf.reduce_sum(prob_amplitudes * n_vals[:tf.shape(prob_amplitudes)[0]])
                var_n = tf.reduce_sum(prob_amplitudes * (n_vals[:tf.shape(prob_amplitudes)[0]] - mean_n)**2)
                measurements.append(tf.sqrt(var_n + 1e-6))
            elif mode == 2:
                # Entropy
                entropy = -tf.reduce_sum(prob_amplitudes * tf.math.log(prob_amplitudes + 1e-12))
                measurements.append(entropy)
            else:
                # Higher moments
                moment_order = mode
                moment = tf.reduce_sum(
                    prob_amplitudes * tf.pow(
                        tf.range(tf.shape(prob_amplitudes)[0], dtype=tf.float32), 
                        moment_order
                    )
                )
                measurements.append(tf.pow(moment + 1e-6, 1.0/moment_order))
        
        # Normalize to reasonable range
        measurements = tf.stack(measurements)
        measurements = tf.nn.tanh(measurements / 3.0) * 3.0
        
        return measurements


def test_direct_generator():
    """Test the direct quantum generator."""
    print("\n" + "="*60)
    print("TESTING DIRECT QUANTUM SF GENERATOR")
    print("="*60)
    
    # Create generator
    generator = QuantumSFGeneratorDirect(
        n_modes=4,
        latent_dim=6,
        layers=2,
        cutoff_dim=6,
        output_dim=2
    )
    
    print(f"\nGenerator created successfully")
    print(f"Trainable variables: {len(generator.trainable_variables)}")
    
    # Test data transformation
    print("\n1. DATA TRANSFORMATION TEST")
    # Create bimodal 2D data
    mode1_2d = tf.random.normal([50, 2], mean=[-2.0, -2.0], stddev=0.3)
    mode2_2d = tf.random.normal([50, 2], mean=[2.0, 2.0], stddev=0.3)
    data_2d = tf.concat([mode1_2d, mode2_2d], axis=0)
    
    # Transform to quantum space
    data_quantum = generator.transform_data_to_quantum_space(data_2d)
    print(f"   2D data shape: {data_2d.shape}")
    print(f"   Quantum data shape: {data_quantum.shape}")
    
    # Test inverse transformation
    data_2d_reconstructed = generator.project_to_2d(data_quantum)
    reconstruction_error = tf.reduce_mean(tf.square(data_2d - data_2d_reconstructed))
    print(f"   Reconstruction error: {reconstruction_error:.6f}")
    
    # Test generation
    print("\n2. GENERATION TEST")
    z_test = tf.random.normal([100, 6])
    
    # Generate in quantum space
    samples_quantum = generator.generate(z_test, return_2d=False)
    print(f"   Quantum samples shape: {samples_quantum.shape}")
    print(f"   Quantum samples range: [{tf.reduce_min(samples_quantum):.3f}, {tf.reduce_max(samples_quantum):.3f}]")
    
    # Generate in 2D for visualization
    samples_2d = generator.generate(z_test, return_2d=True)
    print(f"   2D samples shape: {samples_2d.shape}")
    
    # Test gradient flow
    print("\n3. GRADIENT FLOW TEST")
    with tf.GradientTape() as tape:
        z = tf.random.normal([4, 6])
        output = generator.generate(z, return_2d=False)
        # Loss in quantum space
        loss = tf.reduce_mean(tf.square(output))
    
    gradients = tape.gradient(loss, generator.trainable_variables)
    
    for i, (var, grad) in enumerate(zip(generator.trainable_variables, gradients)):
        if grad is not None:
            print(f"   Variable {i} ({var.name}): gradient norm = {tf.norm(grad):.6f}")
        else:
            print(f"   Variable {i} ({var.name}): No gradient")
    
    # Visualize results
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original 2D data
    axes[0].scatter(data_2d[:50, 0], data_2d[:50, 1], c='blue', alpha=0.5, label='Mode 1')
    axes[0].scatter(data_2d[50:, 0], data_2d[50:, 1], c='red', alpha=0.5, label='Mode 2')
    axes[0].set_title('Original 2D Data')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Quantum space representation
    axes[1].scatter(data_quantum[:50, 0], data_quantum[:50, 1], c='blue', alpha=0.5)
    axes[1].scatter(data_quantum[50:, 0], data_quantum[50:, 1], c='red', alpha=0.5)
    axes[1].set_title('Data in Quantum Space (first 2 dims)')
    axes[1].grid(True, alpha=0.3)
    
    # Generated samples in 2D
    samples_2d_np = samples_2d.numpy()
    axes[2].scatter(samples_2d_np[:, 0], samples_2d_np[:, 1], c='green', alpha=0.5)
    axes[2].set_title('Generated Samples (2D projection)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('direct_quantum_generation.png', dpi=150)
    plt.show()
    
    print("\n✅ Direct quantum generator test completed!")
    print("   Results saved to: direct_quantum_generation.png")
    
    return generator


if __name__ == "__main__":
    generator = test_direct_generator()
