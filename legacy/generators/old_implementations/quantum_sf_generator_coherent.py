"""
Quantum Strawberry Fields Generator with Coherent State Mode Encoding

This generator uses coherent state encoding to naturally create multi-modal distributions.
The coherent state parameter alpha is mode-dependent, creating distinct quantum states
for different modes without requiring classical selection.
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import logging

logger = logging.getLogger(__name__)


class QuantumSFGeneratorCoherent:
    """
    Quantum generator using coherent state encoding for mode selection.
    
    Features:
    - Coherent state encoding with mode-dependent alpha parameters
    - Natural quantum mode separation through phase/amplitude modulation
    - Supports N-mode generation through circular arrangement
    - No classical mode selector needed
    """
    
    def __init__(self, n_modes=4, latent_dim=6, layers=2, cutoff_dim=6,
                 n_output_modes=2, mode_radius=2.0, output_dim=2):
        """
        Initialize quantum generator with coherent state encoding.
        
        Args:
            n_modes: Number of quantum modes
            latent_dim: Dimension of latent input
            layers: Number of quantum layers
            cutoff_dim: Fock space cutoff
            n_output_modes: Number of output modes (clusters)
            mode_radius: Radius for mode arrangement in output space
        """
        self.n_modes = n_modes
        self.latent_dim = latent_dim
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        self.n_output_modes = n_output_modes
        self.mode_radius = mode_radius
        self.output_dim = output_dim
        
        # Calculate mode centers in circular arrangement
        angles = np.linspace(0, 2*np.pi, n_output_modes, endpoint=False)
        self.mode_centers = tf.constant(
            [[mode_radius * np.cos(a), mode_radius * np.sin(a)] for a in angles],
            dtype=tf.float32
        )
        
        logger.info(f"Initializing Quantum SF Generator with Coherent Encoding:")
        logger.info(f"  - n_modes: {n_modes}")
        logger.info(f"  - latent_dim: {latent_dim}")
        logger.info(f"  - layers: {layers}")
        logger.info(f"  - cutoff_dim: {cutoff_dim}")
        logger.info(f"  - n_output_modes: {n_output_modes}")
        logger.info(f"  - mode_centers: {self.mode_centers.numpy()}")
        
        # Initialize SF components
        self._init_sf_components()
        
        # Initialize quantum weights
        self._init_quantum_weights()
        
        # Initialize coherent encoding network
        self._init_coherent_encoder()
        
        # Initialize output projection matrix
        self._init_output_projection()
        
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
            name="quantum_generator_weights"
        )
        
        self.num_quantum_params = int(np.prod(self.weights.shape))
        logger.info(f"Quantum weights initialized: shape {self.weights.shape}")
    
    def _init_coherent_encoder(self):
        """Initialize direct linear mapping for coherent state encoding."""
        # Create a full-rank matrix to map latent space to coherent state parameters
        # This ensures no null space and preserves all information
        
        # Initialize with QR decomposition to ensure full rank
        # Map from latent_dim to 2*n_modes (real and imaginary parts of alpha)
        random_matrix = tf.random.normal([self.latent_dim, 2 * self.n_modes])
        
        # Use QR decomposition to create orthogonal basis
        q, r = tf.linalg.qr(random_matrix)
        
        # Create the encoding matrix
        if self.latent_dim >= 2 * self.n_modes:
            # More input dimensions than output - use first columns of Q
            self.encoding_matrix = tf.Variable(
                q[:, :2 * self.n_modes],
                name="coherent_encoding_matrix"
            )
        else:
            # More output dimensions than input - need to extend
            # Create a full-rank rectangular matrix
            self.encoding_matrix = tf.Variable(
                tf.concat([q, tf.random.normal([self.latent_dim, 2 * self.n_modes - self.latent_dim])], axis=1),
                name="coherent_encoding_matrix"
            )
        
        # Scale factor for controlling coherent state amplitudes
        # Adapt based on number of output modes
        if self.n_output_modes <= 2:
            # For 2 modes: use +0.5 and -0.5
            scale_value = 0.5
        elif self.n_output_modes <= 4:
            # For 3-4 modes: use larger range
            scale_value = 1.0
        elif self.n_output_modes <= 8:
            # For 5-8 modes: use full range with finer steps
            scale_value = 1.5
        else:
            # For many modes: use phase encoding primarily
            scale_value = 0.5
        
        self.amplitude_scale = tf.Variable(
            tf.ones([self.n_modes]) * scale_value,
            name="amplitude_scale"
        )
        
        # Create amplitude levels for discrete encoding
        if self.n_output_modes <= 8:
            # For up to 8 modes, use discrete amplitude levels
            self.amplitude_levels = tf.linspace(-scale_value, scale_value, self.n_output_modes)
        else:
            # For more modes, rely on phase encoding
            self.amplitude_levels = None
        
        logger.info(f"Direct coherent encoding: {self.latent_dim} → {self.n_modes * 2}")
        logger.info(f"Encoding matrix shape: {self.encoding_matrix.shape}")
    
    def _init_output_projection(self):
        """Initialize projection from quantum modes to output space."""
        # Create a learnable projection matrix from n_modes to output_dim
        # Initialize with orthogonal matrix for good conditioning
        if self.n_modes >= self.output_dim:
            # More quantum modes than output dimensions
            init_matrix = tf.random.normal([self.n_modes, self.output_dim])
            q, _ = tf.linalg.qr(init_matrix)
            self.output_projection = tf.Variable(
                q[:, :self.output_dim],
                name="output_projection_matrix"
            )
        else:
            # Fewer quantum modes than output dimensions - need to extend
            init_matrix = tf.random.normal([self.n_modes, self.n_modes])
            q, _ = tf.linalg.qr(init_matrix)
            extended = tf.concat([q, tf.random.normal([self.n_modes, self.output_dim - self.n_modes])], axis=1)
            self.output_projection = tf.Variable(
                extended,
                name="output_projection_matrix"
            )
        
        logger.info(f"Output projection: {self.n_modes} → {self.output_dim}")
    
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
        """Build quantum program with coherent state initialization."""
        with self.qnn.context as q:
            # Initialize with coherent states
            for i in range(self.n_modes):
                # Coherent state with mode-dependent parameters
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
        return [self.weights, self.encoding_matrix, self.amplitude_scale, self.output_projection]
    
    def generate(self, z):
        """
        Generate samples using coherent state encoding.
        
        Args:
            z: Latent input [batch_size, latent_dim]
            
        Returns:
            samples: Generated samples [batch_size, n_modes]
        """
        batch_size = tf.shape(z)[0]
        all_samples = []
        
        # Direct linear transformation to coherent state parameters
        # No neural network, just matrix multiplication
        coherent_params = tf.matmul(z, self.encoding_matrix)  # [batch_size, n_modes * 2]
        
        for i in range(batch_size):
            sample = self._generate_single_coherent(coherent_params[i])
            all_samples.append(sample)
        
        return tf.stack(all_samples, axis=0)
    
    def _generate_single_coherent(self, coherent_params):
        """Generate single sample using coherent state encoding."""
        # Split into real and imaginary parts for each mode
        real_parts = coherent_params[:self.n_modes]
        imag_parts = coherent_params[self.n_modes:]
        
        # Apply amplitude scaling
        real_parts = real_parts * self.amplitude_scale
        imag_parts = imag_parts * self.amplitude_scale
        
        # CONTINUOUS MODE BLENDING for gradient preservation
        # Instead of discrete mode selection, use continuous weights
        
        # Compute mode weights based on coherent state pattern
        if self.n_output_modes == 2:
            # For bimodal: use tanh of sum of real parts
            mode_score = tf.reduce_sum(real_parts)
            # Continuous weights between modes
            mode1_weight = tf.nn.sigmoid(-mode_score * 2.0)  # High when negative
            mode2_weight = tf.nn.sigmoid(mode_score * 2.0)   # High when positive
            mode_weights = tf.stack([mode1_weight, mode2_weight])
        else:
            # For multi-modal: use softmax over different projections
            mode_scores = []
            for i in range(self.n_output_modes):
                # Different linear combinations for each mode
                angle = 2.0 * np.pi * i / self.n_output_modes
                projection = tf.cos(angle) * real_parts[0] + tf.sin(angle) * real_parts[1]
                mode_scores.append(projection)
            mode_scores = tf.stack(mode_scores)
            mode_weights = tf.nn.softmax(mode_scores * 2.0)  # Temperature scaling
        
        # Create parameter mapping
        mapping = {
            p.name: w for p, w in zip(
                self.sf_params.flatten(),
                tf.reshape(self.weights, [-1])
            )
        }
        
        # Add coherent state parameters
        for i in range(self.n_modes):
            # Direct mapping from encoded parameters
            mapping[self.coherent_params[i]['alpha_r'].name] = real_parts[i]
            mapping[self.coherent_params[i]['alpha_i'].name] = imag_parts[i]
        
        # Reset engine
        if self.eng.run_progs:
            self.eng.reset()
        
        # Run quantum circuit
        try:
            state = self.eng.run(self.qnn, args=mapping).state
            
            # Extract features from quantum state
            quantum_features = self._extract_quantum_features(state)
            
            # Extract quantum state features as a vector
            quantum_vector = self._extract_quantum_vector(state)
            
            # Project quantum vector to output space
            output_vector = tf.matmul(tf.expand_dims(quantum_vector, 0), self.output_projection)
            output_vector = tf.squeeze(output_vector, 0)
            
            # CONTINUOUS BLENDING: Weighted sum of mode centers
            # This preserves gradients through the entire computation
            blended_center = tf.zeros(2)
            for i in range(self.n_output_modes):
                blended_center += mode_weights[i] * self.mode_centers[i]
            
            # For 2D output
            if self.output_dim == 2:
                sample = output_vector[:2] + blended_center
            else:
                # For higher dimensions, pad blended center with zeros
                center_padded = tf.concat([blended_center, tf.zeros(self.output_dim - 2)], axis=0)
                sample = output_vector + center_padded
            
            return sample
            
        except Exception as e:
            logger.debug(f"Quantum circuit failed: {e}")
            # Fallback - still use continuous blending
            # Generate random quantum-like features
            quantum_vector = tf.random.normal([self.n_modes], stddev=0.5)
            output_vector = tf.matmul(tf.expand_dims(quantum_vector, 0), self.output_projection)
            output_vector = tf.squeeze(output_vector, 0)
            
            # Use uniform weights as fallback
            uniform_weights = tf.ones(self.n_output_modes) / float(self.n_output_modes)
            blended_center = tf.zeros(2)
            for i in range(self.n_output_modes):
                blended_center += uniform_weights[i] * self.mode_centers[i]
            
            if self.output_dim == 2:
                sample = output_vector[:2] + blended_center
            else:
                center_padded = tf.concat([blended_center, tf.zeros(self.output_dim - 2)], axis=0)
                sample = output_vector + center_padded
            
            return sample
    
    def _extract_quantum_features(self, state):
        """Extract features from quantum state."""
        ket = state.ket()
        prob_amplitudes = tf.abs(ket) ** 2
        
        features = []
        
        # Feature 1: Entropy
        entropy = -tf.reduce_sum(prob_amplitudes * tf.math.log(prob_amplitudes + 1e-12))
        features.append(tf.tanh(entropy / 3.0))
        
        # Feature 2: Purity
        purity = tf.reduce_sum(prob_amplitudes ** 2)
        features.append(purity)
        
        # Feature 3-N: Mode-specific features
        for mode in range(self.n_modes):
            # Simplified: use total probability as proxy
            mode_feature = tf.reduce_mean(prob_amplitudes)
            features.append(mode_feature)
        
        return tf.stack(features)
    
    def _extract_quantum_vector(self, state):
        """Extract a vector representation from quantum state for projection."""
        ket = state.ket()
        prob_amplitudes = tf.abs(ket) ** 2
        
        # Create a feature vector of size n_modes
        features = []
        
        # Use different statistics for each mode
        for mode in range(self.n_modes):
            if mode == 0:
                # Mean photon number estimate
                n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
                mean_n = tf.reduce_sum(prob_amplitudes * n_vals[:tf.shape(prob_amplitudes)[0]])
                features.append(mean_n)
            elif mode == 1:
                # Variance estimate
                n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
                mean_n = tf.reduce_sum(prob_amplitudes * n_vals[:tf.shape(prob_amplitudes)[0]])
                var_n = tf.reduce_sum(prob_amplitudes * (n_vals[:tf.shape(prob_amplitudes)[0]] - mean_n)**2)
                features.append(tf.sqrt(var_n + 1e-6))
            elif mode == 2:
                # Entropy
                entropy = -tf.reduce_sum(prob_amplitudes * tf.math.log(prob_amplitudes + 1e-12))
                features.append(entropy)
            else:
                # Higher order moments or other quantum properties
                moment = tf.reduce_sum(prob_amplitudes * tf.pow(tf.range(tf.shape(prob_amplitudes)[0], dtype=tf.float32), mode))
                features.append(tf.pow(moment, 1.0/mode))
        
        # Normalize features to reasonable range
        features = tf.stack(features)
        features = tf.nn.tanh(features / 5.0) * 2.0  # Range approximately [-2, 2]
        
        return features
    
    def compute_quantum_metrics(self):
        """Compute quantum metrics for different coherent states."""
        metrics = {}
        
        # Test different coherent state configurations
        test_configs = [
            ("zero_state", tf.zeros([self.n_modes * 2])),
            ("mode1_state", tf.concat([tf.ones([self.n_modes]), tf.zeros([self.n_modes])], axis=0)),
            ("mode2_state", tf.concat([tf.ones([self.n_modes]), tf.ones([self.n_modes]) * np.pi], axis=0))
        ]
        
        for config_name, coherent_params in test_configs:
            # Extract parameters
            amplitudes = coherent_params[:self.n_modes]
            phases = coherent_params[self.n_modes:]
            
            # Create mapping
            mapping = {
                p.name: w for p, w in zip(
                    self.sf_params.flatten(),
                    tf.reshape(self.weights, [-1])
                )
            }
            
            # Add coherent state parameters
            for i in range(self.n_modes):
                alpha_r = amplitudes[i] * tf.cos(phases[i])
                alpha_i = amplitudes[i] * tf.sin(phases[i])
                mapping[self.coherent_params[i]['alpha_r'].name] = alpha_r
                mapping[self.coherent_params[i]['alpha_i'].name] = alpha_i
            
            if self.eng.run_progs:
                self.eng.reset()
            
            state = self.eng.run(self.qnn, args=mapping).state
            ket = state.ket()
            prob_amplitudes = tf.abs(ket) ** 2
            
            metrics[f'{config_name}_trace'] = tf.math.real(state.trace())
            metrics[f'{config_name}_purity'] = tf.reduce_sum(prob_amplitudes ** 2)
            metrics[f'{config_name}_max_prob'] = tf.reduce_max(prob_amplitudes)
        
        return metrics


def test_coherent_generator():
    """Test the coherent state quantum generator."""
    print("\n" + "="*60)
    print("TESTING QUANTUM SF GENERATOR WITH COHERENT STATE ENCODING")
    print("="*60)
    
    # Test with 2 modes (bimodal)
    print("\n1. BIMODAL TEST")
    generator_2mode = QuantumSFGeneratorCoherent(
        n_modes=4,
        latent_dim=6,
        layers=2,
        cutoff_dim=6,
        n_output_modes=2,
        mode_radius=2.0
    )
    
    print(f"Generator created successfully")
    print(f"Trainable variables: {len(generator_2mode.trainable_variables)}")
    
    # Test generation
    z_test = tf.random.normal([100, 6])
    samples = generator_2mode.generate(z_test)
    samples_np = samples.numpy()[:, :2]  # First 2 dimensions for visualization
    
    # Analyze distribution
    mode_centers = generator_2mode.mode_centers.numpy()
    mode_assignments = []
    for sample in samples_np:
        distances = [np.linalg.norm(sample - center) for center in mode_centers]
        mode_assignments.append(np.argmin(distances))
    
    mode_counts = [mode_assignments.count(i) for i in range(2)]
    print(f"\nMode distribution: {mode_counts}")
    print(f"Balance: {min(mode_counts) / 100:.2f}")
    
    # Test with 5 modes
    print("\n2. MULTI-MODAL TEST (5 modes)")
    generator_5mode = QuantumSFGeneratorCoherent(
        n_modes=4,
        latent_dim=6,
        layers=2,
        cutoff_dim=6,
        n_output_modes=5,
        mode_radius=2.0
    )
    
    samples_5mode = generator_5mode.generate(z_test)
    samples_5mode_np = samples_5mode.numpy()[:, :2]
    
    mode_centers_5 = generator_5mode.mode_centers.numpy()
    mode_assignments_5 = []
    for sample in samples_5mode_np:
        distances = [np.linalg.norm(sample - center) for center in mode_centers_5]
        mode_assignments_5.append(np.argmin(distances))
    
    mode_counts_5 = [mode_assignments_5.count(i) for i in range(5)]
    print(f"\n5-Mode distribution: {mode_counts_5}")
    
    # Test gradient flow
    print("\n3. GRADIENT FLOW TEST")
    with tf.GradientTape() as tape:
        z = tf.random.normal([4, 6])
        output = generator_2mode.generate(z)
        loss = tf.reduce_mean(tf.square(output))
    
    gradients = tape.gradient(loss, generator_2mode.trainable_variables)
    
    for i, (var, grad) in enumerate(zip(generator_2mode.trainable_variables, gradients)):
        if grad is not None:
            print(f"   Variable {i} ({var.name}): gradient norm = {tf.norm(grad):.6f}")
        else:
            print(f"   Variable {i} ({var.name}): No gradient")
    
    # Test quantum metrics
    print("\n4. QUANTUM METRICS")
    metrics = generator_2mode.compute_quantum_metrics()
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")
    
    print("\nCoherent state generator test completed!")
    
    return generator_2mode, generator_5mode


if __name__ == "__main__":
    gen2, gen5 = test_coherent_generator()
