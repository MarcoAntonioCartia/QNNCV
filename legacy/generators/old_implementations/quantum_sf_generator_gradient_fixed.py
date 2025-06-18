"""
Quantum Strawberry Fields Generator with Fixed Gradient Flow

This generator implements the correct patterns identified in our gradient flow analysis
to ensure proper training with TensorFlow's automatic differentiation.
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import logging

logger = logging.getLogger(__name__)


class QuantumSFGeneratorGradientFixed:
    """
    Quantum generator with properly preserved gradient flow.
    
    Based on the gradient flow analysis, this implementation:
    - Keeps all operations within the computation graph
    - Uses parameter modulation approach
    - Avoids creating separate SF programs
    - No .numpy() conversions
    """
    
    def __init__(self, n_modes=4, latent_dim=6, layers=2, cutoff_dim=6):
        """Initialize quantum generator with gradient-preserving architecture."""
        self.n_modes = n_modes
        self.latent_dim = latent_dim
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        
        logger.info(f"Initializing Gradient-Fixed Quantum SF Generator:")
        logger.info(f"  - n_modes: {n_modes}")
        logger.info(f"  - latent_dim: {latent_dim}")
        logger.info(f"  - layers: {layers}")
        logger.info(f"  - cutoff_dim: {cutoff_dim}")
        
        # Initialize SF components
        self._init_sf_components()
        
        # Initialize quantum weights using proven approach
        self._init_quantum_weights()
        
        # Initialize latent encoder
        self._init_latent_encoder()
        
        # Create symbolic parameters
        self._create_symbolic_params()
        
        # Build quantum program ONCE
        self._build_quantum_program()
    
    def _init_sf_components(self):
        """Initialize Strawberry Fields engine and program."""
        self.eng = sf.Engine(backend="tf", backend_options={
            "cutoff_dim": self.cutoff_dim,
            "pure": True
        })
        # Single program - never create another
        self.prog = sf.Program(self.n_modes)
        logger.info("SF engine and single program initialized")
    
    def _init_quantum_weights(self):
        """Initialize quantum weights as proven in gradient tests."""
        # Calculate parameter count
        M = int(self.n_modes * (self.n_modes - 1)) + max(1, self.n_modes - 1)
        params_per_layer = 2 * M + 4 * self.n_modes
        total_params = self.layers * params_per_layer
        
        # Base quantum parameters (as tensor - proven to work)
        self.base_params = tf.Variable(
            tf.random.normal([total_params], stddev=0.1),
            name="base_quantum_params"
        )
        
        # Modulation parameters for input encoding
        self.modulation_params = tf.Variable(
            tf.random.normal([self.latent_dim, total_params], stddev=0.01),
            name="modulation_params"
        )
        
        self.num_quantum_params = total_params
        logger.info(f"Quantum parameters initialized: {total_params} params")
    
    def _init_latent_encoder(self):
        """Initialize latent to modulation encoder."""
        # Simple linear encoder for modulation weights
        self.encoder_weights = tf.Variable(
            tf.random.normal([self.latent_dim, self.num_quantum_params], stddev=0.1),
            name="encoder_weights"
        )
        
        self.encoder_bias = tf.Variable(
            tf.zeros([self.num_quantum_params]),
            name="encoder_bias"
        )
    
    def _create_symbolic_params(self):
        """Create symbolic parameters for the quantum program."""
        # Create flat array of symbolic parameters
        self.sym_params = []
        for i in range(self.num_quantum_params):
            self.sym_params.append(self.prog.params(f"p{i}"))
        
        # Reshape for layer structure
        M = int(self.n_modes * (self.n_modes - 1)) + max(1, self.n_modes - 1)
        params_per_layer = 2 * M + 4 * self.n_modes
        
        self.layer_params = []
        for layer in range(self.layers):
            start_idx = layer * params_per_layer
            layer_params = self.sym_params[start_idx:start_idx + params_per_layer]
            self.layer_params.append(layer_params)
    
    def _build_quantum_program(self):
        """Build the quantum program once with symbolic parameters."""
        with self.prog.context as q:
            # Apply quantum layers
            for layer_idx in range(self.layers):
                self._quantum_layer(self.layer_params[layer_idx], q)
        
        logger.info(f"Quantum program built with {self.layers} layers")
    
    def _quantum_layer(self, params, q):
        """Single quantum layer with symbolic parameters."""
        N = self.n_modes
        M = int(N * (N - 1)) + max(1, N - 1)
        
        # Extract parameters for each component
        idx = 0
        int1_params = params[idx:idx+M]
        idx += M
        squeeze_params = params[idx:idx+N]
        idx += N
        int2_params = params[idx:idx+M]
        idx += M
        disp_r_params = params[idx:idx+N]
        idx += N
        disp_phi_params = params[idx:idx+N]
        idx += N
        kerr_params = params[idx:idx+N]
        
        # Apply operations
        self._interferometer(int1_params, q)
        
        for i in range(N):
            ops.Sgate(squeeze_params[i]) | q[i]
        
        self._interferometer(int2_params, q)
        
        for i in range(N):
            ops.Dgate(disp_r_params[i], disp_phi_params[i]) | q[i]
            ops.Kgate(kerr_params[i]) | q[i]
    
    def _interferometer(self, params, q):
        """Interferometer with symbolic parameters."""
        N = len(q)
        
        if N == 1:
            # Single mode rotation
            ops.Rgate(params[0]) | q[0]
            return
        
        # Multi-mode interferometer
        n_theta = N * (N - 1) // 2
        n_phi = N * (N - 1) // 2
        n_rphi = max(1, N - 1)
        
        theta = params[:n_theta]
        phi = params[n_theta:n_theta + n_phi]
        rphi = params[n_theta + n_phi:]
        
        # Apply beamsplitter array
        param_idx = 0
        for l in range(N):
            for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
                if (l + k) % 2 != 1:
                    ops.BSgate(theta[param_idx], phi[param_idx]) | (q1, q2)
                    param_idx += 1
        
        # Apply final rotations
        for i in range(len(rphi)):
            ops.Rgate(rphi[i]) | q[i]
    
    @property
    def trainable_variables(self):
        """Return all trainable variables."""
        return [
            self.base_params,
            self.modulation_params,
            self.encoder_weights,
            self.encoder_bias
        ]
    
    def encode_latent(self, z):
        """Encode latent vector to parameter modulation."""
        # Use both encoder weights AND modulation_params
        # First, get base encoding
        base_encoding = tf.matmul(z, self.encoder_weights) + self.encoder_bias
        
        # Then modulate with modulation_params
        modulation_contribution = tf.matmul(z, self.modulation_params)
        
        # Combine both
        modulation = base_encoding + 0.1 * modulation_contribution
        modulation = tf.nn.tanh(modulation) * 0.1  # Small modulation
        return modulation
    
    def generate(self, z):
        """
        Generate samples with proper gradient flow.
        
        This is the key method - all quantum operations must be
        within the gradient computation context.
        """
        batch_size = tf.shape(z)[0]
        all_samples = []
        
        # Process each sample
        for i in range(batch_size):
            # Get modulation for this sample
            modulation = self.encode_latent(tf.expand_dims(z[i], 0))
            modulation = tf.squeeze(modulation, 0)
            
            # Combine base parameters with modulation
            # This is the proven approach from our tests
            quantum_params = self.base_params + modulation
            
            # Create parameter mapping
            mapping = {
                self.sym_params[j].name: quantum_params[j] 
                for j in range(self.num_quantum_params)
            }
            
            # Reset engine
            if self.eng.run_progs:
                self.eng.reset()
            
            # Run quantum circuit
            state = self.eng.run(self.prog, args=mapping).state
            
            # Extract sample from quantum state
            sample = self._extract_sample(state)
            all_samples.append(sample)
        
        return tf.stack(all_samples, axis=0)
    
    def _extract_sample(self, state):
        """Extract classical sample from quantum state."""
        ket = state.ket()
        prob_amplitudes = tf.abs(ket) ** 2
        
        # Extract different features for each output dimension
        features = []
        
        # Mean photon number
        n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
        mean_n = tf.reduce_sum(prob_amplitudes * n_vals[:tf.shape(prob_amplitudes)[0]])
        features.append(mean_n)
        
        # Variance
        var_n = tf.reduce_sum(prob_amplitudes * (n_vals[:tf.shape(prob_amplitudes)[0]] - mean_n)**2)
        features.append(tf.sqrt(var_n + 1e-6))
        
        # Add more features if needed
        while len(features) < self.n_modes:
            # Entropy
            entropy = -tf.reduce_sum(prob_amplitudes * tf.math.log(prob_amplitudes + 1e-12))
            features.append(entropy)
            
            # Max probability
            if len(features) < self.n_modes:
                max_prob = tf.reduce_max(prob_amplitudes)
                features.append(max_prob * 10.0)
        
        # Stack and scale features
        output = tf.stack(features[:self.n_modes])
        output = tf.nn.tanh(output / 3.0) * 3.0
        
        return output


def test_gradient_fixed_generator():
    """Test the gradient-fixed generator."""
    print("\n" + "="*60)
    print("TESTING GRADIENT-FIXED QUANTUM SF GENERATOR")
    print("="*60)
    
    # Create generator
    generator = QuantumSFGeneratorGradientFixed(
        n_modes=4,
        latent_dim=6,
        layers=2,
        cutoff_dim=6
    )
    
    print(f"\nGenerator created successfully")
    print(f"Trainable variables: {len(generator.trainable_variables)}")
    
    # Test generation
    print("\n1. GENERATION TEST")
    z_test = tf.random.normal([4, 6])
    samples = generator.generate(z_test)
    print(f"   Generated samples shape: {samples.shape}")
    print(f"   Sample range: [{tf.reduce_min(samples):.3f}, {tf.reduce_max(samples):.3f}]")
    
    # Test gradient flow
    print("\n2. GRADIENT FLOW TEST")
    with tf.GradientTape() as tape:
        z = tf.random.normal([2, 6])
        output = generator.generate(z)
        loss = tf.reduce_mean(tf.square(output))
    
    gradients = tape.gradient(loss, generator.trainable_variables)
    
    print(f"\n   Loss: {loss:.6f}")
    print("\n   Gradient norms:")
    for i, (var, grad) in enumerate(zip(generator.trainable_variables, gradients)):
        if grad is not None:
            grad_norm = tf.norm(grad)
            print(f"   {var.name}: {grad_norm:.6e}")
        else:
            print(f"   {var.name}: NO GRADIENT ❌")
    
    # Check if all variables have gradients
    all_have_gradients = all(g is not None for g in gradients)
    
    if all_have_gradients:
        print("\n✅ SUCCESS: All variables have gradients!")
    else:
        print("\n❌ FAILURE: Some variables missing gradients")
    
    # Test with bimodal target
    print("\n3. BIMODAL GENERATION TEST")
    # Generate many samples
    z_large = tf.random.normal([100, 6])
    samples_large = generator.generate(z_large)
    
    # Simple bimodal check - see if we get variation
    samples_2d = samples_large[:, :2].numpy()
    mean_sample = np.mean(samples_2d, axis=0)
    std_sample = np.std(samples_2d, axis=0)
    
    print(f"   Mean: {mean_sample}")
    print(f"   Std: {std_sample}")
    print(f"   Sample diversity: {'Good' if np.min(std_sample) > 0.5 else 'Poor'}")
    
    return generator, all_have_gradients


if __name__ == "__main__":
    generator, success = test_gradient_fixed_generator()
