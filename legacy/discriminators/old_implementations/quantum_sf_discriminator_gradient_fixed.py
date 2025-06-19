"""
Quantum Strawberry Fields Discriminator with Fixed Gradient Flow

This discriminator implements the same gradient-preserving patterns as the generator
to ensure proper training with TensorFlow's automatic differentiation.
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import logging

logger = logging.getLogger(__name__)


class QuantumSFDiscriminatorGradientFixed:
    """
    Quantum discriminator with properly preserved gradient flow.
    
    Based on the gradient flow analysis, this implementation:
    - Keeps all operations within the computation graph
    - Uses parameter modulation approach
    - Avoids creating separate SF programs
    - No .numpy() conversions
    """
    
    def __init__(self, n_modes=2, input_dim=2, layers=2, cutoff_dim=6):
        """Initialize quantum discriminator with gradient-preserving architecture."""
        self.n_modes = n_modes
        self.input_dim = input_dim
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        
        logger.info(f"Initializing Gradient-Fixed Quantum SF Discriminator:")
        logger.info(f"  - n_modes: {n_modes}")
        logger.info(f"  - input_dim: {input_dim}")
        logger.info(f"  - layers: {layers}")
        logger.info(f"  - cutoff_dim: {cutoff_dim}")
        
        # Initialize SF components
        self._init_sf_components()
        
        # Initialize quantum weights using proven approach
        self._init_quantum_weights()
        
        # Initialize input encoder
        self._init_input_encoder()
        
        # Initialize output processor
        self._init_output_processor()
        
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
        logger.info("SF discriminator engine and single program initialized")
    
    def _init_quantum_weights(self):
        """Initialize quantum weights as proven in gradient tests."""
        # Calculate parameter count
        M = int(self.n_modes * (self.n_modes - 1)) + max(1, self.n_modes - 1)
        params_per_layer = 2 * M + 4 * self.n_modes
        total_params = self.layers * params_per_layer
        
        # Base quantum parameters (as tensor - proven to work)
        self.base_params = tf.Variable(
            tf.random.normal([total_params], stddev=0.1),
            name="disc_base_quantum_params"
        )
        
        # Modulation parameters for input encoding
        self.modulation_params = tf.Variable(
            tf.random.normal([self.input_dim, total_params], stddev=0.01),
            name="disc_modulation_params"
        )
        
        self.num_quantum_params = total_params
        logger.info(f"Discriminator quantum parameters initialized: {total_params} params")
    
    def _init_input_encoder(self):
        """Initialize input to modulation encoder."""
        # Simple linear encoder for modulation weights
        self.encoder_weights = tf.Variable(
            tf.random.normal([self.input_dim, self.num_quantum_params], stddev=0.1),
            name="disc_encoder_weights"
        )
        
        self.encoder_bias = tf.Variable(
            tf.zeros([self.num_quantum_params]),
            name="disc_encoder_bias"
        )
    
    def _init_output_processor(self):
        """Initialize output processing network."""
        # Process quantum features to discrimination probability
        self.output_weights = tf.Variable(
            tf.random.normal([self.n_modes, 4], stddev=0.5),
            name="disc_output_weights"
        )
        
        self.output_bias = tf.Variable(
            tf.zeros([4]),
            name="disc_output_bias"
        )
        
        self.final_weights = tf.Variable(
            tf.random.normal([4, 1], stddev=0.5),
            name="disc_final_weights"
        )
        
        self.final_bias = tf.Variable(
            tf.zeros([1]),
            name="disc_final_bias"
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
        
        logger.info(f"Discriminator quantum program built with {self.layers} layers")
    
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
            self.encoder_bias,
            self.output_weights,
            self.output_bias,
            self.final_weights,
            self.final_bias
        ]
    
    def encode_input(self, x):
        """Encode input to parameter modulation."""
        # Use both encoder weights AND modulation_params
        # First, get base encoding
        base_encoding = tf.matmul(x, self.encoder_weights) + self.encoder_bias
        
        # Then modulate with modulation_params
        modulation_contribution = tf.matmul(x, self.modulation_params)
        
        # Combine both
        modulation = base_encoding + 0.1 * modulation_contribution
        modulation = tf.nn.tanh(modulation) * 0.1  # Small modulation
        return modulation
    
    def discriminate(self, x):
        """
        Discriminate samples with proper gradient flow.
        
        This is the key method - all quantum operations must be
        within the gradient computation context.
        """
        batch_size = tf.shape(x)[0]
        all_features = []
        
        # Process each sample
        for i in range(batch_size):
            # Get modulation for this sample
            modulation = self.encode_input(tf.expand_dims(x[i], 0))
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
            
            # Extract features from quantum state
            features = self._extract_features(state)
            all_features.append(features)
        
        # Stack features
        quantum_features = tf.stack(all_features, axis=0)
        
        # Process through output network
        hidden = tf.matmul(quantum_features, self.output_weights) + self.output_bias
        hidden = tf.nn.relu(hidden)
        
        output = tf.matmul(hidden, self.final_weights) + self.final_bias
        probabilities = tf.nn.sigmoid(output)
        
        return probabilities
    
    def _extract_features(self, state):
        """Extract features from quantum state."""
        ket = state.ket()
        prob_amplitudes = tf.abs(ket) ** 2
        
        # Extract different features for each mode
        features = []
        
        # Mean photon number
        n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
        mean_n = tf.reduce_sum(prob_amplitudes * n_vals[:tf.shape(prob_amplitudes)[0]])
        features.append(mean_n / self.cutoff_dim)
        
        # Variance
        var_n = tf.reduce_sum(prob_amplitudes * (n_vals[:tf.shape(prob_amplitudes)[0]] - mean_n)**2)
        features.append(tf.tanh(var_n / self.cutoff_dim))
        
        # Add more features if needed
        while len(features) < self.n_modes:
            # Entropy
            entropy = -tf.reduce_sum(prob_amplitudes * tf.math.log(prob_amplitudes + 1e-12))
            features.append(tf.tanh(entropy))
            
            # Max probability
            if len(features) < self.n_modes:
                max_prob = tf.reduce_max(prob_amplitudes)
                features.append(max_prob)
        
        return tf.stack(features[:self.n_modes])


def test_gradient_fixed_discriminator():
    """Test the gradient-fixed discriminator."""
    print("\n" + "="*60)
    print("TESTING GRADIENT-FIXED QUANTUM SF DISCRIMINATOR")
    print("="*60)
    
    # Create discriminator
    discriminator = QuantumSFDiscriminatorGradientFixed(
        n_modes=2,
        input_dim=2,
        layers=2,
        cutoff_dim=6
    )
    
    print(f"\nDiscriminator created successfully")
    print(f"Trainable variables: {len(discriminator.trainable_variables)}")
    
    # Test discrimination
    print("\n1. DISCRIMINATION TEST")
    x_test = tf.random.normal([4, 2])
    probs = discriminator.discriminate(x_test)
    print(f"   Discrimination output shape: {probs.shape}")
    print(f"   Probability range: [{tf.reduce_min(probs):.3f}, {tf.reduce_max(probs):.3f}]")
    
    # Test gradient flow
    print("\n2. GRADIENT FLOW TEST")
    with tf.GradientTape() as tape:
        x = tf.random.normal([2, 2])
        output = discriminator.discriminate(x)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(output),
            logits=output
        ))
    
    gradients = tape.gradient(loss, discriminator.trainable_variables)
    
    print(f"\n   Loss: {loss:.6f}")
    print("\n   Gradient norms:")
    for i, (var, grad) in enumerate(zip(discriminator.trainable_variables, gradients)):
        if grad is not None:
            grad_norm = tf.norm(grad)
            print(f"   {var.name}: {grad_norm:.6e}")
        else:
            print(f"   {var.name}: NO GRADIENT ❌")
    
    # Check if all variables have gradients
    all_have_gradients = all(g is not None for g in gradients)
    
    if all_have_gradients:
        print("\n✅ SUCCESS: All discriminator variables have gradients!")
    else:
        print("\n❌ FAILURE: Some discriminator variables missing gradients")
    
    return discriminator, all_have_gradients


if __name__ == "__main__":
    discriminator, success = test_gradient_fixed_discriminator()
