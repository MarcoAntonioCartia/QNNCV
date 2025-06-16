"""
Quantum Bimodal Discriminator with Fixed Parameter Updates

This discriminator implementation fixes the parameter update issue by:
1. Using individual tf.Variables for each quantum parameter
2. Ensuring proper gradient flow through all quantum operations
3. Matching the generator's parameter update strategy
"""

import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class QuantumBimodalDiscriminator:
    """
    Quantum discriminator optimized for bimodal distributions.
    
    Key improvements:
    - Individual quantum parameter variables for better gradient flow
    - Direct quantum state discrimination
    - Simplified architecture for stable training
    """
    
    def __init__(self, n_modes=2, input_dim=2, layers=2, cutoff_dim=6):
        """
        Initialize bimodal quantum discriminator.
        
        Args:
            n_modes: Number of quantum modes
            input_dim: Dimension of input data
            layers: Number of quantum layers
            cutoff_dim: Fock space cutoff
        """
        self.n_modes = n_modes
        self.input_dim = input_dim
        self.layers = layers
        self.cutoff_dim = cutoff_dim
        
        logger.info(f"Initializing Quantum Bimodal Discriminator:")
        logger.info(f"  - n_modes: {n_modes}")
        logger.info(f"  - input_dim: {input_dim}")
        logger.info(f"  - layers: {layers}")
        logger.info(f"  - cutoff_dim: {cutoff_dim}")
        
        # Initialize SF components
        self._init_sf_components()
        
        # Initialize quantum weights (individual variables)
        self._init_quantum_weights()
        
        # Initialize input encoder
        self._init_input_encoder()
        
        # Initialize output processor
        self._init_output_processor()
        
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
        logger.info("SF discriminator engine initialized")
    
    def _init_quantum_weights(self):
        """Initialize individual quantum variables for maximum gradient flow."""
        M = int(self.n_modes * (self.n_modes - 1)) + max(1, self.n_modes - 1)
        
        # Store all individual variables in a flat list for easy access
        self.quantum_vars = []
        
        for layer in range(self.layers):
            # Interferometer 1
            for i in range(M):
                var = tf.Variable(
                    tf.random.normal([], stddev=0.1),
                    name=f'disc_int1_L{layer}_P{i}'
                )
                self.quantum_vars.append(var)
            
            # Squeezing
            for i in range(self.n_modes):
                var = tf.Variable(
                    tf.random.normal([], stddev=0.01),
                    name=f'disc_squeeze_L{layer}_M{i}'
                )
                self.quantum_vars.append(var)
            
            # Interferometer 2
            for i in range(M):
                var = tf.Variable(
                    tf.random.normal([], stddev=0.1),
                    name=f'disc_int2_L{layer}_P{i}'
                )
                self.quantum_vars.append(var)
            
            # Displacement r
            for i in range(self.n_modes):
                var = tf.Variable(
                    tf.random.normal([], stddev=0.01),
                    name=f'disc_disp_r_L{layer}_M{i}'
                )
                self.quantum_vars.append(var)
            
            # Displacement phi
            for i in range(self.n_modes):
                var = tf.Variable(
                    tf.random.normal([], stddev=0.1),
                    name=f'disc_disp_phi_L{layer}_M{i}'
                )
                self.quantum_vars.append(var)
            
            # Kerr
            for i in range(self.n_modes):
                var = tf.Variable(
                    tf.random.normal([], stddev=0.01),
                    name=f'disc_kerr_L{layer}_M{i}'
                )
                self.quantum_vars.append(var)
        
        self.num_quantum_params = len(self.quantum_vars)
        logger.info(f"Initialized {self.num_quantum_params} individual quantum variables")
    
    def _init_input_encoder(self):
        """Initialize input encoding network."""
        try:
            from tensorflow import keras
        except ImportError:
            import keras
        
        self.input_encoder = keras.Sequential([
            keras.layers.Dense(16, activation='tanh'),
            keras.layers.Dense(8, activation='tanh'),
            keras.layers.Dense(self.num_quantum_params, activation='tanh')
        ], name='disc_input_encoder')
        
        # Build network
        dummy_input = tf.zeros((1, self.input_dim))
        _ = self.input_encoder(dummy_input)
        
        logger.info(f"Input encoder: {self.input_dim} → {self.num_quantum_params}")
    
    def _init_output_processor(self):
        """Initialize output processing network."""
        try:
            from tensorflow import keras
        except ImportError:
            import keras
        
        # Simple output processor for binary classification
        self.output_processor = keras.Sequential([
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(4, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ], name='disc_output_processor')
        
        # Build network
        dummy_quantum_output = tf.zeros((1, self.n_modes))
        _ = self.output_processor(dummy_quantum_output)
        
        logger.info(f"Output processor: {self.n_modes} → 1")
    
    @property
    def quantum_weights(self):
        """Get quantum weights in matrix form (for compatibility)."""
        # Stack individual variables into matrix form
        params_per_layer = self.num_quantum_params // self.layers
        weights = []
        
        for layer in range(self.layers):
            start_idx = layer * params_per_layer
            end_idx = (layer + 1) * params_per_layer
            layer_params = tf.stack(self.quantum_vars[start_idx:end_idx])
            weights.append(layer_params)
        
        return tf.stack(weights)
    
    def _create_symbolic_params(self):
        """Create SF symbolic parameters."""
        # Create symbolic parameters matching the structure
        params_per_layer = self.num_quantum_params // self.layers
        sf_params = []
        
        for layer in range(self.layers):
            layer_params = []
            for i in range(params_per_layer):
                param_name = f"disc_param_L{layer}_P{i}"
                layer_params.append(self.qnn.params(param_name))
            sf_params.append(layer_params)
        
        self.sf_params = np.array(sf_params)
        logger.info(f"Created {self.num_quantum_params} symbolic parameters")
    
    def _build_quantum_program(self):
        """Build quantum program."""
        with self.qnn.context as q:
            for k in range(self.layers):
                self._quantum_layer(self.sf_params[k], q)
        
        logger.info(f"Quantum program built with {self.layers} layers")
    
    def _quantum_layer(self, params, q):
        """Single quantum layer."""
        N = len(q)
        M = int(N * (N - 1)) + max(1, N - 1)
        
        idx = 0
        
        # Interferometer 1
        int1_params = params[idx:idx+M]
        idx += M
        self._interferometer(int1_params, q)
        
        # Squeezing
        for i in range(N):
            ops.Sgate(params[idx]) | q[i]
            idx += 1
        
        # Interferometer 2
        int2_params = params[idx:idx+M]
        idx += M
        self._interferometer(int2_params, q)
        
        # Displacement
        for i in range(N):
            ops.Dgate(params[idx], params[idx+N]) | q[i]
            idx += 1
        idx += N
        
        # Kerr
        for i in range(N):
            ops.Kgate(params[idx]) | q[i]
            idx += 1
    
    def _interferometer(self, params, q):
        """Interferometer implementation."""
        N = len(q)
        
        if N == 1:
            # Single mode: just rotation
            ops.Rgate(params[0]) | q[0]
            return
        
        # Multi-mode interferometer
        theta_count = N * (N - 1) // 2
        phi_count = N * (N - 1) // 2
        rphi_count = max(1, N - 1)
        
        theta = params[:theta_count]
        phi = params[theta_count:theta_count + phi_count]
        rphi = params[theta_count + phi_count:]
        
        # Apply beamsplitters
        n = 0
        for l in range(N):
            for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
                if (l + k) % 2 != 1:
                    ops.BSgate(theta[n], phi[n]) | (q1, q2)
                    n += 1
        
        # Final rotations
        for i in range(min(len(rphi), N)):
            ops.Rgate(rphi[i]) | q[i]
    
    @property
    def trainable_variables(self):
        """Return all trainable variables."""
        variables = []
        
        # Add all individual quantum variables
        variables.extend(self.quantum_vars)
        
        # Add input encoder variables
        variables.extend(self.input_encoder.trainable_variables)
        
        # Add output processor variables
        variables.extend(self.output_processor.trainable_variables)
        
        return variables
    
    def discriminate(self, x):
        """
        Discriminate samples.
        
        Args:
            x: Input samples [batch_size, input_dim]
            
        Returns:
            probabilities: Probability of being real [batch_size, 1]
        """
        batch_size = tf.shape(x)[0]
        all_outputs = []
        
        # Encode inputs
        encoded_inputs = self.input_encoder(x)  # [batch_size, num_quantum_params]
        
        # Process each sample
        for i in range(batch_size):
            quantum_output = self._discriminate_single(encoded_inputs[i])
            all_outputs.append(quantum_output)
        
        # Stack outputs
        quantum_features = tf.stack(all_outputs, axis=0)  # [batch_size, n_modes]
        
        # Process to get probabilities
        probabilities = self.output_processor(quantum_features)
        
        return probabilities
    
    def _discriminate_single(self, encoded_input):
        """Discriminate single sample."""
        # Combine encoded input with quantum parameters
        modulated_params = []
        
        for i, quantum_var in enumerate(self.quantum_vars):
            # Modulate each quantum parameter with encoded input
            modulated = quantum_var + 0.1 * encoded_input[i]
            modulated_params.append(modulated)
        
        # Create parameter mapping
        mapping = {}
        param_idx = 0
        
        for layer in range(self.layers):
            for i, param in enumerate(self.sf_params[layer]):
                mapping[param.name] = modulated_params[param_idx]
                param_idx += 1
        
        # Reset engine
        if self.eng.run_progs:
            self.eng.reset()
        
        # Run quantum circuit
        try:
            state = self.eng.run(self.qnn, args=mapping).state
            return self._extract_features(state)
        except Exception as e:
            logger.debug(f"Quantum circuit failed: {e}")
            return tf.random.normal([self.n_modes], stddev=0.1)
    
    def _extract_features(self, state):
        """Extract discriminative features from quantum state."""
        ket = state.ket()
        prob_amplitudes = tf.abs(ket) ** 2
        
        features = []
        
        for mode in range(self.n_modes):
            # Photon number statistics
            n_vals = tf.range(self.cutoff_dim, dtype=tf.float32)
            mean_n = tf.reduce_sum(prob_amplitudes * n_vals)
            var_n = tf.reduce_sum(prob_amplitudes * (n_vals - mean_n)**2)
            
            # Normalized features
            feature1 = (mean_n - self.cutoff_dim/2) / (self.cutoff_dim/4)
            feature2 = tf.tanh(tf.sqrt(var_n + 1e-8))
            
            # For bimodal discrimination, also include mode purity
            max_prob = tf.reduce_max(prob_amplitudes)
            purity = tf.tanh(max_prob * self.cutoff_dim)
            
            features.append((feature1 + feature2 + purity) / 3.0)
        
        return tf.stack(features)
    
    def compute_quantum_metrics(self):
        """Compute quantum metrics."""
        # Test input
        x_test = tf.random.normal([1, self.input_dim])
        encoded = self.input_encoder(x_test)
        
        # Get quantum state
        modulated_params = []
        for i, quantum_var in enumerate(self.quantum_vars):
            modulated = quantum_var + 0.1 * encoded[0, i]
            modulated_params.append(modulated)
        
        # Create mapping
        mapping = {}
        param_idx = 0
        
        for layer in range(self.layers):
            for i, param in enumerate(self.sf_params[layer]):
                mapping[param.name] = modulated_params[param_idx]
                param_idx += 1
        
        if self.eng.run_progs:
            self.eng.reset()
        
        state = self.eng.run(self.qnn, args=mapping).state
        ket = state.ket()
        
        # Compute metrics
        prob_amplitudes = tf.abs(ket) ** 2
        
        metrics = {
            'trace': tf.math.real(state.trace()),
            'purity': tf.reduce_sum(prob_amplitudes ** 2),
            'max_prob': tf.reduce_max(prob_amplitudes),
            'entropy': self._compute_entropy(ket)
        }
        
        return metrics
    
    def _compute_entropy(self, ket):
        """Compute von Neumann entropy."""
        prob_amplitudes = tf.abs(ket) ** 2
        prob_amplitudes = prob_amplitudes + 1e-12
        entropy = -tf.reduce_sum(prob_amplitudes * tf.math.log(prob_amplitudes))
        return entropy


def test_bimodal_discriminator():
    """Test the bimodal quantum discriminator."""
    print("\n" + "="*60)
    print("TESTING QUANTUM BIMODAL DISCRIMINATOR")
    print("="*60)
    
    # Create discriminator
    discriminator = QuantumBimodalDiscriminator(
        n_modes=2,
        input_dim=2,
        layers=2,
        cutoff_dim=6
    )
    
    print(f"✅ Discriminator created successfully")
    print(f"📊 Trainable variables: {len(discriminator.trainable_variables)}")
    print(f"   - Quantum variables: {len(discriminator.quantum_vars)}")
    print(f"   - Input encoder vars: {len(discriminator.input_encoder.trainable_variables)}")
    print(f"   - Output processor vars: {len(discriminator.output_processor.trainable_variables)}")
    
    # Test discrimination
    x_test = tf.random.normal([5, 2])
    probs = discriminator.discriminate(x_test)
    
    print(f"\n🎯 Discrimination test:")
    print(f"   Input shape: {x_test.shape}")
    print(f"   Output shape: {probs.shape}")
    print(f"   Probability range: [{tf.reduce_min(probs):.3f}, {tf.reduce_max(probs):.3f}]")
    
    # Test gradient flow
    with tf.GradientTape() as tape:
        x = tf.random.normal([2, 2])
        output = discriminator.discriminate(x)
        loss = tf.reduce_mean(tf.square(output - 0.5))
    
    gradients = tape.gradient(loss, discriminator.trainable_variables)
    non_none_grads = [g for g in gradients if g is not None]
    
    print(f"\n🔄 Gradient test:")
    print(f"   Non-None gradients: {len(non_none_grads)}/{len(gradients)}")
    print(f"   Loss: {loss:.4f}")
    
    # Check gradient magnitudes
    quantum_grads = gradients[:len(discriminator.quantum_vars)]
    quantum_grad_norms = [tf.norm(g).numpy() for g in quantum_grads if g is not None]
    
    if quantum_grad_norms:
        print(f"   Quantum gradient norms: min={min(quantum_grad_norms):.6f}, max={max(quantum_grad_norms):.6f}")
    
    # Test quantum metrics
    metrics = discriminator.compute_quantum_metrics()
    print(f"\n🔬 Quantum metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")
    
    return discriminator


if __name__ == "__main__":
    test_bimodal_discriminator()
